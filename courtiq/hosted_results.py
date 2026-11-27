"""Read published result snapshots without calling the tennis data provider.

The publisher owns durable state, pagination, and the SportStats request budget.
This module reads a versioned, allowlisted projection of that existing cache.
The optional in-process copy only saves HTTP reads; it is not durable storage.

``fetcher(url, headers, timeout_seconds, max_bytes) -> bytes`` is an optional
transport seam, including for an authenticated storage SDK. It must enforce
its timeout and download limit. The built-in transport never follows redirects.
Neither transport errors nor credentials are included in public status.
"""

import copy
import hashlib
from http.client import HTTPSConnection
import ipaddress
import json
import math
import os
import re
import threading
import time
from datetime import date, datetime, timezone
from urllib.parse import urlsplit

from courtiq.sportstats import (
    CACHE_VERSION, RECENT_TTL, SCORE_SCHEMA_VERSION, SOURCE_URL,
    SportStatsResults, _covers, _day, _merge_ranges,
)


SNAPSHOT_FORMAT = "courtiq-results-snapshot"
SNAPSHOT_VERSION = 1
MAX_SNAPSHOT_BYTES = 16 * 1024 * 1024
MAX_RECORDS = 30000
MAX_RANGES = 1000
UTC = timezone.utc


def _timestamp(value, nullable=False):
    if value is None and nullable:
        return None
    result = SportStatsResults._valid_timestamp(value)
    if result is None:
        raise ValueError("Invalid snapshot timestamp.")
    return result


def _seconds(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _iso(value):
    return datetime.fromtimestamp(value, UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _finite_number(value, maximum=None, nullable=False):
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Invalid snapshot number.")
    if not math.isfinite(value) or value < 0 or (maximum is not None and value > maximum):
        raise ValueError("Invalid snapshot number.")
    return value


def _public_cache(cache, published_at):
    """Validate all records before accepting any coverage claims or replacing data."""
    if not isinstance(cache, dict):
        raise ValueError("Invalid results cache.")
    for field, expected in (("version", CACHE_VERSION), ("score_schema_version", SCORE_SCHEMA_VERSION)):
        if type(cache.get(field)) is not int or cache[field] != expected:
            raise ValueError("Unsupported results cache schema.")
    revision = cache.get("revision")
    if type(revision) is not int or not 0 <= revision <= 1000000000:
        raise ValueError("Invalid cache revision.")
    published_seconds = _seconds(published_at)
    published_day = datetime.fromtimestamp(published_seconds, UTC).date()
    rows, ranges = cache.get("records"), cache.get("covered_ranges")
    if not isinstance(rows, list) or len(rows) > MAX_RECORDS:
        raise ValueError("Invalid snapshot records.")
    if not isinstance(ranges, list) or len(ranges) > MAX_RANGES:
        raise ValueError("Invalid snapshot coverage.")
    records, identifiers = [], set()
    for row in rows:
        clean = SportStatsResults._cached_record(row)
        if clean is None or clean["match_id"] in identifiers:
            raise ValueError("Invalid or duplicate snapshot record.")
        if (_day(clean["match_date"]) > published_day
                or _seconds(clean["observed_at"]) > published_seconds + 300
                or (clean["score_observed_at"]
                    and _seconds(clean["score_observed_at"]) > published_seconds + 300)):
            raise ValueError("Snapshot contains a future result.")
        identifiers.add(clean["match_id"])
        records.append(clean)
    parsed_ranges = []
    for item in ranges:
        if not isinstance(item, dict):
            raise ValueError("Invalid snapshot coverage.")
        start, end = _day(item.get("start")), _day(item.get("end"))
        if start is None or end is None or not start <= end <= published_day:
            raise ValueError("Invalid snapshot coverage.")
        parsed_ranges.append((start, end))
    last_sync = _timestamp(cache.get("last_sync"), nullable=True)
    if last_sync and _seconds(last_sync) > published_seconds + 300:
        raise ValueError("Invalid snapshot sync time.")
    if (records or parsed_ranges) and last_sync is None:
        raise ValueError("Populated snapshot is missing its sync time.")
    recent_refresh = _finite_number(cache.get("recent_refresh"), nullable=True)
    if recent_refresh is not None and recent_refresh > published_seconds + 300:
        raise ValueError("Invalid snapshot refresh time.")
    recent_range = cache.get("recent_range")
    if recent_range is not None:
        if not isinstance(recent_range, str):
            raise ValueError("Invalid recent coverage range.")
        parts = recent_range.split("/")
        if len(parts) != 2:
            raise ValueError("Invalid recent coverage range.")
        start, end = (_day(part) for part in parts)
        if start is None or end is None or not start <= end <= published_day:
            raise ValueError("Invalid recent coverage range.")
    data_delay = cache.get("data_delay_seconds")
    if data_delay is None and isinstance(cache.get("quota"), dict):
        data_delay = cache["quota"].get("data_delay")
    data_delay = _finite_number(data_delay, maximum=86400, nullable=True)
    # Reparse scores and explicitly construct every public field. No raw
    # response, checkpoint, quota object, token, or arbitrary metadata survives.
    return {
        "version": CACHE_VERSION, "score_schema_version": SCORE_SCHEMA_VERSION,
        "revision": revision,
        "records": sorted(records, key=lambda row: (row["match_date"], row["match_id"]), reverse=True),
        "covered_ranges": [{"start": a.isoformat(), "end": b.isoformat()}
                           for a, b in _merge_ranges(parsed_ranges)],
        "last_sync": last_sync, "recent_refresh": recent_refresh,
        "recent_range": recent_range, "data_delay_seconds": data_delay,
    }


def serialize_public_snapshot(cache, published_at):
    """Return UTF-8 JSON bytes safe to publish from a validated provider cache."""
    published_at = _timestamp(published_at)
    envelope = {
        "format": SNAPSHOT_FORMAT, "version": SNAPSHOT_VERSION,
        "published_at": published_at, "cache": _public_cache(cache, published_at),
    }
    encoded = json.dumps(envelope, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_SNAPSHOT_BYTES:
        raise ValueError("Published snapshot exceeds the size limit.")
    return encoded


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key.")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError("Nonfinite JSON number.")


def _read_envelope(payload, now, max_bytes):
    if not isinstance(payload, bytes) or len(payload) > max_bytes:
        raise ValueError("Invalid snapshot body.")
    raw = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object,
                     parse_constant=_invalid_constant)
    if (not isinstance(raw, dict) or raw.get("format") != SNAPSHOT_FORMAT
            or type(raw.get("version")) is not int or raw["version"] != SNAPSHOT_VERSION):
        raise ValueError("Unsupported snapshot envelope.")
    published_at = _timestamp(raw.get("published_at"))
    if _seconds(published_at) > now + 300:
        raise ValueError("Snapshot publication is in the future.")
    return {"format": SNAPSHOT_FORMAT, "version": SNAPSHOT_VERSION,
            "published_at": published_at, "cache": _public_cache(raw.get("cache"), published_at)}


def _snapshot_url(value, authenticated=False):
    if not isinstance(value, str):
        raise ValueError("Snapshot URL must be HTTPS.")
    value = value.strip()
    if not value:
        return ""
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("Invalid snapshot URL.")
    try:
        parts = urlsplit(value)
        host, port = parts.hostname, parts.port
    except ValueError:
        raise ValueError("Invalid snapshot URL.") from None
    if (parts.scheme != "https" or not host or parts.username is not None
            or parts.password is not None or parts.fragment or port not in (None, 443)
            or len(host) > 253 or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", host)
            or host == "localhost" or host.endswith((".localhost", ".local", ".internal"))):
        raise ValueError("Snapshot URL must identify an HTTPS host.")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("Snapshot URL must use a hostname.")
    if authenticated and not re.fullmatch(r"[a-z0-9-]+\.private\.blob\.vercel-storage\.com", host):
        raise ValueError("Authorized snapshots require a private Vercel Blob URL.")
    return value


def _fetch_https(url, headers, timeout_seconds, max_bytes):
    """Fetch one HTTPS response with a size bound and read deadline, no redirects."""
    parts = urlsplit(url)
    deadline = time.monotonic() + timeout_seconds
    connection = HTTPSConnection(parts.hostname, port=parts.port or 443, timeout=timeout_seconds)

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Snapshot read timed out.")
        return seconds

    try:
        connection.connect()
        sock = connection.sock
        sock.settimeout(remaining())
        target = (parts.path or "/") + (("?" + parts.query) if parts.query else "")
        connection.request("GET", target, headers=headers)
        sock.settimeout(remaining())
        response = connection.getresponse()
        if response.status != 200:
            raise ValueError("Snapshot HTTP request failed.")
        length = response.getheader("Content-Length")
        if length is not None and (not length.isdigit() or int(length) > max_bytes):
            raise ValueError("Snapshot exceeds its size limit.")
        if response.getheader("Content-Encoding", "identity").lower() != "identity":
            raise ValueError("Unexpected snapshot content encoding.")
        chunks, size = [], 0
        while True:
            sock.settimeout(remaining())
            chunk = response.read1(min(65536, max_bytes + 1 - size))
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise ValueError("Snapshot exceeds its size limit.")
            chunks.append(chunk)
            if length is not None and size == int(length):
                break
        if length is not None and size != int(length):
            raise ValueError("Incomplete snapshot response.")
        return b"".join(chunks)
    finally:
        connection.close()


class HostedResults:
    """Thread-safe reader implementing CourtIQ's results-provider interface."""

    read_only = True

    def __init__(self, url=None, authorization_token=None, clock=None, fetcher=None,
                 cache_ttl=60.0, request_timeout=8.0, max_bytes=MAX_SNAPSHOT_BYTES,
                 manual_refresh_minimum=15.0):
        token = (os.environ.get("COURTIQ_RESULTS_SNAPSHOT_TOKEN", "")
                 if authorization_token is None else authorization_token)
        if not isinstance(token, str) or any(ord(c) < 32 for c in token):
            raise ValueError("Invalid snapshot authorization configuration.")
        self._token = token.strip()
        self._url = _snapshot_url(os.environ.get("COURTIQ_RESULTS_SNAPSHOT_URL", "")
                                  if url is None else url, authenticated=bool(self._token))
        for value in (cache_ttl, request_timeout, manual_refresh_minimum):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Snapshot timing limits must be positive numbers.")
        if not 1 <= request_timeout <= 30 or type(max_bytes) is not int or not 1 <= max_bytes <= MAX_SNAPSHOT_BYTES:
            raise ValueError("Invalid snapshot read limits.")
        self._clock, self._fetcher = clock or time.time, fetcher or _fetch_https
        self._ttl, self._timeout = float(cache_ttl), float(request_timeout)
        self._manual_minimum, self._max_bytes = float(manual_refresh_minimum), max_bytes
        self._lock, self._refresh_lock = threading.RLock(), threading.Lock()
        self._closed, self._envelope, self._digest = False, None, None
        self._last_attempt, self._fetched_at, self._error = None, None, None
        self._revision = 0

    def status(self):
        """Report the last observed snapshot without performing network I/O."""
        with self._lock:
            now = self._clock()
            today = datetime.fromtimestamp(now, UTC).date()
            year_start = date(today.year, 1, 1)
            cache = self._envelope["cache"] if self._envelope else {}
            ranges = [(_day(item["start"]), _day(item["end"])) for item in cache.get("covered_ranges", [])]
            current = _merge_ranges([(max(a, year_start), min(b, today)) for a, b in ranges
                                     if b >= year_start and a <= today])
            covered_days = sum((b - a).days + 1 for a, b in current)
            total_days = (today - year_start).days + 1
            complete = _covers(current, year_start, today)
            state = ("disabled" if not self._url else "paused" if self._closed else
                     "error" if self._error else "ready")
            cooldown = (max(0, int(math.ceil(self._last_attempt + self._manual_minimum - now)))
                        if self._last_attempt is not None else 0)
            message = ("Periodically published match results. Reload checks for the latest published snapshot; "
                       "it does not start a provider sync.")
            if not self._envelope:
                message = ("Published results have not loaded yet. Saved model history remains available."
                           if self._url else "Updated match results are not configured.")
            elif not complete:
                message += " Current-season coverage is partial."
            return {
                "revision": self._revision, "state": state, "configured": bool(self._url),
                "mode": "published_snapshot", "read_only": True,
                "refresh_label": "Reload latest results", "refresh_action": "reload_snapshot",
                "last_sync": cache.get("last_sync"), "updated_at": cache.get("last_sync"),
                "snapshot_published_at": self._envelope["published_at"] if self._envelope else None,
                "snapshot_fetched_at": self._fetched_at,
                "coverage_start": current[0][0].isoformat() if current else None,
                "coverage_end": current[-1][1].isoformat() if current else None,
                "covered_ranges": [{"start": a.isoformat(), "end": b.isoformat()} for a, b in current],
                "coverage_complete": complete, "coverage_year": today.year,
                "coverage_scope": "Completed ATP singles main-draw results",
                "matches": len(cache.get("records", [])), "error": self._error,
                "cache_warning": ("The last successfully loaded snapshot is still displayed."
                                  if self._error and self._envelope else None),
                "delay_seconds": 0, "data_delay_seconds": cache.get("data_delay_seconds"),
                "source_url": SOURCE_URL,
                "progress": {"percent": round(100.0 * covered_days / total_days, 1), "message": message},
                "refresh_interval_seconds": RECENT_TTL,
                "snapshot_check_interval_seconds": self._ttl,
                "manual_refresh_interval_seconds": self._manual_minimum,
                "manual_refresh_in_seconds": cooldown,
                "last_recent_refresh": (_iso(cache["recent_refresh"])
                                        if cache.get("recent_refresh") is not None else None),
            }

    def refresh(self, force=False):
        """Reload published bytes, synchronously; never request provider results."""
        if not self._refresh_lock.acquire(blocking=False):
            return self.status()
        try:
            with self._lock:
                now = self._clock()
                interval = self._manual_minimum if force else self._ttl
                if (self._closed or not self._url or (self._last_attempt is not None
                        and now - self._last_attempt < interval)):
                    return self.status()
                self._last_attempt = now
            headers = {"Accept": "application/json", "Accept-Encoding": "identity",
                       "Cache-Control": "no-cache", "User-Agent": "CourtIQ-Snapshot/1.0"}
            if self._token:
                headers["Authorization"] = "Bearer " + self._token
            try:
                payload = self._fetcher(self._url, headers, self._timeout, self._max_bytes)
                envelope = _read_envelope(payload, self._clock(), self._max_bytes)
                digest = hashlib.sha256(json.dumps(envelope, sort_keys=True,
                                                  separators=(",", ":")).encode()).hexdigest()
                with self._lock:
                    if self._closed:
                        return self.status()
                    if self._envelope:
                        old_time = _seconds(self._envelope["published_at"])
                        new_time = _seconds(envelope["published_at"])
                        old_revision = self._envelope["cache"]["revision"]
                        new_revision = envelope["cache"]["revision"]
                        if (new_time < old_time or (new_time == old_time and
                                (new_revision < old_revision or
                                 (new_revision == old_revision and digest != self._digest)))):
                            raise ValueError("Snapshot publication moved backwards or changed in place.")
                    if digest != self._digest or self._error:
                        self._revision += 1
                    self._envelope, self._digest = envelope, digest
                    self._fetched_at, self._error = _iso(self._clock()), None
            except Exception:
                with self._lock:
                    if not self._closed:
                        error = "The latest published results could not be loaded. Please try again shortly."
                        if self._error != error:
                            self._revision += 1
                        self._error = error
            return self.status()
        finally:
            self._refresh_lock.release()

    def snapshot(self):
        self.refresh()
        with self._lock:
            records = self._envelope["cache"]["records"] if self._envelope else []
            return {"records": copy.deepcopy(records), "status": self.status()}

    def close(self):
        with self._lock:
            self._closed = True
