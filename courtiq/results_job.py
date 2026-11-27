"""Bounded result refreshes with one durable, conditionally written state object.

The state contains the request reservation, lease, pagination checkpoint and
published snapshot. Updating that object with its previous ETag fences delayed
workers out of every write. A reservation is never refunded after a crash.

Blob's Python SDK currently lacks a put ``if_match`` argument. Its documented
HTTP-client extension point supplies the same ``x-if-match`` header used by the
official JavaScript SDK, while the Python SDK still handles authentication,
pathname validation and error decoding. Private reads use the JS SDK's
``cache=0`` delivery query to read the current object from origin.
"""

import copy
import json
import math
import os
import re
import tempfile
import time
import uuid
from http.client import HTTPSConnection
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit

from courtiq.hosted_results import serialize_public_snapshot
from courtiq.sportstats import (
    CACHE_VERSION, DAILY_LIMIT, REQUEST_SPACING, SCORE_SCHEMA_VERSION,
    SportStatsResults, MAX_RESPONSE_BYTES, _day, _iso, _utc,
)


STATE_FORMAT = "courtiq-results-state"
STATE_VERSION = 1
MAX_STATE_BYTES = 30 * 1024 * 1024
MAX_REQUESTS = 12
RUN_SECONDS = 200.0
LEASE_SECONDS = 600.0
PROVIDER_TIMEOUT = 20.0
STORE_TIMEOUT = 10.0


def _bounded_http_transport(url, headers, timeout):
    """One provider request with a body limit and an absolute read deadline."""
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.netloc != "api.sportstatsapi.com"
            or parts.path != "/api/v1/tennis/matches"):
        raise ValueError("Only the configured results endpoint is allowed.")
    deadline = time.monotonic() + min(float(timeout), PROVIDER_TIMEOUT)
    connection = HTTPSConnection(parts.hostname, timeout=min(float(timeout), PROVIDER_TIMEOUT))

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Provider read deadline reached.")
        return seconds

    try:
        connection.connect()
        sock = connection.sock
        sock.settimeout(remaining())
        connection.request("GET", parts.path + "?" + parts.query, headers=headers)
        sock.settimeout(remaining())
        response = connection.getresponse()
        response_headers = dict(response.headers.items())
        if response.status not in (200, 429):
            return response.status, response_headers, {}
        maximum = MAX_RESPONSE_BYTES if response.status == 200 else 65536
        length = response.getheader("Content-Length")
        if length is not None and (not length.isdigit() or int(length) > maximum):
            raise ValueError("Provider response exceeds its size limit.")
        chunks, count = [], 0
        while True:
            sock.settimeout(remaining())
            chunk = response.read1(min(65536, maximum + 1 - count))
            if not chunk:
                break
            count += len(chunk)
            if count > maximum:
                raise ValueError("Provider response exceeds its size limit.")
            chunks.append(chunk)
        if length is not None and count != int(length):
            raise ValueError("Incomplete provider response.")
        body = b"".join(chunks)
        if response.status == 429:
            try:
                payload = json.loads(body)
                code = payload.get("error", {}).get("code")
            except (TypeError, ValueError, AttributeError):
                code = None
            body = {"error": {"code": code}} if code in ("QUOTA_EXCEEDED", "RATE_LIMIT_EXCEEDED") else {}
        return response.status, response_headers, body
    finally:
        connection.close()


class StateConflict(Exception):
    """The expected version is no longer the current durable state."""


class StateUnavailable(Exception):
    """No provider work is safe while durable state is unavailable."""


def _json_bytes(value):
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_STATE_BYTES:
        raise StateUnavailable("Results state exceeds the storage size limit.")
    return encoded


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate state field.")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("Nonfinite state number.")


def _private_location(url):
    if not isinstance(url, str):
        raise ValueError("A private Blob state URL is required.")
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.username or parts.password or parts.fragment
            or parts.port not in (None, 443)
            or not re.fullmatch(r"[a-z0-9-]+\.private\.blob\.vercel-storage\.com", parts.hostname or "")):
        raise ValueError("Results state must use a private Vercel Blob URL.")
    path = unquote(parts.path).lstrip("/")
    if not path or any(bit in ("", ".", "..") for bit in path.split("/")):
        raise ValueError("Invalid results state pathname.")
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["cache"] = "0"
    return path, urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


class BlobStateStore:
    """Private Blob transport; imports the hosting-only SDK lazily."""

    def __init__(self, url=None, timeout_seconds=STORE_TIMEOUT):
        self.url = url or os.environ.get("COURTIQ_RESULTS_STATE_URL", "")
        self.pathname, self.read_url = _private_location(self.url)
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 30):
            raise ValueError("Invalid state-store timeout.")
        self.timeout = float(timeout_seconds)

    @staticmethod
    def _sdk():
        import httpx2 as httpx
        from vercel.api import session
        from vercel.blob import sync as blob
        return httpx, session, blob

    def read(self):
        """Return (document, ETag), or (None, None) only for a missing object."""
        httpx, session, blob = self._sdk()
        try:
            with session(httpx_client_factory=lambda: httpx.Client(timeout=self.timeout)):
                with blob.stream_blob(self.read_url, access="private") as stream:
                    if stream.metadata.size is not None and stream.metadata.size > MAX_STATE_BYTES:
                        raise StateUnavailable("Results state exceeds the storage size limit.")
                    etag = stream.metadata.etag
                    if not isinstance(etag, str) or not etag:
                        raise StateUnavailable("Results storage omitted its version.")
                    chunks, length = [], 0
                    read_deadline = time.monotonic() + self.timeout
                    for chunk in stream:
                        if time.monotonic() >= read_deadline:
                            raise StateUnavailable("Results state read deadline reached.")
                        length += len(chunk)
                        if length > MAX_STATE_BYTES:
                            raise StateUnavailable("Results state exceeds the storage size limit.")
                        chunks.append(chunk)
                payload = b"".join(chunks)
                document = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object,
                                      parse_constant=_reject_constant)
                if not isinstance(document, dict):
                    raise StateUnavailable("Invalid results state.")
                return document, etag
        except Exception as error:
            if getattr(error, "status_code", None) == 404:
                return None, None
            if isinstance(error, StateUnavailable):
                raise
            raise StateUnavailable("Results storage could not be read.") from None

    def compare_and_set(self, expected_etag, document):
        """Create once, or replace only the version that the caller owns."""
        payload = _json_bytes(document)
        httpx, session, blob = self._sdk()
        headers = {}
        if expected_etag is not None:
            if (not isinstance(expected_etag, str) or not expected_etag
                    or len(expected_etag) > 512 or any(ord(c) < 32 for c in expected_etag)):
                raise StateUnavailable("Invalid expected state version.")
            headers["x-if-match"] = expected_etag
        try:
            with session(httpx_client_factory=lambda: httpx.Client(timeout=self.timeout, headers=headers)):
                result = blob.put_blob(self.pathname, payload, access="private",
                                       content_type="application/json", add_random_suffix=False,
                                       allow_overwrite=expected_etag is not None,
                                       cache_control_max_age=60)
            if not isinstance(result.etag, str) or not result.etag:
                raise StateUnavailable("Results storage did not confirm its version.")
            return result.etag
        except Exception as error:
            status = getattr(error, "status_code", None)
            if status == 412 or (expected_etag is None and status == 400):
                raise StateConflict("Results state changed.") from None
            if isinstance(error, StateUnavailable):
                raise
            # A timeout can mean the write committed but its response was lost.
            # The caller must stop, not retry a provider request or release a lease.
            raise StateUnavailable("Results storage did not confirm the write.") from None


def fetch_published_snapshot(url, headers, timeout_seconds, max_bytes):
    """HostedResults transport seam; credentials remain inside the Blob SDK."""
    state, _ = BlobStateStore(url, timeout_seconds).read()
    if (not isinstance(state, dict) or state.get("format") != STATE_FORMAT
            or state.get("version") != STATE_VERSION or not isinstance(state.get("snapshot"), dict)):
        raise StateUnavailable("No published results snapshot is available.")
    payload = _json_bytes(state["snapshot"])
    if len(payload) > max_bytes:
        raise StateUnavailable("Published results exceed the read limit.")
    return payload


def _empty_cache(now):
    return {
        "version": CACHE_VERSION, "score_schema_version": SCORE_SCHEMA_VERSION,
        "revision": 0, "records": [], "covered_ranges": [], "checkpoints": [],
        "last_sync": None, "recent_refresh": None, "recent_range": None,
        "quota": {"date": _utc(now).date().isoformat(), "requests": 0,
                  "remaining": None, "reset": None, "last_request": None,
                  "rate_reset": 0.0, "backoff_until": 0.0, "data_delay": None},
    }


def _new_state(now):
    cache = _empty_cache(now)
    return {"format": STATE_FORMAT, "version": STATE_VERSION, "revision": 0,
            "quota": {"date": _utc(now).date().isoformat(), "reserved": 0},
            "lease": None, "cache": cache,
            "snapshot": json.loads(serialize_public_snapshot(cache, _iso(now)))}


def _validate_state(document, now):
    if (not isinstance(document, dict) or document.get("format") != STATE_FORMAT
            or type(document.get("version")) is not int or document["version"] != STATE_VERSION
            or type(document.get("revision")) is not int or document["revision"] < 0):
        raise StateUnavailable("Invalid results state.")
    quota = document.get("quota")
    if (not isinstance(quota, dict) or _day(quota.get("date")) is None
            or quota["date"] > _utc(now).date().isoformat()
            or type(quota.get("reserved")) is not int or not 0 <= quota["reserved"] <= DAILY_LIMIT):
        raise StateUnavailable("Invalid request reservation.")
    lease = document.get("lease")
    if lease is not None:
        if (not isinstance(lease, dict) or not isinstance(lease.get("owner"), str)
                or not re.fullmatch(r"[a-f0-9]{32}", lease["owner"])
                or isinstance(lease.get("expires_at"), bool)
                or not isinstance(lease.get("expires_at"), (int, float))
                or not math.isfinite(lease["expires_at"]) or lease["expires_at"] < 0):
            raise StateUnavailable("Invalid refresh lease.")
    cache = document.get("cache")
    try:
        serialize_public_snapshot(cache, _iso(now))
    except (TypeError, ValueError, OverflowError):
        raise StateUnavailable("Invalid saved results cache.") from None
    if not isinstance(cache.get("quota"), dict) or not isinstance(cache.get("checkpoints"), list):
        raise StateUnavailable("Invalid saved results checkpoint.")
    return copy.deepcopy(document)


class _Publisher:
    def __init__(self, store, state, etag, owner, clock):
        self.store, self.state, self.etag = store, state, etag
        self.owner, self.clock, self.failed = owner, clock, False

    def save(self, cache, release=False):
        if self.failed:
            raise StateUnavailable("Refresh lost its durable state.")
        now = self.clock()
        lease = self.state.get("lease")
        if not lease or lease["owner"] != self.owner or now >= lease["expires_at"]:
            self.failed = True
            raise StateConflict("Refresh lease expired.")
        candidate = copy.deepcopy(self.state)
        candidate["cache"] = cache
        candidate["snapshot"] = json.loads(serialize_public_snapshot(cache, _iso(now)))
        candidate["revision"] += 1
        if release:
            candidate["lease"] = None
        try:
            etag = self.store.compare_and_set(self.etag, candidate)
        except Exception:
            self.failed = True
            raise
        self.state, self.etag = candidate, etag


class _BoundedProvider(SportStatsResults):
    def __init__(self, root, cache_path, api_key, clock, transport, publisher,
                 deadline, allowance, reservation_day, reserved_today):
        self._publisher = publisher
        self._deadline, self._allowance = deadline, allowance
        self._reservation_day, self._reserved_today = reservation_day, reserved_today
        self.calls_used = 0
        super().__init__(root, api_key=api_key, clock=clock, transport=transport, cache_path=cache_path)
        self._requests_today = reserved_today
        if self._cache_warning:
            raise StateUnavailable("The saved checkpoint could not be restored.")

    def _cache_document(self):
        return {
            "version": CACHE_VERSION, "score_schema_version": SCORE_SCHEMA_VERSION,
            "revision": self._revision, "records": list(self._records.values()),
            "covered_ranges": [{"start": a.isoformat(), "end": b.isoformat()} for a, b in self._ranges],
            "checkpoints": list(self._checkpoints.values()), "last_sync": self._last_sync,
            "recent_refresh": self._recent_refresh, "recent_range": self._recent_range,
            "quota": {"date": self._quota_date, "requests": self._reserved_today,
                      "remaining": self._remote_remaining, "reset": self._remote_reset,
                      "last_request": self._last_request, "rate_reset": self._rate_reset,
                      "backoff_until": self._backoff_until, "data_delay": self._data_delay},
        }

    def _save_cache(self):
        if self._publisher.failed:
            self._stop.set()
            return
        try:
            self._publisher.save(self._cache_document())
        except Exception:
            self._publisher.failed = True
            self._stop.set()
            self._state = "paused"
            self._error = "Refresh stopped because its stored checkpoint could not be confirmed."

    def _wait_for_request(self):
        while not self._stop.is_set() and not self._publisher.failed:
            now = self._clock()
            if _utc(now).date().isoformat() != self._reservation_day:
                self._pause("Refresh stopped at UTC midnight; the next run uses a new reservation.")
                return False
            if self.calls_used >= self._allowance:
                self._pause("This refresh batch is complete. Remaining pages resume on the next scheduled run.")
                return False
            if self._remote_remaining == 0:
                self._pause("The provider's allowance is exhausted. Published results remain available.")
                return False
            target = max(now, self._rate_reset, self._backoff_until,
                         self._last_request + REQUEST_SPACING if self._last_request is not None else now)
            # Leave room for the request, checkpoint publication and final release.
            if target + PROVIDER_TIMEOUT + 4 * STORE_TIMEOUT >= self._deadline:
                self._pause("This refresh reached its time limit. Remaining pages resume on the next run.")
                return False
            delay = target - now
            if delay > 0:
                if hasattr(self._clock, "wait"):
                    stopped = self._clock.wait(self._stop, min(delay, 30.0))
                else:
                    stopped = self._stop.wait(min(delay, 30.0))
                if stopped:
                    return False
                continue
            self._last_request = now
            if self._remote_remaining is not None:
                self._remote_remaining = max(0, self._remote_remaining - 1)
            self._revision += 1
            # Verify/fence ownership before every provider request, including retries.
            self._save_cache()
            if self._publisher.failed or self._stop.is_set():
                return False
            self.calls_used += 1
            return True
        return False

    def run_batch(self, force=False):
        tasks, recent_key = self._tasks(_utc(self._clock()).date(), force=force)
        self._state = "syncing"
        self._error = None
        self._revision += 1
        self._run(tasks, recent_key)


def run_results_job(store=None, api_key=None, clock=None, transport=None,
                    max_requests=MAX_REQUESTS, run_seconds=RUN_SECONDS, force=False):
    """Perform one bounded refresh and return an allowlisted operational summary.

    Call from a protected scheduled endpoint and wait for it to finish. No thread
    is created, no model is loaded, and no provider key enters durable state.
    """
    clock = clock or time.time
    key = str(os.environ.get("SPORTSTATS_KEY", "") if api_key is None else api_key).strip()
    if not key:
        return {"state": "disabled", "requests": 0, "message": "The results provider is not configured."}
    if type(max_requests) is not int or not 1 <= max_requests <= MAX_REQUESTS:
        raise ValueError("Refresh requests must be between 1 and 12.")
    if (isinstance(run_seconds, bool) or not isinstance(run_seconds, (int, float))
            or not 45 <= run_seconds <= RUN_SECONDS):
        raise ValueError("Refresh duration must be between 45 and 200 seconds.")
    started = clock()
    deadline = started + float(run_seconds)
    day, owner = _utc(started).date().isoformat(), uuid.uuid4().hex
    store = store or BlobStateStore()
    publisher = provider = None
    try:
        for _ in range(4):
            state, etag = store.read()
            state = _new_state(clock()) if state is None else _validate_state(state, clock())
            lease = state.get("lease")
            if lease and clock() < lease["expires_at"]:
                return {"state": "busy", "requests": 0, "message": "A refresh is already running."}
            if state["quota"]["date"] != day:
                state["quota"] = {"date": day, "reserved": 0}
            allowance = min(max_requests, DAILY_LIMIT - state["quota"]["reserved"])
            if allowance <= 0:
                return {"state": "budget_exhausted", "requests": 0, "reserved_today": DAILY_LIMIT}
            state["quota"]["reserved"] += allowance
            state["lease"] = {"owner": owner, "expires_at": clock() + LEASE_SECONDS}
            state["revision"] += 1
            try:
                etag = store.compare_and_set(etag, state)
            except StateConflict:
                continue
            publisher = _Publisher(store, state, etag, owner, clock)
            break
        if publisher is None:
            return {"state": "busy", "requests": 0, "message": "Another refresh updated the checkpoint."}

        with tempfile.TemporaryDirectory(prefix="courtiq-results-job-") as root:
            cache_path = Path(root) / "checkpoint.json"
            cache_path.write_bytes(_json_bytes(publisher.state["cache"]))
            provider = _BoundedProvider(root, cache_path, key, clock, transport or _bounded_http_transport,
                                        publisher, deadline, allowance, day,
                                        publisher.state["quota"]["reserved"])
            provider.run_batch(force=bool(force))
            if publisher.failed:
                return {"state": "storage_error", "requests": provider.calls_used,
                        "message": "The refresh checkpoint could not be confirmed; its reservation remains consumed."}
            publisher.save(provider._cache_document(), release=True)
            return {"state": provider._state, "requests": provider.calls_used,
                    "reserved_today": publisher.state["quota"]["reserved"],
                    "matches": len(provider._records), "last_sync": provider._last_sync,
                    "elapsed_seconds": round(clock() - started, 3)}
    except Exception:
        # An unconfirmed write may already exist. Keep its reservation and lease;
        # the next invocation reads the actual durable version after expiration.
        return {"state": "storage_error", "requests": provider.calls_used if provider else 0,
                "message": "The refresh could not safely confirm its durable state."}
