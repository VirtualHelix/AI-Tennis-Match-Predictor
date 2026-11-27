"""A bounded, read-only SportStatsAPI results cache for CourtIQ.

Only completed ATP singles main-draw results are accepted. This is a results
feed, not a model-training adapter. Player IDs belong to SportStatsAPI and must
not be treated as ATP IDs; abbreviated names deliberately remain abbreviated.

The background worker visits the latest 28 days first, then earlier windows
back to January 1. A window is covered only after all of its pages succeed.
Completed window refreshes reconcile disappeared/corrected records. Corrections
outside a subsequently refreshed window cannot be detected. Calls to refresh()
are nonblocking; a completed recent window is reused for six hours.

Testing seams: clock is a callable returning Unix seconds; an optional
clock.wait(stop_event, seconds) can advance a fake clock. transport is a callable
(url, headers, timeout) -> (HTTP status, response headers, decoded JSON or bytes).
Neither seam receives or writes credentials except the transport's required
Authorization header. There is no third-party dependency.
"""

import copy
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


BASE_URL = "https://api.sportstatsapi.com/api/v1/tennis/matches"
SOURCE_URL = "https://sportstatsapi.com/docs/api/tennis-matches/"
CACHE_VERSION = 1
SCORE_SCHEMA_VERSION = 2
PAGE_SIZE = 100
REQUEST_SPACING = 12.5
DAILY_LIMIT = 90  # Leave ten of the Free plan's 100 calls for other uses.
RECENT_TTL = 6 * 60 * 60
MANUAL_REFRESH_MINIMUM = 15 * 60
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
UTC = timezone.utc


def _utc(now):
    return datetime.fromtimestamp(float(now), UTC)


def _iso(now):
    return _utc(now).isoformat(timespec="seconds").replace("+00:00", "Z")


def _day(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _integer(value, upper=1000000):
    # Numeric strings are used for set scores, but booleans are not numbers here.
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value <= upper else None
    if isinstance(value, str) and re.fullmatch(r"\d{1,9}", value.strip()):
        number = int(value)
        return number if number <= upper else None
    return None


def _text(value, maximum=240):
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    if not value or len(value) > maximum or any(ord(char) < 32 for char in value):
        return None
    return value


def _opaque(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return str(value) if value > 0 else None
    value = _text(value, 100)
    # Public codes are opaque; restrict characters only to prevent unsafe IDs.
    return value if value and re.fullmatch(r"[A-Za-z0-9_.:-]+", value) else None


def _identifier(obj, *fields):
    if not isinstance(obj, dict):
        return None
    for field in fields:
        candidate = _opaque(obj.get(field))
        if candidate:
            return candidate
    return None


def _match_id(raw):
    value = _identifier(raw, "code", "match_code", "match_id", "id")
    return "sportstats:" + value if value else None


def _surface(value):
    value = _text(value, 80)
    if not value:
        return None
    lowered = value.casefold()
    for known in ("hard", "clay", "grass", "carpet"):
        if lowered == known or lowered.startswith(known + " ("):
            return known.title()
    return None  # Do not turn missing/unknown surface information into a guess.


def _valid_set(first, second, match_tiebreak=False):
    if first is None or second is None or first == second:
        return False
    high, low = max(first, second), min(first, second)
    if match_tiebreak:
        return (high == 10 and low <= 8) or (high > 10 and high - low == 2)
    return ((high == 6 and low <= 4) or (high == 7 and low in (5, 6))
            or (high >= 8 and high - low == 2))


def _parse_score_token(token):
    """Read one winner-first score token without inferring missing games."""
    match_tiebreak = token.startswith("[") and token.endswith("]")
    if match_tiebreak:
        token = token[1:-1]
    match = re.fullmatch(r"(\d{1,2})[-\u2013](\d{1,2})(?:\((\d{1,2})\))?", token)
    if not match:
        return None
    first, second = int(match[1]), int(match[2])
    points = int(match[3]) if match[3] is not None else None
    if (not _valid_set(first, second, match_tiebreak)
            or (points is not None and (match_tiebreak or {first, second} != {6, 7}))):
        return None
    return {"winner_games": first, "loser_games": second,
            "tiebreak_loser_points": points, "is_match_tiebreak": match_tiebreak}


def _valid_match_sets(details):
    if not 2 <= len(details) <= 5:
        return False
    wins = sum(row["winner_games"] > row["loser_games"] for row in details)
    losses = len(details) - wins
    if wins not in (2, 3) or losses >= wins:
        return False
    # A completed match ends when the winning player reaches the required sets.
    if sum(row["winner_games"] > row["loser_games"] for row in details[:-1]) >= wins:
        return False
    return not any(row["is_match_tiebreak"] for row in details[:-1])


def parse_completed_score(score):
    """Parse recorded winner-first set notation into an independent list.

    Empty means no reliable complete set detail, including aggregate "2-0 sets"
    scores. Square brackets explicitly identify a match tie-break. A plain
    10-8 remains an advantage-set score rather than guessing its format.
    """
    value = _text(score, 100)
    if not value:
        return []
    details = [_parse_score_token(part) for part in value.split()]
    if any(row is None for row in details) or not _valid_match_sets(details):
        return []
    return details


def _format_set_scores(details):
    parts = []
    for row in details:
        part = "{}-{}".format(row["winner_games"], row["loser_games"])
        if row["is_match_tiebreak"]:
            part = "[" + part + "]"
        elif row["tiebreak_loser_points"] is not None:
            part += "({})".format(row["tiebreak_loser_points"])
        parts.append(part)
    return " ".join(parts)


def _api_games(value):
    plain = _integer(value, 99)
    if plain is not None:
        return plain, None
    if isinstance(value, str):
        # Observed provider notation is games.tie-break-points, not a decimal
        # number: "6.0" must preserve a zero-point tie-break loss.
        annotated = re.fullmatch(r"\s*([67])\.(\d{1,2})\s*", value)
        if annotated:
            return int(annotated[1]), int(annotated[2])
    return None, None


def _valid_tiebreak_points(winner, loser):
    return ((winner == 7 and loser <= 5) or (winner == 10 and loser <= 8)
            or (winner >= 8 and winner - loser == 2))


def _api_set(row, winner_index):
    if not isinstance(row, dict):
        return None
    first, embedded_first = _api_games(row.get("competitor1"))
    second, embedded_second = _api_games(row.get("competitor2"))
    score_text = _text(row.get("score"), 30)
    pair = _parse_score_token(score_text) if score_text else None
    # The provider also documents a complete per-set score string. It can fill
    # absent cells, but must agree with any cells that are already known.
    if pair:
        if ((first is not None and first != pair["winner_games"])
                or (second is not None and second != pair["loser_games"])):
            raise ValueError("Per-set score fields contradict each other")
        first, second = pair["winner_games"], pair["loser_games"]
    match_tiebreak = bool(pair and pair["is_match_tiebreak"])
    if not _valid_set(first, second, match_tiebreak):
        return None
    losing_points = None
    if {first, second} == {6, 7} and not match_tiebreak:
        loser_side = 1 if first < second else 2
        raw_loser = _integer(row.get("tiebreak" + str(loser_side)), 99)
        raw_winner = _integer(row.get("tiebreak" + str(3 - loser_side)), 99)
        embedded_loser = embedded_first if loser_side == 1 else embedded_second
        embedded_winner = embedded_second if loser_side == 1 else embedded_first
        candidates = [v for v in (raw_loser, embedded_loser,
                      pair.get("tiebreak_loser_points") if pair else None) if v is not None]
        if candidates and len(set(candidates)) == 1:
            losing_points = candidates[0]
        # Bad optional tie-break metadata should not erase reliable game scores.
        winning_candidates = [v for v in (raw_winner, embedded_winner) if v is not None]
        if len(set(winning_candidates)) > 1:
            losing_points = None
        if winning_candidates and losing_points is not None:
            if not _valid_tiebreak_points(winning_candidates[0], losing_points):
                losing_points = None
    left, right = (first, second) if winner_index == 0 else (second, first)
    return {"winner_games": left, "loser_games": right,
            "tiebreak_loser_points": losing_points, "is_match_tiebreak": match_tiebreak}


def _games_score(raw, winner_index, first_sets, second_sets):
    sets = raw.get("set_scores")
    if not isinstance(sets, dict):
        return []
    expected = first_sets + second_sets
    details = [_api_set(sets.get("s" + str(index)), winner_index)
               for index in range(1, expected + 1)]
    # Populated extra sets contradict the declared match score.
    for index in range(expected + 1, 6):
        extra = sets.get("s" + str(index))
        if isinstance(extra, dict):
            a, _ = _api_games(extra.get("competitor1"))
            b, _ = _api_games(extra.get("competitor2"))
            if ((a is not None and a > 0) or (b is not None and b > 0)
                    or _parse_score_token(_text(extra.get("score"), 30) or "")):
                raise ValueError("Unexpected additional completed sets")
    if any(row is None for row in details):
        return []
    winners = sum(row["winner_games"] > row["loser_games"] for row in details)
    declared_winners = (first_sets, second_sets)[winner_index]
    if winners != declared_winners or not _valid_match_sets(details):
        raise ValueError("Set winners contradict declared sets won")
    return details


def _aggregate_sets(score):
    value = _text(score, 30)
    match = re.fullmatch(r"([23])[-\u2013]([0-2]) sets", value or "")
    if not match or int(match[1]) <= int(match[2]):
        return None
    return int(match[1]), int(match[2])


def _set_totals(details):
    wins = sum(row["winner_games"] > row["loser_games"] for row in details)
    return wins, len(details) - wins


def _partial_scores_agree(raw, details):
    """A new incomplete payload may omit scores, but may not correct old ones."""
    rows = raw.get("set_scores")
    if rows is None:
        return True
    if not isinstance(rows, dict):
        return False
    winner_index = 0 if raw["competitor1"]["winner"] else 1
    for index, previous in enumerate(details, 1):
        row = rows.get("s" + str(index))
        if row is None:
            continue
        if not isinstance(row, dict):
            return False
        expected = (previous["winner_games"], previous["loser_games"])
        if winner_index:
            expected = expected[::-1]
        for side, games in enumerate(expected, 1):
            value = row.get("competitor" + str(side))
            if value is None or value == "":
                continue
            observed, points = _api_games(value)
            if observed != games:
                return False
            if (points is not None and previous["tiebreak_loser_points"] is not None
                    and games == min(expected) and points != previous["tiebreak_loser_points"]):
                return False
        value = row.get("score")
        if value is not None and value != "":
            pair = _parse_score_token(_text(value, 30) or "")
            if not pair or (pair["winner_games"], pair["loser_games"]) != expected:
                return False
            if pair["is_match_tiebreak"] != previous["is_match_tiebreak"]:
                return False
            if (pair["tiebreak_loser_points"] is not None
                    and previous["tiebreak_loser_points"] is not None
                    and pair["tiebreak_loser_points"] != previous["tiebreak_loser_points"]):
                return False
        # A changed or invalid recorded tie-break must not resurrect old detail.
        old_points = previous["tiebreak_loser_points"]
        loser_side = 1 if expected[0] < expected[1] else 2
        if old_points is not None and row.get("tiebreak" + str(loser_side)) is not None:
            if _integer(row.get("tiebreak" + str(loser_side)), 99) != old_points:
                return False
        if old_points is not None:
            winner_side = 3 - loser_side
            _, embedded = _api_games(row.get("competitor" + str(winner_side)))
            winning_points = [value for value in (
                _integer(row.get("tiebreak" + str(winner_side)), 99), embedded)
                if value is not None]
            if any(not _valid_tiebreak_points(value, old_points) for value in winning_points):
                return False
    return True


def _retain_known_scores(previous, current, raw):
    """Preserve verified detail across omissions, never across result changes."""
    if not previous or any(previous.get(key) != current.get(key)
                           for key in ("match_id", "match_date")):
        return current
    if any(previous[side]["provider_id"] != current[side]["provider_id"]
           for side in ("winner", "loser")):
        return current
    old_details = parse_completed_score(previous.get("score"))
    if not old_details or previous.get("score_kind") != "games":
        return current
    new_details = current["set_scores"]
    if not new_details:
        if (_aggregate_sets(current["score"]) != _set_totals(old_details)
                or not _partial_scores_agree(raw, old_details)):
            return current
        current["score"] = previous["score"]
        current["score_kind"] = "games"
        current["set_scores"] = old_details
        current["score_observed_at"] = previous.get("score_observed_at") or previous["observed_at"]
        return current
    if len(new_details) != len(old_details):
        return current
    # The same games may arrive without the optional tie-break points. Keep
    # those points only when this payload contains no conflicting statement.
    if not _partial_scores_agree(raw, old_details):
        return current
    retained = False
    for old, new in zip(old_details, new_details):
        if (new["tiebreak_loser_points"] is None and old["tiebreak_loser_points"] is not None):
            new["tiebreak_loser_points"] = old["tiebreak_loser_points"]
            retained = True
    if retained:
        current["score"] = _format_set_scores(new_details)
        current["score_observed_at"] = previous.get("score_observed_at") or previous["observed_at"]
    return current


def normalize_match(raw, observed_at):
    """Return a conservative normalized result, or None for an ineligible row.

    Status is mandatory. A score alone is never evidence of completion. Both
    winner flags and sets won must agree, and tour/discipline/stage are checked
    again even though they were sent as server-side filters.
    """
    if not isinstance(raw, dict):
        return None
    if str(raw.get("tour", "")).casefold().strip() != "atp":
        return None
    if str(raw.get("discipline", "")).casefold().strip() != "singles":
        return None
    if str(raw.get("stage", "")).casefold().strip() != "main":
        return None
    sport = str(raw.get("sport", "tennis")).casefold().strip()
    if sport and sport != "tennis":
        return None
    status = str(raw.get("status", "")).casefold().strip()
    code = str(raw.get("status_code", "")).casefold().strip()
    # An explicit non-final code overrides even a contradictory raw status.
    if code and code != "final":
        return None
    if status not in ("finished", "completed", "final", "ft"):
        return None
    # Supplementary retirement flags may exist even with a generic final status.
    for flag in ("retired", "retirement", "walkover", "walk_over", "cancelled", "abandoned"):
        if raw.get(flag) is True:
            return None
    result_type = str(raw.get("result_type", "")).casefold().strip()
    if result_type and result_type not in ("normal", "completed", "finished"):
        return None
    for field in ("score", "result", "result_description"):
        value = raw.get(field)
        if isinstance(value, str) and re.search(r"\b(?:ret(?:ired|irement)?|w/?o|walkover|default(?:ed)?)\b", value, re.I):
            return None
    players = [raw.get("competitor1"), raw.get("competitor2")]
    if not all(isinstance(player, dict) for player in players):
        return None
    if any(isinstance(player.get("member_ids"), list) and len(player["member_ids"]) > 1
           for player in players):
        return None
    flags = [player.get("winner") for player in players]
    if any(type(flag) is not bool for flag in flags) or flags[0] == flags[1]:
        return None
    counts = [_integer(player.get("sets_won"), 3) for player in players]
    if any(value is None for value in counts):
        return None
    winner_index = 0 if flags[0] else 1
    loser_index = 1 - winner_index
    if counts[winner_index] not in (2, 3) or counts[loser_index] >= counts[winner_index]:
        return None
    identity = []
    for player in players:
        identifier = _identifier(player, "code", "player_code", "competitor_code", "id", "player_id")
        name = _text(player.get("name"))
        if not identifier or not name:
            return None
        identity.append({"provider_id": identifier, "name": name})
    if identity[0]["provider_id"] == identity[1]["provider_id"]:
        return None
    identifier = _match_id(raw)
    match_date = _day(raw.get("start_date"))
    if not identifier or match_date is None:
        return None
    try:
        details = _games_score(raw, winner_index, counts[0], counts[1])
    except ValueError:
        return None
    return {
        "match_id": identifier,
        "match_date": match_date.isoformat(),
        "winner": identity[winner_index],
        "loser": identity[loser_index],
        "event": _text(raw.get("category_name")),
        "surface": _surface(raw.get("surface")),
        "score": _format_set_scores(details) if details else "{}-{} sets".format(counts[winner_index], counts[loser_index]),
        "score_kind": "games" if details else "sets",
        "set_scores": details,
        "score_observed_at": observed_at if details else None,
        "source": "sportstats",
        "date_basis": "match_date",
        "observed_at": observed_at,
    }


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_transport(url, headers, timeout):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "api.sportstatsapi.com"
            or parsed.path != "/api/v1/tennis/matches"):
        raise ValueError("Only the configured results endpoint is allowed")
    opener = build_opener(_NoRedirect())
    request = Request(url, headers=headers, method="GET")
    try:
        response = opener.open(request, timeout=timeout)
    except HTTPError as error:
        response = error
    with response:
        status = response.getcode()
        response_headers = dict(response.headers.items())
        # Keep only the one documented error code needed to distinguish an
        # exhausted daily quota from a minute-rate pause. Never retain a raw
        # error body, which may contain sensitive diagnostics.
        if status == 429:
            try:
                body = response.read(65537)
                error = json.loads(body) if len(body) <= 65536 else {}
                code = error.get("error", {}).get("code") if isinstance(error, dict) else None
                if code in ("QUOTA_EXCEEDED", "RATE_LIMIT_EXCEEDED"):
                    return status, response_headers, {"error": {"code": code}}
            except (ValueError, TypeError, AttributeError):
                pass
            return status, response_headers, {}
        if status != 200:
            return status, response_headers, {}
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ValueError("Provider response was too large")
        return status, response_headers, body


def _cache_location(root):
    fingerprint = hashlib.sha256(str(Path(root).resolve()).encode("utf-8")).hexdigest()[:20]
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local"))
    else:
        configured = os.environ.get("XDG_CACHE_HOME", "")
        base = Path(configured) if configured and Path(configured).is_absolute() else Path.home() / ".cache"
    return base / "CourtIQ" / fingerprint / "sportstats_results.json"


def _merge_ranges(ranges):
    valid = sorted((a, b) for a, b in ranges if a <= b)
    merged = []
    for start, end in valid:
        if merged and start <= merged[-1][1] + timedelta(days=1):
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _covers(ranges, start, end):
    return any(a <= start and b >= end for a, b in _merge_ranges(ranges))


def _window_key(start, end):
    return start.isoformat() + "/" + end.isoformat()


def _reset_timestamp(value, now, relative=False):
    numeric = _number(value)
    if numeric is not None:
        if relative:
            return now + max(0, min(numeric, 86400))
        return numeric if numeric > now else None
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return None
            return dt.timestamp() if dt.timestamp() > now else None
        except (ValueError, OverflowError):
            try:
                dt = parsedate_to_datetime(value)
                return dt.timestamp() if dt.tzinfo and dt.timestamp() > now else None
            except (ValueError, TypeError, OverflowError):
                return None
    return None


class SportStatsResults:
    """Thread-safe cached current-season ATP results; no model changes."""

    def __init__(self, root, api_key=None, clock=None, transport=None, cache_path=None):
        self._key = str(os.environ.get("SPORTSTATS_KEY", "") if api_key is None else api_key).strip()
        self._clock = clock or time.time
        self._transport = transport or _http_transport
        self._cache_path = Path(cache_path) if cache_path is not None else _cache_location(root)
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self._records = {}
        self._ranges = []
        self._checkpoints = {}
        self._last_sync = None
        self._recent_refresh = None
        self._recent_range = None
        self._quota_date = _utc(self._clock()).date().isoformat()
        self._requests_today = 0
        self._remote_remaining = None
        self._remote_reset = None
        self._last_request = None
        self._rate_reset = 0.0
        self._backoff_until = 0.0
        self._data_delay = None
        self._state = "disabled" if not self._key else "ready"
        self._revision = 0
        self._error = None
        self._cache_warning = None
        self._progress = {"window_start": None, "window_end": None, "offset": 0, "total": None,
                          "pages_this_run": 0, "excluded_rows_this_run": 0}
        self._load_cache()
        self._revision += 1

    def _roll_day(self, now):
        today = _utc(now).date().isoformat()
        if self._quota_date != today:
            self._quota_date = today
            self._requests_today = 0
            self._revision += 1
        if self._remote_reset is not None and now >= self._remote_reset:
            self._remote_remaining = None
            self._remote_reset = None
            self._revision += 1

    def _midnight(self, now):
        tomorrow = _utc(now).date() + timedelta(days=1)
        return datetime.combine(tomorrow, datetime.min.time(), UTC).timestamp()

    def _delay(self, now):
        until = max(self._backoff_until, self._rate_reset)
        if self._last_request is not None and self._state == "syncing":
            until = max(until, self._last_request + REQUEST_SPACING)
        if self._requests_today >= DAILY_LIMIT:
            until = max(until, self._midnight(now))
        if self._remote_remaining == 0:
            until = max(until, self._remote_reset or self._midnight(now))
        return max(0, int(math.ceil(until - now)))

    def status(self):
        with self._lock:
            now = self._clock()
            self._roll_day(now)
            today = _utc(now).date()
            year_start = date(today.year, 1, 1)
            current = [(max(a, year_start), min(b, today)) for a, b in self._ranges
                       if b >= year_start and a <= today]
            merged = _merge_ranges(current)
            covered_days = sum((b - a).days + 1 for a, b in merged)
            total_days = (today - year_start).days + 1
            progress = dict(self._progress)
            progress["percent"] = round(100.0 * covered_days / total_days, 1)
            progress["message"] = ("Current-season results are covered." if covered_days == total_days else
                                   "Fetching recent results first, then earlier matches this year.")
            delay = self._delay(now)
            return {
                "revision": self._revision,
                "state": self._state,
                "configured": bool(self._key),
                "last_sync": self._last_sync,
                "updated_at": self._last_sync,
                "retry_at": _iso(now + delay) if delay and self._state == "paused" else None,
                "coverage_start": merged[0][0].isoformat() if merged else None,
                "coverage_end": merged[-1][1].isoformat() if merged else None,
                "covered_ranges": [{"start": a.isoformat(), "end": b.isoformat()} for a, b in merged],
                "coverage_complete": _covers(merged, year_start, today),
                "coverage_year": today.year,
                "coverage_scope": "Completed ATP singles main-draw results",
                "matches": len(self._records),
                "requests_today": self._requests_today,
                "request_budget": DAILY_LIMIT,
                "quota_remaining": self._remote_remaining,
                "quota_reset": _iso(self._remote_reset) if self._remote_reset else None,
                "error": self._error,
                "cache_warning": self._cache_warning,
                "delay_seconds": delay,
                "data_delay_seconds": self._data_delay,
                "source_url": SOURCE_URL,
                "progress": progress,
                "refresh_interval_seconds": RECENT_TTL,
                "manual_refresh_interval_seconds": MANUAL_REFRESH_MINIMUM,
                "last_recent_refresh": _iso(self._recent_refresh) if self._recent_refresh else None,
                "manual_refresh_in_seconds": max(0, int(math.ceil(
                    (self._recent_refresh or 0) + MANUAL_REFRESH_MINIMUM - now))),
            }

    def snapshot(self):
        with self._lock:
            records = sorted(self._records.values(), key=lambda row: (row["match_date"], row["match_id"]), reverse=True)
            return {"records": copy.deepcopy(records), "status": self.status()}

    def _load_cache(self):
        try:
            if not self._cache_path.exists():
                return
            if self._cache_path.stat().st_size > 30 * 1024 * 1024:
                raise ValueError("oversized cache")
            cached = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if not isinstance(cached, dict) or cached.get("version") != CACHE_VERSION:
                raise ValueError("unsupported cache")
            self._revision = _integer(cached.get("revision"), 1000000000) or 0
            for row in cached.get("records", []):
                clean = self._cached_record(row)
                if clean:
                    self._records[clean["match_id"]] = clean
            ranges = []
            for item in cached.get("covered_ranges", []):
                if not isinstance(item, dict):
                    continue
                a, b = _day(item.get("start")), _day(item.get("end"))
                if a and b and a <= b:
                    ranges.append((a, b))
            self._ranges = _merge_ranges(ranges)
            for item in cached.get("checkpoints", [])[:30]:
                if not isinstance(item, dict):
                    continue
                a, b = _day(item.get("start")), _day(item.get("end"))
                offset = _integer(item.get("offset"), 100000)
                if not a or not b or a > b or offset is None:
                    continue
                seen = [value for value in item.get("seen", [])[:100000]
                        if isinstance(value, str) and value.startswith("sportstats:") and len(value) <= 120]
                signature = item.get("last_signature")
                if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{64}", signature):
                    signature = None
                self._checkpoints[_window_key(a, b)] = {
                    "start": a.isoformat(), "end": b.isoformat(), "offset": offset,
                    "seen": seen, "last_signature": signature,
                    "total": _integer(item.get("total"), 100000),
                }
            self._last_sync = self._valid_timestamp(cached.get("last_sync"))
            self._recent_refresh = _number(cached.get("recent_refresh"))
            recent_range = cached.get("recent_range")
            if isinstance(recent_range, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}", recent_range):
                self._recent_range = recent_range
            if cached.get("score_schema_version") != SCORE_SCHEMA_VERSION:
                # Earlier versions discarded the provider's dotted tie-break
                # notation. Invalidate only affected current-season dates so
                # the existing bounded worker revisits their normal windows.
                # Keep all records visible while migration waits for its budget.
                today = _utc(self._clock()).date()
                year_start = date(today.year, 1, 1)
                affected = sorted({_day(row["match_date"]) for row in self._records.values()
                                   if row["score_kind"] == "sets"
                                   and year_start.isoformat() <= row["match_date"] <= today.isoformat()})
                retained_ranges = []
                for start, end in self._ranges:
                    for changed in affected:
                        if start <= changed <= end:
                            if start < changed:
                                retained_ranges.append((start, changed - timedelta(days=1)))
                            start = changed + timedelta(days=1)
                    if start <= end:
                        retained_ranges.append((start, end))
                self._ranges = retained_ranges
                for key, checkpoint in list(self._checkpoints.items()):
                    if any(checkpoint["start"] <= changed.isoformat() <= checkpoint["end"]
                           for changed in affected):
                        self._checkpoints.pop(key)
                recent_start = max(year_start, today - timedelta(days=27))
                if any(changed >= recent_start for changed in affected):
                    self._recent_refresh = None
                    self._recent_range = None
            quota = cached.get("quota", {})
            if isinstance(quota, dict):
                quota_day = _day(quota.get("date"))
                if quota_day:
                    self._quota_date = quota_day.isoformat()
                self._requests_today = _integer(quota.get("requests"), 100000) or 0
                self._remote_remaining = _integer(quota.get("remaining"))
                self._remote_reset = _number(quota.get("reset"))
                self._last_request = _number(quota.get("last_request"))
                self._rate_reset = _number(quota.get("rate_reset")) or 0.0
                self._backoff_until = _number(quota.get("backoff_until")) or 0.0
                self._data_delay = _integer(quota.get("data_delay"), 86400)
            self._roll_day(self._clock())
        except (OSError, ValueError, TypeError, AttributeError, OverflowError):
            # Never expose cache contents or filesystem details in public errors.
            self._cache_warning = "The results cache could not be read; a new sync will rebuild it."

    @staticmethod
    def _valid_timestamp(value):
        if not isinstance(value, str) or len(value) > 40:
            return None
        try:
            stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return _iso(stamp.timestamp()) if stamp.tzinfo else None
        except (ValueError, OverflowError, OSError):
            return None

    @staticmethod
    def _cached_record(row):
        if not isinstance(row, dict) or row.get("source") != "sportstats":
            return None
        mid = row.get("match_id")
        if not isinstance(mid, str) or not mid.startswith("sportstats:") or not _opaque(mid[len("sportstats:"):]):
            return None
        match_day = _day(row.get("match_date"))
        if not match_day or row.get("score_kind") not in ("games", "sets"):
            return None
        identities = []
        for side in ("winner", "loser"):
            player = row.get(side)
            if not isinstance(player, dict):
                return None
            identifier, name = _opaque(player.get("provider_id")), _text(player.get("name"))
            if not identifier or not name:
                return None
            identities.append({"provider_id": identifier, "name": name})
        score = _text(row.get("score"), 100)
        observed_at = SportStatsResults._valid_timestamp(row.get("observed_at"))
        if not score or not observed_at:
            return None
        if identities[0]["provider_id"] == identities[1]["provider_id"]:
            return None
        # Reparse the allowlisted score when loading both old and new caches;
        # unverified arrays cannot manufacture details missing from the score.
        details = parse_completed_score(score) if row["score_kind"] == "games" else []
        if row["score_kind"] == "games" and not details:
            return None
        if row["score_kind"] == "sets" and _aggregate_sets(score) is None:
            return None
        score_observed_at = (SportStatsResults._valid_timestamp(row.get("score_observed_at"))
                             or observed_at) if details else None
        return {"match_id": mid, "match_date": match_day.isoformat(), "winner": identities[0],
                "loser": identities[1], "event": _text(row.get("event")), "surface": _surface(row.get("surface")),
                "score": score, "score_kind": row["score_kind"], "source": "sportstats",
                "set_scores": details, "score_observed_at": score_observed_at,
                "date_basis": "match_date", "observed_at": observed_at}

    def _save_cache(self):
        # Caller holds the lock. Everything here is allowlisted, never raw HTTP.
        payload = {
            "version": CACHE_VERSION, "score_schema_version": SCORE_SCHEMA_VERSION,
            "revision": self._revision, "records": list(self._records.values()),
            "covered_ranges": [{"start": a.isoformat(), "end": b.isoformat()} for a, b in self._ranges],
            "checkpoints": list(self._checkpoints.values()),
            "last_sync": self._last_sync, "recent_refresh": self._recent_refresh,
            "recent_range": self._recent_range,
            "quota": {"date": self._quota_date, "requests": self._requests_today,
                      "remaining": self._remote_remaining, "reset": self._remote_reset,
                      "last_request": self._last_request, "rate_reset": self._rate_reset,
                      "backoff_until": self._backoff_until, "data_delay": self._data_delay},
        }
        temporary = None
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=str(self._cache_path.parent),
                                             prefix=".sportstats-", suffix=".tmp", delete=False) as output:
                temporary = output.name
                os.chmod(temporary, 0o600)
                json.dump(payload, output, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self._cache_path)
            self._cache_warning = None
        except (OSError, TypeError, ValueError):
            self._cache_warning = "Results are available in memory, but the local cache could not be saved."
        finally:
            if temporary:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def refresh(self, force=False):
        with self._lock:
            if self._closed:
                self._state = "paused" if self._key else "disabled"
                return self.status()
            if not self._key:
                self._state = "disabled"
                return self.status()
            if self._thread is not None and self._thread.is_alive():
                return self.status()
            now = self._clock()
            self._roll_day(now)
            if self._requests_today >= DAILY_LIMIT or self._remote_remaining == 0 or self._backoff_until > now:
                if self._state != "paused" or not self._error:
                    self._revision += 1
                self._state = "paused"
                if not self._error:
                    self._error = "Sync is paused until the request allowance is available."
                return self.status()
            self._error = None
            tasks, recent_key = self._tasks(_utc(now).date(), force=bool(force))
            if not tasks:
                if self._state != "ready":
                    self._revision += 1
                self._state = "ready"
                return self.status()
            self._state = "syncing"
            self._revision += 1
            self._progress["pages_this_run"] = 0
            self._progress["excluded_rows_this_run"] = 0
            self._thread = threading.Thread(target=self._run, args=(tasks, recent_key),
                                            name="courtiq-sportstats", daemon=True)
            self._thread.start()
            return self.status()

    def close(self):
        with self._lock:
            self._closed = True
            self._stop.set()
            self._state = "paused" if self._key else "disabled"
            self._revision += 1
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.3)

    def _pause(self, message):
        with self._lock:
            self._state = "paused"
            self._error = message
            self._revision += 1
            self._save_cache()

    def _fail(self, message):
        with self._lock:
            self._state = "error"
            self._error = message
            self._revision += 1
            self._save_cache()

    def _wait_for_request(self):
        while not self._stop.is_set():
            with self._lock:
                now = self._clock()
                self._roll_day(now)
                if self._requests_today >= DAILY_LIMIT:
                    self._pause("Daily sync budget reached. Cached results remain available; try again after UTC midnight.")
                    return False
                if self._remote_remaining == 0:
                    self._pause("The provider's daily allowance is exhausted. Cached results remain available.")
                    return False
                target = max(self._rate_reset, self._backoff_until,
                             self._last_request + REQUEST_SPACING if self._last_request is not None else now)
                delay = max(0, target - now)
                if delay <= 0:
                    self._requests_today += 1
                    self._last_request = now
                    if self._remote_remaining is not None:
                        self._remote_remaining = max(0, self._remote_remaining - 1)
                    self._revision += 1
                    self._save_cache()  # Count even a network failure/restart against the allowance.
                    return True
            waiter = getattr(self._clock, "wait", self._stop.wait)
            if hasattr(self._clock, "wait"):
                stopped = waiter(self._stop, min(delay, 60))
            else:
                stopped = waiter(min(delay, 60))
            if stopped:
                return False
        return False

    def _headers(self, headers, now):
        if not isinstance(headers, dict):
            headers = dict(headers)
        lowered = {str(key).lower(): value for key, value in headers.items()}
        with self._lock:
            remaining = _integer(lowered.get("x-quota-remaining"))
            if remaining is not None:
                self._remote_remaining = remaining
                self._remote_reset = (_reset_timestamp(lowered.get("x-quota-reset"), now)
                                      or self._midnight(now))
            rate_remaining = _integer(lowered.get("x-ratelimit-remaining"))
            if rate_remaining == 0:
                self._rate_reset = (_reset_timestamp(lowered.get("x-ratelimit-reset"), now, relative=True)
                                    or now + 60)
            data_delay = _integer(lowered.get("x-data-delay"), 86400)
            if data_delay is not None:
                self._data_delay = data_delay
        return lowered

    def _request_page(self, start, end, offset):
        if not self._wait_for_request():
            return None
        query = urlencode({"date_from": start.isoformat(), "date_to": end.isoformat(), "status": "Finished",
                           "tour": "atp", "discipline": "singles", "stage": "main",
                           "limit": PAGE_SIZE, "offset": offset})
        headers = {"Authorization": "Bearer " + self._key, "Accept": "application/json",
                   "User-Agent": "CourtIQ-results/1.0"}
        try:
            status, response_headers, payload = self._transport(BASE_URL + "?" + query, headers, 20)
            now = self._clock()
            lowered = self._headers(response_headers, now)
        except Exception:
            self._fail("The results provider could not be reached. Cached results remain available; retry later.")
            return None
        if self._stop.is_set():
            return None
        if status in (401, 403):
            self._fail("The provider rejected the API key or its access permissions. Check SPORTSTATS_KEY on the server.")
            return None
        if status == 429:
            with self._lock:
                self._backoff_until = (_reset_timestamp(lowered.get("retry-after"), now, relative=True)
                                       or now + 60)
                error = payload.get("error") if isinstance(payload, dict) else None
                if isinstance(error, dict) and error.get("code") == "QUOTA_EXCEEDED":
                    self._remote_remaining = 0
                    self._remote_reset = self._remote_reset or self._midnight(now)
            self._pause("The provider asked sync to pause. Cached results remain available; retry after the displayed delay.")
            return None
        if 300 <= status < 400:
            self._fail("The provider returned a redirect. It was refused to keep the API key on its configured host.")
            return None
        if status == 402:
            self._fail("This endpoint requires paid access for this key. No payment was attempted.")
            return None
        if status != 200:
            self._fail("The provider returned an unsuccessful response. Cached results remain available; retry later.")
            return None
        try:
            if isinstance(payload, (bytes, str)):
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise ValueError("oversized response")
                payload = json.loads(payload)
            if not isinstance(payload, dict) or payload.get("success") is not True:
                raise ValueError("not a successful envelope")
            rows = payload.get("data")
            meta = payload.get("meta") or {}
            if not isinstance(rows, list) or len(rows) > PAGE_SIZE or not isinstance(meta, dict):
                raise ValueError("invalid page")
            # The documented meaning is "query parameters ignored or changed".
            # With a supported 100-row limit no warning is expected; do not
            # silently certify coverage of a different query.
            if meta.get("warnings"):
                raise ValueError("provider changed or ignored query parameters")
            for raw in rows:
                if not isinstance(raw, dict):
                    raise ValueError("invalid match")
                raw_day = _day(raw.get("start_date"))
                if raw_day and not start <= raw_day <= end:
                    raise ValueError("date filter was not honored")
            returned_offset = _integer(meta.get("offset"), 100000)
            if "offset" in meta and returned_offset is None:
                raise ValueError("invalid pagination offset")
            if returned_offset is not None and returned_offset != offset:
                raise ValueError("offset filter was not honored")
            total = _integer(meta.get("total"), 100000)
            if "total" in meta and total is None:
                raise ValueError("invalid pagination total")
            page_limit = _integer(meta.get("limit"), PAGE_SIZE) if "limit" in meta else PAGE_SIZE
            if not page_limit or len(rows) > page_limit:
                raise ValueError("invalid pagination limit")
            next_offset = offset + len(rows)
            # has_more/next_cursor belong to /changes, not this offset endpoint.
            if total is not None:
                if total < next_offset or (not rows and next_offset < total):
                    raise ValueError("contradictory pagination")
                done = next_offset >= total
            else:
                done = len(rows) < page_limit
            if not done and not rows:
                raise ValueError("empty nonterminal page")
            return rows, next_offset, total, done
        except (TypeError, ValueError, OverflowError):
            with self._lock:
                # A changed total can invalidate an interrupted page position.
                # Retry this bounded window from page zero next time; never loop
                # automatically or mark an inconsistent response as complete.
                self._checkpoints.pop(_window_key(start, end), None)
            self._fail("The provider returned an inconsistent result page. Coverage was not marked complete; cached results remain available.")
            return None

    def _tasks(self, today, force=False):
        start_of_year = date(today.year, 1, 1)
        latest_start = max(start_of_year, today - timedelta(days=27))
        recent_key = _window_key(latest_start, today)
        now = self._clock()
        tasks = []
        if (self._recent_range != recent_key or self._recent_refresh is None
                or now - self._recent_refresh >= RECENT_TTL
                or (force and now - self._recent_refresh >= MANUAL_REFRESH_MINIMUM)
                or not _covers(self._ranges, latest_start, today)
                or recent_key in self._checkpoints):
            tasks.append((latest_start, today))
        # Resume older interrupted pages after first checking the recent window.
        for item in sorted(self._checkpoints.values(), key=lambda value: value["end"], reverse=True):
            a, b = _day(item["start"]), _day(item["end"])
            if a >= start_of_year and b <= today and (a, b) not in tasks:
                tasks.append((a, b))
        end = latest_start - timedelta(days=1)
        while end >= start_of_year:
            start = max(start_of_year, end - timedelta(days=27))
            if not _covers(self._ranges, start, end) and (start, end) not in tasks:
                tasks.append((start, end))
            end = start - timedelta(days=1)
        return tasks, recent_key

    def _run(self, tasks, recent_key):
        try:
            for start, end in tasks:
                if self._stop.is_set():
                    break
                key = _window_key(start, end)
                with self._lock:
                    checkpoint = self._checkpoints.setdefault(key, {
                        "start": start.isoformat(), "end": end.isoformat(), "offset": 0,
                        "seen": [], "last_signature": None, "total": None,
                    })
                    self._progress.update({"window_start": start.isoformat(), "window_end": end.isoformat(),
                                           "offset": checkpoint["offset"], "total": checkpoint["total"]})
                while not self._stop.is_set():
                    page = self._request_page(start, end, checkpoint["offset"])
                    if page is None:
                        return
                    rows, next_offset, total, done = page
                    signature = hashlib.sha256(json.dumps([_match_id(row) for row in rows],
                                                           separators=(",", ":")).encode()).hexdigest()
                    with self._lock:
                        if rows and checkpoint["offset"] and signature == checkpoint["last_signature"]:
                            self._checkpoints.pop(key, None)
                            self._fail("The provider repeated a result page. Sync stopped without marking the window complete.")
                            return
                        observed = _iso(self._clock())
                        seen = set(checkpoint["seen"])
                        for raw in rows:
                            record = normalize_match(raw, observed)
                            if record:
                                record = _retain_known_scores(self._records.get(record["match_id"]), record, raw)
                                self._records[record["match_id"]] = record
                                seen.add(record["match_id"])
                            else:
                                self._progress["excluded_rows_this_run"] += 1
                                identifier = _match_id(raw)
                                previous = self._records.get(identifier)
                                if previous and start.isoformat() <= previous["match_date"] <= end.isoformat():
                                    del self._records[identifier]
                                    seen.discard(identifier)
                        checkpoint.update({"offset": next_offset, "seen": sorted(seen),
                                           "last_signature": signature, "total": total})
                        self._last_sync = observed
                        self._progress.update({"offset": next_offset, "total": total})
                        self._progress["pages_this_run"] += 1
                        self._revision += 1
                        if done:
                            # A complete filtered refetch also removes formerly finished
                            # matches now omitted (e.g. corrected to retired/cancelled).
                            self._records = {mid: row for mid, row in self._records.items()
                                             if not (start.isoformat() <= row["match_date"] <= end.isoformat()
                                                     and mid not in seen)}
                            self._ranges = _merge_ranges(self._ranges + [(start, end)])
                            self._checkpoints.pop(key, None)
                            if key == recent_key:
                                self._recent_refresh = self._clock()
                                self._recent_range = key
                        self._save_cache()
                    if done:
                        break
            with self._lock:
                if self._state == "syncing":
                    self._state = "paused" if self._stop.is_set() else "ready"
                    self._error = None
                    self._revision += 1
        except Exception:
            self._fail("Results sync stopped unexpectedly. Cached results remain available; retry later.")
        finally:
            with self._lock:
                if self._state == "syncing":
                    self._state = "paused" if self._stop.is_set() else "error"
                self._save_cache()
