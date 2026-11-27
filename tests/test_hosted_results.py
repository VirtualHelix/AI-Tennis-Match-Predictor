"""Hosted snapshots retain score/coverage semantics without provider access."""

import copy
from datetime import datetime, timezone
import json
import threading
import unittest
from unittest.mock import patch

from courtiq.hosted_results import (
    HostedResults, MAX_SNAPSHOT_BYTES, _fetch_https, serialize_public_snapshot,
)


PUBLISHED = "2026-01-20T12:00:00Z"
PUBLIC_URL = "https://example.com/courtiq/results.json"
PRIVATE_URL = "https://test-store.private.blob.vercel-storage.com/results.json?cache=0"


def record(identifier="match1"):
    return {
        "match_id": "sportstats:" + identifier, "source": "sportstats",
        "match_date": "2026-01-20", "observed_at": PUBLISHED,
        "winner": {"provider_id": "player1", "name": "A. Player"},
        "loser": {"provider_id": "player2", "name": "B. Player"},
        "event": "Test Open", "surface": "Hard", "score": "7-6(0) 6-4",
        "score_kind": "games", "date_basis": "match_date",
        "set_scores": [{"untrusted": "must not survive"}],
        "score_observed_at": PUBLISHED,
    }


def cache():
    return {
        "version": 1, "score_schema_version": 2, "revision": 5,
        "records": [record()],
        "covered_ranges": [{"start": "2026-01-01", "end": "2026-01-20"}],
        "last_sync": PUBLISHED, "recent_refresh": None, "recent_range": None,
        "checkpoints": [{"private": "writer-only-checkpoint"}],
        "quota": {"date": "2026-01-20", "requests": 9, "data_delay": 20,
                  "unexpected": "writer-only-secret"},
    }


class Clock:
    def __init__(self):
        self.value = datetime(2026, 1, 20, 12, 1, tzinfo=timezone.utc).timestamp()

    def __call__(self):
        return self.value


class Fetcher:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, timeout, max_bytes):
        self.calls.append((url, headers.copy(), timeout, max_bytes))
        if not self.responses:
            raise AssertionError("Unexpected snapshot request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class PublisherTests(unittest.TestCase):
    def test_export_is_allowlisted_and_reparses_individual_set_scores(self):
        original = cache()
        original["records"][0]["raw_response"] = {"token": "never-publish"}
        before = copy.deepcopy(original)
        encoded = serialize_public_snapshot(original, PUBLISHED)
        data = json.loads(encoded)
        self.assertEqual(data["format"], "courtiq-results-snapshot")
        self.assertEqual(data["version"], 1)
        self.assertEqual(data["cache"]["data_delay_seconds"], 20)
        self.assertNotIn(b"writer-only", encoded)
        self.assertNotIn(b"never-publish", encoded)
        self.assertNotIn("quota", data["cache"])
        self.assertNotIn("checkpoints", data["cache"])
        self.assertEqual(data["cache"]["records"][0]["set_scores"], [
            {"winner_games": 7, "loser_games": 6, "tiebreak_loser_points": 0, "is_match_tiebreak": False},
            {"winner_games": 6, "loser_games": 4, "tiebreak_loser_points": None, "is_match_tiebreak": False},
        ])
        self.assertEqual(original, before)

    def test_missing_set_details_stay_missing(self):
        data = cache()
        data["records"][0].update(score="2-0 sets", score_kind="sets")
        row = json.loads(serialize_public_snapshot(data, PUBLISHED))["cache"]["records"][0]
        self.assertEqual(row["set_scores"], [])
        self.assertEqual(row["score_kind"], "sets")
        self.assertIsNone(row["score_observed_at"])

    def test_malformed_records_cannot_keep_complete_coverage(self):
        mutations = [
            lambda c: c["records"].append({"source": "sportstats"}),
            lambda c: c["records"].append(copy.deepcopy(c["records"][0])),
            lambda c: c["records"][0].update(score="6-4", score_kind="games"),
            lambda c: c["records"][0].update(match_date="2026-01-21"),
            lambda c: c.update(last_sync=None),
            lambda c: c.update(version=True),
            lambda c: c.update(score_schema_version=1),
            lambda c: c["covered_ranges"][0].update(end="2026-01-21"),
            lambda c: c.update(recent_refresh=float("nan")),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                data = cache()
                mutate(data)
                with self.assertRaises(ValueError):
                    serialize_public_snapshot(data, PUBLISHED)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.good = serialize_public_snapshot(cache(), PUBLISHED)

    def reader(self, fetcher, **kwargs):
        return HostedResults(url=PUBLIC_URL, authorization_token="", clock=self.clock,
                             fetcher=fetcher, **kwargs)

    def test_status_is_nonblocking_and_snapshot_has_compatible_read_only_status(self):
        fetcher = Fetcher(self.good)
        reader = self.reader(fetcher)
        self.assertEqual(reader.status()["matches"], 0)
        self.assertEqual(fetcher.calls, [])
        result = reader.snapshot()
        status = result["status"]
        self.assertTrue(status["read_only"])
        self.assertTrue(status["configured"])
        self.assertEqual(status["mode"], "published_snapshot")
        self.assertEqual(status["refresh_label"], "Reload latest results")
        self.assertEqual(status["state"], "ready")
        self.assertTrue(status["coverage_complete"])
        self.assertEqual(status["last_sync"], PUBLISHED)
        self.assertEqual(status["snapshot_published_at"], PUBLISHED)
        self.assertEqual(status["progress"]["percent"], 100)
        self.assertNotIn("requests_today", status)
        self.assertNotIn("quota", status)
        self.assertEqual(result["records"][0]["score"], "7-6(0) 6-4")
        self.assertEqual(len(fetcher.calls), 1)

    def test_ttl_manual_cooldown_and_recovery_keep_real_sync_time(self):
        fetcher = Fetcher(self.good, self.good, self.good)
        reader = self.reader(fetcher)
        first = reader.refresh()
        reader.refresh(force=True)
        self.assertEqual(len(fetcher.calls), 1)
        self.clock.value += 16
        second = reader.refresh(force=True)
        self.assertEqual(len(fetcher.calls), 2)
        self.assertEqual(second["revision"], first["revision"])
        self.assertEqual(second["last_sync"], PUBLISHED)
        self.assertNotEqual(second["snapshot_fetched_at"], first["snapshot_fetched_at"])
        self.clock.value += 59
        reader.refresh()
        self.assertEqual(len(fetcher.calls), 2)
        self.clock.value += 1
        reader.refresh()
        self.assertEqual(len(fetcher.calls), 3)

    def test_failure_preserves_records_and_redacts_sdk_exception(self):
        fetcher = Fetcher(self.good, RuntimeError("token=secret-value https://private-url"), self.good)
        reader = self.reader(fetcher)
        first = reader.snapshot()
        self.clock.value += 61
        failed = reader.snapshot()
        self.assertEqual(failed["records"], first["records"])
        self.assertEqual(failed["status"]["state"], "error")
        self.assertEqual(failed["status"]["last_sync"], PUBLISHED)
        self.assertIsNotNone(failed["status"]["cache_warning"])
        self.assertNotIn("secret-value", json.dumps(failed))
        self.assertNotIn("private-url", json.dumps(failed))
        self.clock.value += 61
        recovered = reader.refresh()
        self.assertEqual(recovered["state"], "ready")
        self.assertIsNone(recovered["error"])

    def test_bad_payloads_do_not_replace_last_good_data(self):
        wrong_version = json.loads(self.good)
        wrong_version["cache"]["score_schema_version"] = 1
        invalid_record = json.loads(self.good)
        invalid_record["cache"]["records"].append({})
        future = json.loads(self.good)
        future["published_at"] = "2027-01-20T12:00:00Z"
        for bad in (b"not-json", b'{"format": NaN}', b'{"version":1,"version":2}',
                    json.dumps(wrong_version).encode(), json.dumps(invalid_record).encode(),
                    json.dumps(future).encode(), b"x" * (len(self.good) + 1)):
            with self.subTest(body=bad[:50]):
                fetcher = Fetcher(self.good, bad)
                reader = self.reader(fetcher, max_bytes=len(self.good))
                original = reader.snapshot()
                self.clock.value += 61
                result = reader.snapshot()
                self.assertEqual(result["records"], original["records"])
                self.assertEqual(result["status"]["state"], "error")

    def test_first_failure_keeps_saved_history_available_with_empty_overlay(self):
        reader = self.reader(Fetcher(OSError("offline")))
        result = reader.snapshot()
        self.assertEqual(result["records"], [])
        self.assertFalse(result["status"]["coverage_complete"])
        self.assertIsNone(result["status"]["last_sync"])
        self.assertEqual(result["status"]["state"], "error")

    def test_new_publication_changes_revision_and_stale_delivery_cannot_roll_back(self):
        newer_cache = cache()
        newer_cache["revision"] += 1
        newer_cache["records"][0]["score"] = "6-4 6-4"
        newer = serialize_public_snapshot(newer_cache, "2026-01-20T12:00:30Z")
        fetcher = Fetcher(self.good, newer, self.good)
        reader = self.reader(fetcher)
        first = reader.refresh()
        self.clock.value += 61
        second = reader.refresh()
        self.assertGreater(second["revision"], first["revision"])
        self.clock.value += 61
        result = reader.snapshot()
        self.assertEqual(result["records"][0]["score"], "6-4 6-4")
        self.assertEqual(result["status"]["state"], "error")
        self.assertEqual(result["status"]["snapshot_published_at"], "2026-01-20T12:00:30Z")

    def test_coverage_is_derived_and_becomes_partial_on_new_day_or_year(self):
        reader = self.reader(Fetcher(self.good))
        self.assertTrue(reader.refresh()["coverage_complete"])
        self.clock.value += 86400
        self.assertFalse(reader.status()["coverage_complete"])
        self.clock.value = datetime(2027, 1, 1, tzinfo=timezone.utc).timestamp()
        status = reader.status()
        self.assertFalse(status["coverage_complete"])
        self.assertEqual(status["coverage_year"], 2027)
        self.assertEqual(status["covered_ranges"], [])
        self.assertEqual(status["progress"]["percent"], 0)

    def test_same_second_publications_use_revision_to_distinguish_updates(self):
        updated = cache()
        updated["revision"] += 1
        updated["records"][0]["score"] = "6-4 6-4"
        published = serialize_public_snapshot(updated, PUBLISHED)
        conflicting = copy.deepcopy(updated)
        conflicting["records"][0]["score"] = "6-1 6-1"
        reader = self.reader(Fetcher(self.good, published,
                                     serialize_public_snapshot(conflicting, PUBLISHED)))
        reader.refresh()
        self.clock.value += 61
        self.assertEqual(reader.refresh()["state"], "ready")
        self.assertEqual(reader.snapshot()["records"][0]["score"], "6-4 6-4")
        self.clock.value += 61
        self.assertEqual(reader.refresh()["state"], "error")
        self.assertEqual(reader.snapshot()["records"][0]["score"], "6-4 6-4")

    def test_mutating_returned_status_or_records_cannot_change_reader(self):
        reader = self.reader(Fetcher(self.good))
        result = reader.snapshot()
        result["records"][0]["winner"]["name"] = "Modified"
        result["status"]["covered_ranges"][0]["start"] = "1900-01-01"
        self.assertEqual(reader.snapshot()["records"][0]["winner"]["name"], "A. Player")
        self.assertEqual(reader.status()["covered_ranges"][0]["start"], "2026-01-01")

    def test_private_blob_authorization_is_only_sent_to_configured_private_host(self):
        fetcher = Fetcher(self.good)
        reader = HostedResults(url=PRIVATE_URL, authorization_token="test-token", fetcher=fetcher,
                               clock=self.clock)
        reader.refresh()
        self.assertEqual(fetcher.calls[0][0], PRIVATE_URL)
        self.assertEqual(fetcher.calls[0][1]["Authorization"], "Bearer test-token")
        self.assertNotIn("test-token", json.dumps(reader.status()))
        for bad in (PUBLIC_URL, "https://store.public.blob.vercel-storage.com/r.json",
                    "https://test-store.private.blob.vercel-storage.com.evil.test/r.json"):
            with self.assertRaises(ValueError):
                HostedResults(url=bad, authorization_token="test-token", fetcher=fetcher)

    def test_unconfigured_and_closed_readers_never_fetch(self):
        fetcher = Fetcher()
        reader = HostedResults(url="", authorization_token="", fetcher=fetcher)
        self.assertEqual(reader.refresh()["state"], "disabled")
        self.assertEqual(reader.snapshot()["records"], [])
        reader = self.reader(fetcher)
        reader.close()
        self.assertEqual(reader.refresh()["state"], "paused")
        self.assertEqual(fetcher.calls, [])

    def test_one_refresh_in_flight_does_not_block_status_or_duplicate_http(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def fetch(url, headers, timeout, max_bytes):
            calls.append(url)
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test refresh did not finish")
            return self.good

        reader = self.reader(fetch)
        worker = threading.Thread(target=reader.refresh)
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertEqual(reader.status()["matches"], 0)
            self.assertEqual(reader.refresh()["matches"], 0)
            self.assertEqual(len(calls), 1)
        finally:
            release.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(reader.status()["matches"], 1)

    def test_invalid_url_and_limits_fail_without_network(self):
        for url in ("http://example.com/data", "https://localhost/data", "https://127.0.0.1/data",
                    "https://name:password@example.com/data", "https://example.com:8443/data",
                    "https://example.com/data#fragment", "https://example.com/\r\nInjected: yes"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                HostedResults(url=url, authorization_token="")
        for limits in ({"max_bytes": MAX_SNAPSHOT_BYTES + 1}, {"request_timeout": 31},
                       {"request_timeout": float("nan")}, {"cache_ttl": 0},
                       {"manual_refresh_minimum": 0}):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                self.reader(Fetcher(), **limits)


class HTTPSLimitsTests(unittest.TestCase):
    def response(self, body=b"{}", status=200, content_length=None, encoding="identity"):
        class Response:
            def __init__(self):
                self.status = status
                self.body = body

            def getheader(self, key, default=None):
                return {"Content-Length": content_length, "Content-Encoding": encoding}.get(key, default)

            def read1(self, limit):
                chunk, self.body = self.body[:limit], self.body[limit:]
                return chunk

        return Response()

    def connection(self, response):
        class Socket:
            def settimeout(self, value):
                self.timeout = value

        class Connection:
            def __init__(self):
                self.sock = Socket()
                self.requests = []
                self.closed = False

            def connect(self):
                pass

            def request(self, method, target, headers):
                self.requests.append((method, target, headers))

            def getresponse(self):
                return response

            def close(self):
                self.closed = True

        return Connection()

    def test_redirects_wrong_lengths_oversized_or_encoded_bodies_are_rejected(self):
        responses = [self.response(status=302), self.response(content_length="999"),
                     self.response(content_length="3"), self.response(content_length="invalid"),
                     self.response(body=b"12345"), self.response(encoding="gzip")]
        for response in responses:
            connection = self.connection(response)
            with self.subTest(response=response), patch("courtiq.hosted_results.HTTPSConnection", return_value=connection):
                with self.assertRaises(ValueError):
                    _fetch_https(PUBLIC_URL, {}, 8, 4)
            self.assertEqual(len(connection.requests), 1)
            self.assertTrue(connection.closed)

    def test_default_transport_preserves_query_and_enforces_read_deadline(self):
        connection = self.connection(self.response(content_length="2"))
        with patch("courtiq.hosted_results.HTTPSConnection", return_value=connection):
            self.assertEqual(_fetch_https(PRIVATE_URL, {"Authorization": "Bearer test-token"}, 8, 4), b"{}")
        self.assertEqual(connection.requests[0][1], "/results.json?cache=0")
        connection = self.connection(self.response())
        with patch("courtiq.hosted_results.HTTPSConnection", return_value=connection), \
                patch("courtiq.hosted_results.time.monotonic", side_effect=[0, 0, 0, 9]):
            with self.assertRaises(TimeoutError):
                _fetch_https(PUBLIC_URL, {}, 8, 4)
        self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
