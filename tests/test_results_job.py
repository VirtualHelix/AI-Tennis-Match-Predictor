"""Durable refresh tests: no live API, Blob store, real sleeps, or credentials."""

import copy
import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from courtiq import results_job as job
from courtiq.hosted_results import HostedResults
from tests.test_sportstats import FakeClock, STAMP, TEST_CREDENTIAL, match, response


PRIVATE_URL = "https://example.private.blob.vercel-storage.com/results/state.json"


class MemoryStore:
    """Atomic compare-and-set, with hooks for crashed/concurrent workers."""

    def __init__(self, document=None):
        self.document = copy.deepcopy(document)
        self.version = 1 if document is not None else 0
        self.reads = 0
        self.writes = []
        self.before_write = None
        self.after_write = None

    @property
    def etag(self):
        return '"version-%d"' % self.version if self.version else None

    def read(self):
        self.reads += 1
        return copy.deepcopy(self.document), self.etag

    def compare_and_set(self, expected, document):
        if self.before_write:
            self.before_write(self, expected, document)
        if expected != self.etag:
            raise job.StateConflict("Concurrent write")
        self.document = copy.deepcopy(document)
        self.version += 1
        self.writes.append(copy.deepcopy(document))
        if self.after_write:
            self.after_write(self, expected, document)
        return self.etag


class PagingTransport:
    def __init__(self, clock, total=2000, delay=0, hook=None):
        self.clock, self.total, self.delay, self.hook = clock, total, delay, hook
        self.calls = []

    def __call__(self, url, headers, timeout):
        query = parse_qs(urlsplit(url).query)
        offset = int(query["offset"][0])
        self.calls.append({"time": self.clock(), "offset": offset,
                           "url": url, "headers": headers, "timeout": timeout})
        if self.hook:
            self.hook()
        self.clock.value += self.delay
        rows = [match("m%d" % index, query["date_to"][0])
                for index in range(offset, min(offset + 100, self.total))]
        return response(rows, offset, self.total)


class ResultsJobTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.store = MemoryStore()

    def run_job(self, transport=None, **kwargs):
        return job.run_results_job(store=self.store, api_key=TEST_CREDENTIAL,
                                   clock=self.clock, transport=transport or PagingTransport(self.clock),
                                   **kwargs)

    def test_disabled_key_never_touches_storage_or_provider(self):
        result = job.run_results_job(store=self.store, api_key="", clock=self.clock)
        self.assertEqual(result["state"], "disabled")
        self.assertEqual(self.store.reads, 0)
        self.assertEqual(self.store.writes, [])

    def test_batch_is_spaced_bounded_reserved_and_resumable(self):
        transport = PagingTransport(self.clock)
        result = self.run_job(transport)
        self.assertEqual(result["requests"], 12)
        self.assertEqual(result["reserved_today"], 12)
        self.assertEqual(self.store.writes[0]["quota"]["reserved"], 12)
        self.assertEqual(result["state"], "paused")
        self.assertLess(result["elapsed_seconds"], 200)
        self.assertEqual([c["offset"] for c in transport.calls], list(range(0, 1200, 100)))
        self.assertTrue(all(b["time"] - a["time"] >= 12.5
                            for a, b in zip(transport.calls, transport.calls[1:])))
        self.assertIsNone(self.store.document["lease"])
        cache = self.store.document["cache"]
        self.assertEqual(cache["checkpoints"][0]["offset"], 1200)
        self.assertEqual(cache["covered_ranges"], [])
        self.assertEqual(len(cache["records"]), 1200)
        # A new invocation resumes the persisted page; it does not re-fetch page zero.
        self.clock.value += 3600
        resumed = PagingTransport(self.clock)
        result = self.run_job(resumed)
        self.assertEqual(resumed.calls[0]["offset"], 1200)
        self.assertEqual(result["requests"], 8)
        self.assertEqual(result["reserved_today"], 24)  # unused reservations are not refunded
        self.assertEqual(self.store.document["cache"]["checkpoints"], [])
        self.assertEqual(result["state"], "ready")

    def test_daily_ceiling_applies_to_reservations_and_every_attempt(self):
        state = job._new_state(self.clock())
        state["quota"]["reserved"] = 85
        self.store = MemoryStore(state)
        transport = PagingTransport(self.clock)
        first = self.run_job(transport)
        second = self.run_job(transport)
        self.assertEqual(first["requests"], 5)
        self.assertEqual(first["reserved_today"], 90)
        self.assertEqual(second["state"], "budget_exhausted")
        self.assertEqual(len(transport.calls), 5)

    def test_active_lease_does_not_reserve_or_request_again(self):
        state = job._new_state(self.clock())
        state["lease"] = {"owner": "a" * 32, "expires_at": self.clock() + 600}
        self.store = MemoryStore(state)
        transport = PagingTransport(self.clock)
        result = self.run_job(transport)
        self.assertEqual(result["state"], "busy")
        self.assertEqual(transport.calls, [])
        self.assertEqual(self.store.writes, [])

    def test_lost_acquisition_response_keeps_reservation_and_lease(self):
        def lost_response(store, expected, document):
            raise TimeoutError("A committed write lost its response")

        self.store.after_write = lost_response
        transport = PagingTransport(self.clock)
        result = self.run_job(transport)
        self.assertEqual(result["state"], "storage_error")
        self.assertEqual(transport.calls, [])
        self.assertEqual(self.store.document["quota"]["reserved"], 12)
        self.assertIsNotNone(self.store.document["lease"])
        self.store.after_write = None
        self.assertEqual(self.run_job(transport)["state"], "busy")

    def test_pre_request_write_failure_prevents_provider_call(self):
        def fail_second_write(store, expected, document):
            if len(store.writes) == 1:
                raise TimeoutError(TEST_CREDENTIAL)

        self.store.before_write = fail_second_write
        transport = PagingTransport(self.clock)
        result = self.run_job(transport)
        self.assertEqual(result["state"], "storage_error")
        self.assertEqual(transport.calls, [])
        self.assertEqual(self.store.document["quota"]["reserved"], 12)
        self.assertNotIn(TEST_CREDENTIAL, json.dumps(result))

    def test_failed_page_publication_retains_previous_snapshot(self):
        def fail_third_write(store, expected, document):
            if len(store.writes) == 2:
                raise TimeoutError("Unconfirmed page write")

        self.store.before_write = fail_third_write
        transport = PagingTransport(self.clock)
        result = self.run_job(transport)
        self.assertEqual(result["state"], "storage_error")
        self.assertEqual(result["requests"], 1)
        self.assertEqual(self.store.document["snapshot"]["cache"]["records"], [])
        self.assertIsNotNone(self.store.document["lease"])
        self.assertEqual(self.store.document["cache"]["checkpoints"][0]["offset"], 0)

    def test_stale_writer_cannot_overwrite_another_owner(self):
        def replace_owner():
            current, etag = self.store.read()
            current["lease"] = {"owner": "b" * 32, "expires_at": self.clock() + 600}
            current["quota"]["reserved"] += 12
            current["revision"] += 1
            self.store.compare_and_set(etag, current)

        transport = PagingTransport(self.clock, hook=replace_owner)
        result = self.run_job(transport)
        self.assertEqual(result["state"], "storage_error")
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(self.store.document["lease"]["owner"], "b" * 32)
        self.assertEqual(self.store.document["quota"]["reserved"], 24)
        self.assertEqual(self.store.document["snapshot"]["cache"]["records"], [])

    def test_expired_lease_does_not_refund_its_budget(self):
        state = job._new_state(self.clock())
        state["quota"]["reserved"] = 12
        state["lease"] = {"owner": "a" * 32, "expires_at": self.clock() - 1}
        self.store = MemoryStore(state)
        result = self.run_job(PagingTransport(self.clock, total=1))
        self.assertEqual(result["state"], "ready")
        self.assertEqual(result["reserved_today"], 24)
        self.assertEqual(result["requests"], 1)

    def test_new_utc_day_resets_reservation(self):
        state = job._new_state(self.clock())
        state["quota"].update(date="2026-01-19", reserved=90)
        self.store = MemoryStore(state)
        result = self.run_job(PagingTransport(self.clock, total=1))
        self.assertEqual(result["reserved_today"], 12)
        self.assertEqual(self.store.document["quota"]["date"], "2026-01-20")

    def test_malformed_durable_state_fails_closed(self):
        for mutation in (lambda s: s["quota"].update(reserved=-1),
                         lambda s: s["quota"].update(reserved=91),
                         lambda s: s["quota"].update(reserved=True),
                         lambda s: s["quota"].update(date="2027-01-01"),
                         lambda s: s["cache"].update(score_schema_version=1),
                         lambda s: s.update(lease={"owner": "bad", "expires_at": 1})):
            state = job._new_state(self.clock())
            mutation(state)
            self.store = MemoryStore(state)
            transport = PagingTransport(self.clock)
            self.assertEqual(self.run_job(transport)["state"], "storage_error")
            self.assertEqual(transport.calls, [])

    def test_quota_response_is_counted_and_backoff_persisted(self):
        calls = []

        def quota_response(*args):
            calls.append(args)
            return 429, {"retry-after": "120"}, {"error": {"code": "QUOTA_EXCEEDED"}}

        result = self.run_job(quota_response)
        self.assertEqual(result["requests"], 1)
        self.assertEqual(result["state"], "paused")
        self.assertEqual(len(calls), 1)
        cache = self.store.document["cache"]
        self.assertEqual(cache["quota"]["remaining"], 0)
        self.assertEqual(cache["quota"]["backoff_until"], self.clock() + 120)
        self.assertIsNone(self.store.document["lease"])

    def test_failed_http_attempt_is_counted_without_error_or_key_leakage(self):
        def failed_response(*args):
            raise RuntimeError("provider error with " + TEST_CREDENTIAL)

        result = self.run_job(failed_response)
        self.assertEqual(result["requests"], 1)
        self.assertEqual(result["state"], "error")
        self.assertEqual(result["reserved_today"], 12)
        self.assertNotIn(TEST_CREDENTIAL, json.dumps(result) + json.dumps(self.store.document))

    def test_slow_requests_stop_with_publication_time_reserved(self):
        transport = PagingTransport(self.clock, delay=20)
        result = self.run_job(transport)
        self.assertEqual(result["state"], "paused")
        self.assertEqual(result["requests"], 7)
        self.assertLessEqual(result["elapsed_seconds"], 200)

    def test_midnight_crossing_never_spends_a_previous_days_reservation(self):
        self.clock = FakeClock("2026-01-20T23:59:55Z")
        transport = PagingTransport(self.clock, delay=10)
        result = self.run_job(transport)
        self.assertEqual(result["requests"], 1)
        self.assertEqual(result["state"], "paused")
        self.assertEqual(self.store.document["quota"]["date"], "2026-01-20")

    def test_published_projection_preserves_scores_and_excludes_internal_state(self):
        self.run_job(PagingTransport(self.clock, total=1))
        snapshot = self.store.document["snapshot"]
        cache = snapshot["cache"]
        self.assertNotIn("quota", cache)
        self.assertNotIn("checkpoints", cache)
        self.assertNotIn("lease", snapshot)
        self.assertNotIn(TEST_CREDENTIAL, json.dumps(snapshot))
        row = cache["records"][0]
        self.assertEqual(row["surface"], "Hard")
        self.assertEqual(row["score"], "6-1 6-4")
        self.assertEqual(len(row["set_scores"]), 2)
        with patch.object(job.BlobStateStore, "read", return_value=self.store.read()):
            encoded = job.fetch_published_snapshot(PRIVATE_URL, {}, 10, 16 * 1024 * 1024)
        self.assertEqual(json.loads(encoded), snapshot)
        # Exercise the actual hosted-reader boundary, including its schema validation.
        reader = HostedResults(url=PRIVATE_URL, clock=self.clock,
                               fetcher=lambda *args: encoded)
        self.assertEqual(reader.snapshot()["records"][0]["score"], "6-1 6-4")


class BlobProtocolTests(unittest.TestCase):
    def fake_sdk(self, put_error=None, read_payload=b"{}", read_error=None):
        seen = {"clients": [], "puts": [], "reads": []}

        class FakeHTTPX:
            @staticmethod
            def Client(**kwargs):
                seen["clients"].append(kwargs)
                return SimpleNamespace(**kwargs)

        @contextmanager
        def session(httpx_client_factory):
            httpx_client_factory()
            yield

        class FakeBlob:
            @staticmethod
            def put_blob(*args, **kwargs):
                seen["puts"].append((args, kwargs))
                if put_error:
                    raise put_error
                return SimpleNamespace(etag='"next"')

            @staticmethod
            @contextmanager
            def stream_blob(*args, **kwargs):
                seen["reads"].append((args, kwargs))
                if read_error:
                    raise read_error

                class Stream:
                    metadata = SimpleNamespace(etag='"read"', size=len(read_payload))

                    def __iter__(self):
                        yield read_payload

                yield Stream()

        return (FakeHTTPX, session, FakeBlob), seen

    def test_conditional_overwrite_uses_official_header_and_disables_suffix(self):
        sdk, seen = self.fake_sdk()
        with patch.object(job.BlobStateStore, "_sdk", return_value=sdk):
            store = job.BlobStateStore(PRIVATE_URL)
            self.assertEqual(store.compare_and_set('"previous"', {"data": 1}), '"next"')
        self.assertEqual(seen["clients"][0]["headers"], {"x-if-match": '"previous"'})
        args, kwargs = seen["puts"][0]
        self.assertEqual(args[0], "results/state.json")
        self.assertEqual(json.loads(args[1]), {"data": 1})
        self.assertTrue(kwargs["allow_overwrite"])
        self.assertFalse(kwargs["add_random_suffix"])
        self.assertEqual(kwargs["access"], "private")

    def test_first_create_never_allows_overwrite(self):
        sdk, seen = self.fake_sdk()
        with patch.object(job.BlobStateStore, "_sdk", return_value=sdk):
            job.BlobStateStore(PRIVATE_URL).compare_and_set(None, {})
        self.assertEqual(seen["clients"][0]["headers"], {})
        self.assertFalse(seen["puts"][0][1]["allow_overwrite"])

    def test_precondition_failure_is_a_conflict_not_a_retry(self):
        error = RuntimeError("conflict")
        error.status_code = 412
        sdk, seen = self.fake_sdk(put_error=error)
        with patch.object(job.BlobStateStore, "_sdk", return_value=sdk):
            with self.assertRaises(job.StateConflict):
                job.BlobStateStore(PRIVATE_URL).compare_and_set('"old"', {})
        self.assertEqual(len(seen["puts"]), 1)

    def test_origin_read_uses_full_private_url_and_cache_zero(self):
        sdk, seen = self.fake_sdk(read_payload=b'{"revision":1}')
        with patch.object(job.BlobStateStore, "_sdk", return_value=sdk):
            value, etag = job.BlobStateStore(PRIVATE_URL + "?cache=1").read()
        self.assertEqual(value, {"revision": 1})
        self.assertEqual(etag, '"read"')
        self.assertEqual(seen["reads"][0][0][0], PRIVATE_URL + "?cache=0")
        self.assertEqual(seen["reads"][0][1], {"access": "private"})

    def test_missing_object_is_distinct_from_failed_storage(self):
        error = RuntimeError("not found")
        error.status_code = 404
        sdk, _ = self.fake_sdk(read_error=error)
        with patch.object(job.BlobStateStore, "_sdk", return_value=sdk):
            self.assertEqual(job.BlobStateStore(PRIVATE_URL).read(), (None, None))
        sdk, _ = self.fake_sdk(read_error=TimeoutError(TEST_CREDENTIAL))
        with patch.object(job.BlobStateStore, "_sdk", return_value=sdk):
            with self.assertRaises(job.StateUnavailable) as raised:
                job.BlobStateStore(PRIVATE_URL).read()
        self.assertNotIn(TEST_CREDENTIAL, str(raised.exception))

    def test_invalid_json_duplicate_fields_and_oversize_reads_are_rejected(self):
        for payload in (b'{"x":1,"x":2}', b'{"x":NaN}', b"[]", b"not json"):
            sdk, _ = self.fake_sdk(read_payload=payload)
            with patch.object(job.BlobStateStore, "_sdk", return_value=sdk):
                with self.assertRaises(job.StateUnavailable):
                    job.BlobStateStore(PRIVATE_URL).read()
        sdk, _ = self.fake_sdk(read_payload=b'{"x":"too big"}')
        with patch.object(job.BlobStateStore, "_sdk", return_value=sdk), patch.object(job, "MAX_STATE_BYTES", 5):
            with self.assertRaises(job.StateUnavailable):
                job.BlobStateStore(PRIVATE_URL).read()

    def test_private_url_and_path_are_validated_before_sdk_authentication(self):
        for value in ("http://example.private.blob.vercel-storage.com/state.json",
                      "https://example.public.blob.vercel-storage.com/state.json",
                      "https://example.private.blob.vercel-storage.com/../state.json",
                      "https://someone@example.private.blob.vercel-storage.com/state.json",
                      "https://example.private.blob.vercel-storage.com/",
                      "https://example.private.blob.vercel-storage.com:444/state.json"):
            with self.assertRaises(ValueError):
                job.BlobStateStore(value)


if __name__ == "__main__":
    unittest.main()
