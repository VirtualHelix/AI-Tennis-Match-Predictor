"""Adapter contract tests. These fake records never ship as predictions."""
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from courtiq import api as adapter

ORIGIN = "https://court.example"
PAYLOAD = {"player_a": "A", "player_b": "B", "surface": "Clay"}


class APIError(Exception):
    def __init__(self, message, code="invalid_request", status=400):
        super().__init__(message)
        self.code, self.status = code, status


class Feed:
    def __init__(self):
        self.refreshes = 0
        self.closed = False
        self.refreshed_twice = threading.Event()

    def refresh(self):
        self.refreshes += 1
        if self.refreshes >= 2:
            self.refreshed_twice.set()
        return {"state": "syncing"}

    def close(self):
        self.closed = True


class Service:
    def __init__(self, ready=True, configured=False):
        self.ready = ready
        self.configured = configured
        self.results_feed = Feed()
        self.predictions = []
        self.failure = None
        self.started = threading.Event()
        self.block = None

    def status(self):
        return {"ready": self.ready, "app_version": "test"}

    def players(self):
        return {"players": [{"name": "A"}, {"name": "B"}]}

    def evaluation(self):
        return {"source": "saved_model", "report": {"validation_year": 2025}}

    def methodology(self):
        return {"source": "saved_model", "features": ["a", "b"]}

    def results_status(self):
        return {"configured": self.configured, "state": "ready"}

    def sync_results(self, payload):
        if payload:
            raise APIError("Send an empty JSON object to refresh match results.")
        return self.results_feed.refresh()

    def compare(self, payload):
        return {"comparison": payload}

    def predict(self, payload, progress=None):
        self.predictions.append(payload)
        if progress:
            progress({"type": "progress", "stage": "predicting", "message": "Calculating", "progress": 18})
        self.started.set()
        if self.block is not None and not self.block.wait(timeout=4):
            raise RuntimeError("Test release was not signaled")
        if self.failure:
            raise self.failure
        if progress:
            progress({"type": "progress", "stage": "complete", "message": "Ready", "progress": 100})
        return {"prediction": {"probability_a": 0.6, "probability_b": 0.4}, "input": payload}


def application(service=None, **changes):
    service = service or Service()
    args = {"service_factory": lambda: service, "api_error_type": APIError,
            "html": "<html>LOCAL MODEL READY · predictions use your local model.</html>",
            "public_origin": ORIGIN}
    args.update(changes)
    return adapter.create_app(**args), service


class SnapshotFeed(Feed):
    def __init__(self):
        super().__init__()
        self.revision = 1
        self.published_revision = 1
        self.events = []

    def refresh(self):
        super().refresh()
        self.revision = self.published_revision
        self.events.append(("refresh", self.revision))
        return {"state": "ready", "revision": self.revision}


class SnapshotService(Service):
    """A service cache must see the newly published revision before lookup."""
    def __init__(self):
        super().__init__(configured=True)
        self.results_feed = SnapshotFeed()
        self.cached_revision = None

    def results_status(self):
        return {"configured": True, "state": "ready", "read_only": True,
                "revision": self.results_feed.revision}

    def predict(self, payload, progress=None):
        revision = self.results_feed.revision
        self.results_feed.events.append(("predict", revision))
        result = super().predict(payload, progress)
        result.update(revision=revision, cached=revision == self.cached_revision)
        self.cached_revision = revision
        return result

    def compare(self, payload):
        self.results_feed.events.append(("compare", self.results_feed.revision))
        return {"comparison": payload, "revision": self.results_feed.revision}


class HostingTests(unittest.TestCase):
    JOB_PATH = "/api/internal/refresh-results"

    def test_scheduled_job_requires_a_configured_exact_bearer_secret(self):
        calls = []
        with patch.dict(os.environ, {"CRON_SECRET": "test-secret", "COURTIQ_ENABLE_RESULTS_JOB": "1"}, clear=True):
            app, _ = application(results_job=lambda: calls.append(True))
            with TestClient(app, base_url=ORIGIN) as client:
                for headers in ({}, {"Authorization": "Bearer wrong"},
                                {"Authorization": "Basic test-secret"},
                                {"Authorization": "Bearer test-secret extra"}):
                    response = client.post(self.JOB_PATH, headers=headers)
                    self.assertEqual(response.status_code, 401)
                    self.assertEqual(response.json()["error"]["code"], "unauthorized")
                with patch.dict(os.environ, {"CRON_SECRET": ""}):
                    self.assertEqual(client.post(self.JOB_PATH, headers={"Authorization": "Bearer "}).status_code, 401)
        self.assertEqual(calls, [])

    def test_scheduled_job_rejects_non_ascii_authorization_as_unauthorized(self):
        calls = []
        with patch.dict(os.environ, {"CRON_SECRET": "test-secret", "COURTIQ_ENABLE_RESULTS_JOB": "1"}, clear=True):
            app, _ = application(results_job=lambda: calls.append(True))
            with TestClient(app, base_url=ORIGIN, raise_server_exceptions=False) as client:
                response = client.post(self.JOB_PATH, headers=[(b"authorization", b"Bearer \xff")])
                self.assertEqual(response.status_code, 401)
                self.assertEqual(response.json()["error"]["code"], "unauthorized")
        self.assertEqual(calls, [])

    def test_scheduled_job_stays_disabled_in_preview_without_enable_flag(self):
        calls = []
        settings = {"VERCEL": "1", "VERCEL_ENV": "preview", "CRON_SECRET": "test-secret"}
        with patch.dict(os.environ, settings, clear=True):
            app, _ = application(results_job=lambda: calls.append(True))
            with TestClient(app, base_url=ORIGIN) as client:
                response = client.post(self.JOB_PATH, headers={"Authorization": "Bearer test-secret"})
                self.assertEqual(response.status_code, 409)
                self.assertEqual(response.json()["error"]["code"], "disabled")
        self.assertEqual(calls, [])

    def test_authenticated_enabled_schedule_executes_worker_once(self):
        calls = []
        expected = {"state": "published", "requests_reserved": 3}
        def worker():
            calls.append(True)
            return expected
        settings = {"VERCEL": "1", "VERCEL_ENV": "production", "CRON_SECRET": "test-secret",
                    "COURTIQ_ENABLE_RESULTS_JOB": "1"}
        with patch.dict(os.environ, settings, clear=True):
            app, _ = application(results_job=worker)
            with TestClient(app, base_url=ORIGIN) as client:
                response = client.post(self.JOB_PATH, headers={"Authorization": "Bearer test-secret"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), expected)
        self.assertEqual(calls, [True])

    def test_vercel_system_domains_are_allowed_exactly_without_wildcards(self):
        domains = ["courtiq-abc123.vercel.app", "courtiq-git-main-team.vercel.app", "courtiq.vercel.app"]
        settings = dict(zip(("VERCEL_URL", "VERCEL_BRANCH_URL", "VERCEL_PROJECT_PRODUCTION_URL"), domains))
        settings["VERCEL"] = "1"
        with patch.dict(os.environ, settings, clear=True):
            app, _ = application()
            with TestClient(app, base_url=ORIGIN) as client:
                for domain in domains:
                    origin = "https://" + domain
                    response = client.post(origin + "/api/predict", json=PAYLOAD, headers={"Origin": origin})
                    self.assertEqual(response.status_code, 200, domain)
                for domain in (domains[0] + ".evil.example", "other-team.vercel.app"):
                    response = client.post("https://" + domain + "/api/predict", json=PAYLOAD,
                                           headers={"Origin": "https://" + domains[0],
                                                    "X-Forwarded-Host": domains[0]})
                    self.assertEqual(response.status_code, 403)
                    self.assertEqual(response.json()["error"]["code"], "invalid_host")
                response = client.post("https://" + domains[0] + "/api/predict", json=PAYLOAD,
                                       headers={"Origin": "http://" + domains[0]})
                self.assertEqual(response.status_code, 403)
                self.assertEqual(response.json()["error"]["code"], "invalid_origin")

    def test_vercel_domains_are_not_trusted_outside_vercel(self):
        with patch.dict(os.environ, {"VERCEL_URL": "courtiq.vercel.app"}, clear=True):
            app, _ = application()
            with TestClient(app, base_url=ORIGIN) as client:
                response = client.get("https://courtiq.vercel.app/healthz")
                self.assertEqual(response.status_code, 403)
                self.assertEqual(response.json()["error"]["code"], "invalid_host")

    def test_snapshot_refresh_precedes_cached_prediction_lookup(self):
        service = SnapshotService()
        app, _ = application(service, background_refresh=False)
        with TestClient(app, base_url=ORIGIN) as client:
            service.results_feed.events.clear()
            initial = client.post("/api/predict", json=PAYLOAD).json()
            repeated = client.post("/api/predict", json=PAYLOAD).json()
            service.results_feed.published_revision = 2
            refreshed = client.post("/api/predict", json=PAYLOAD).json()
        self.assertEqual([initial["cached"], repeated["cached"], refreshed["cached"]], [False, True, False])
        self.assertEqual([initial["revision"], repeated["revision"], refreshed["revision"]], [1, 1, 2])
        self.assertEqual(service.results_feed.events,
                         [("refresh",1),("predict",1),("refresh",1),("predict",1),("refresh",2),("predict",2)])

    def test_snapshot_refresh_precedes_stream_compare_and_status(self):
        service = SnapshotService()
        app, _ = application(service, background_refresh=False)
        with TestClient(app, base_url=ORIGIN) as client:
            service.results_feed.events.clear()
            service.results_feed.published_revision = 2
            stream = client.post("/api/predict/stream", json=PAYLOAD)
            events = [json.loads(line) for line in stream.text.splitlines()]
            self.assertEqual(events[-1]["data"]["revision"], 2)
            service.results_feed.published_revision = 3
            self.assertEqual(client.get("/api/compare", params=PAYLOAD).json()["revision"], 3)
            service.results_feed.published_revision = 4
            self.assertEqual(client.get("/api/results/status").json()["revision"], 4)
        self.assertEqual(service.results_feed.events,
                         [("refresh",2),("predict",2),("refresh",3),("compare",3),("refresh",4)])

    def test_read_only_snapshot_never_starts_a_periodic_background_reader(self):
        service = SnapshotService()
        app, _ = application(service, background_refresh=True, refresh_seconds=0.01)
        with TestClient(app, base_url=ORIGIN):
            self.assertEqual(service.results_feed.refreshes, 1)
            self.assertFalse(service.results_feed.refreshed_twice.wait(timeout=0.08))
        self.assertEqual(service.results_feed.refreshes, 1)
        self.assertTrue(service.results_feed.closed)

    def test_vercel_disables_periodic_background_refresh_by_default(self):
        service = Service(configured=True)
        with patch.dict(os.environ, {"VERCEL":"1"}, clear=True):
            app, _ = application(service, refresh_seconds=0.01)
            with TestClient(app, base_url=ORIGIN):
                self.assertEqual(service.results_feed.refreshes, 1)
                self.assertFalse(service.results_feed.refreshed_twice.wait(timeout=0.08))
        self.assertEqual(service.results_feed.refreshes, 1)
        self.assertTrue(service.results_feed.closed)


class AdapterTests(unittest.TestCase):
    def test_readiness_and_hosted_interface_preserve_working_model(self):
        app, service = application()
        with TestClient(app, base_url=ORIGIN) as client:
            self.assertEqual(client.get("/healthz").json(), {"status": "ready", "model_ready": True})
            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn("MODEL READY", page.text)
            self.assertNotIn("LOCAL MODEL", page.text)
            self.assertNotIn("your local model", page.text)
            self.assertEqual(page.headers["x-content-type-options"], "nosniff")
            self.assertEqual(page.headers["cache-control"], "no-store")
        self.assertTrue(service.results_feed.closed)

    def test_missing_model_is_503_and_has_no_preview_success(self):
        app, service = application(Service(ready=False))
        with self.assertLogs("courtiq.hosted", level="ERROR"):
            with TestClient(app, base_url=ORIGIN) as client:
                for path in ("/", "/healthz", "/api/status", "/api/players", "/api/evaluation"):
                    response = client.get(path)
                    self.assertEqual(response.status_code, 503, path)
                    self.assertEqual(response.json()["error"]["code"], "model_unavailable")
                self.assertEqual(client.post("/api/predict", json=PAYLOAD).status_code, 503)
        self.assertEqual(service.predictions, [])

    def test_failed_startup_still_exposes_unready_health_without_private_errors(self):
        def fail():
            raise RuntimeError("private/server/path secret detail")
        app, _ = application(service_factory=fail)
        with self.assertLogs("courtiq.hosted", level="ERROR"):
            with TestClient(app, base_url=ORIGIN) as client:
                response = client.get("/healthz")
                self.assertEqual(response.status_code, 503)
                self.assertNotIn("private", response.text)

    def test_all_read_routes_and_compare_contract(self):
        app, service = application()
        with TestClient(app, base_url=ORIGIN) as client:
            for path, expected in (("/api/status", service.status()), ("/api/players", service.players()),
                                   ("/api/evaluation", service.evaluation()), ("/api/methodology", service.methodology()),
                                   ("/api/results/status", service.results_status())):
                self.assertEqual(client.get(path).json(), expected)
            response = client.get("/api/compare", params=dict(PAYLOAD, best_of=5))
            self.assertEqual(response.json()["comparison"], dict(PAYLOAD, best_of=5))
            self.assertEqual(client.get("/api/compare?player_a=A&player_a=B").status_code, 400)
            self.assertEqual(client.get("/api/compare?best_of=4").status_code, 400)
            self.assertEqual(client.get("/does-not-exist").status_code, 404)
            self.assertEqual(client.get("/favicon.ico").status_code, 204)

    def test_results_sync_uses_empty_object_and_existing_api_errors(self):
        app, service = application()
        with TestClient(app, base_url=ORIGIN) as client:
            self.assertEqual(client.post("/api/results/sync", json={}).json(), {"state": "syncing"})
            bad = client.post("/api/results/sync", json={"unexpected": True})
            self.assertEqual(bad.status_code, 400)
            self.assertEqual(bad.json()["error"]["code"], "invalid_request")
            self.assertEqual(service.results_feed.refreshes, 1)

    def test_json_limits_content_type_and_exact_boundary(self):
        app, service = application()
        body = json.dumps(PAYLOAD)
        with TestClient(app, base_url=ORIGIN) as client:
            headers = {"Content-Type": "application/json"}
            cases = [("invalid", headers, 400), ("[]", headers, 400), ("", headers, 413),
                     (body, {"Content-Type": "text/plain"}, 415),
                     (body.ljust(adapter.BODY_LIMIT + 1), headers, 413)]
            for content, request_headers, expected in cases:
                self.assertEqual(client.post("/api/predict", content=content, headers=request_headers).status_code, expected)
            self.assertEqual(client.post("/api/predict", content=body.ljust(adapter.BODY_LIMIT), headers=headers).status_code, 200)
        self.assertEqual(service.predictions, [PAYLOAD])

    def test_chunked_body_is_bounded_even_without_content_length(self):
        app, service = application()
        with TestClient(app, base_url=ORIGIN) as client:
            response = client.post("/api/predict", content=iter([b"{" + b" " * 6000, b" " * 6000 + b"}"]),
                                   headers={"Content-Type": "application/json"})
            self.assertEqual(response.status_code, 413)
        self.assertEqual(service.predictions, [])

    def test_exact_origin_and_host_cannot_be_replaced_by_forwarded_headers(self):
        app, _ = application()
        with TestClient(app, base_url=ORIGIN) as client:
            self.assertEqual(client.post("/api/predict", json=PAYLOAD, headers={"Origin": ORIGIN}).status_code, 200)
            for origin in ("https://evil.example", "null", ORIGIN + ".evil.example"):
                response = client.post("/api/predict", json=PAYLOAD, headers={"Origin": origin})
                self.assertEqual(response.status_code, 403)
                self.assertEqual(response.json()["error"]["code"], "invalid_origin")
            response = client.post("/api/predict", json=PAYLOAD,
                                   headers={"Host": "evil.example", "X-Forwarded-Host": "court.example", "Origin": ORIGIN})
            self.assertEqual(response.status_code, 403)
            self.assertEqual(response.json()["error"]["code"], "invalid_host")
            self.assertEqual(client.get("http://localhost:8000/healthz").status_code, 200)

    def test_prediction_and_stream_have_same_payload_and_one_terminal_result(self):
        app, _ = application()
        with TestClient(app, base_url=ORIGIN) as client:
            standard = client.post("/api/predict", json=PAYLOAD).json()
            response = client.post("/api/predict/stream", json=PAYLOAD)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "application/x-ndjson")
            self.assertEqual(response.headers["x-accel-buffering"], "no")
            self.assertTrue(response.content.endswith(b"\n"))
            events = [json.loads(line) for line in response.text.splitlines()]
            self.assertEqual([event["type"] for event in events], ["progress", "progress", "result"])
            self.assertEqual(events[-1]["data"], standard)
            self.assertEqual(events[0]["progress"], 18)

    def test_api_errors_keep_status_code_and_stream_terminal_error(self):
        service = Service()
        service.failure = APIError("Choose two different players.", "same_player", 422)
        app, _ = application(service)
        with TestClient(app, base_url=ORIGIN) as client:
            response = client.post("/api/predict", json=PAYLOAD)
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["error"]["code"], "same_player")
            events = [json.loads(line) for line in client.post("/api/predict/stream", json=PAYLOAD).text.splitlines()]
            self.assertEqual(events[-1], {"type": "error", "error": "Choose two different players.", "code": "same_player"})
            self.assertEqual(sum(event["type"] in ("result", "error") for event in events), 1)

    def test_unexpected_model_errors_are_generic_in_both_responses(self):
        service = Service()
        service.failure = RuntimeError("internal secret or path")
        app, _ = application(service)
        with TestClient(app, base_url=ORIGIN) as client, self.assertLogs("courtiq.hosted", level="ERROR"):
            response = client.post("/api/predict", json=PAYLOAD)
            self.assertEqual(response.status_code, 500)
            self.assertNotIn("internal secret", response.text)
            stream = client.post("/api/predict/stream", json=PAYLOAD)
            self.assertNotIn("internal secret", stream.text)
            self.assertEqual(json.loads(stream.text.splitlines()[-1])["code"], "server_error")

    def test_nonfinite_output_returns_an_error_instead_of_an_unterminated_stream(self):
        service = Service()
        service.predict = lambda payload, progress=None: {"probability": float("nan")}
        app, _ = application(service)
        with TestClient(app, base_url=ORIGIN, raise_server_exceptions=False) as client:
            with self.assertLogs("courtiq.hosted", level="ERROR"):
                ordinary = client.post("/api/predict", json=PAYLOAD)
                streamed = client.post("/api/predict/stream", json=PAYLOAD)
            self.assertEqual(ordinary.status_code, 500)
            self.assertEqual(ordinary.json()["error"]["code"], "server_error")
            events = [json.loads(line) for line in streamed.text.splitlines()]
            self.assertEqual(events, [{"type": "error", "error": adapter.GENERIC_ERROR, "code": "server_error"}])

    def test_pending_inference_capacity_is_bounded(self):
        service = Service()
        service.block = threading.Event()
        app, _ = application(service, max_inflight=2)
        with TestClient(app, base_url=ORIGIN) as client, ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [pool.submit(client.post, "/api/predict", json=PAYLOAD) for _ in range(3)]
            try:
                first = next(as_completed(jobs, timeout=2)).result()
                self.assertEqual(first.status_code, 429)
                self.assertEqual(first.json()["error"]["code"], "busy")
            finally:
                service.block.set()
            self.assertEqual(sorted(job.result(timeout=2).status_code for job in jobs), [200, 200, 429])

    def test_lifespan_initial_periodic_sync_and_close(self):
        service = Service(configured=True)
        app, _ = application(service, refresh_seconds=0.02)
        with TestClient(app, base_url=ORIGIN):
            self.assertTrue(service.results_feed.refreshed_twice.wait(timeout=2))
        self.assertTrue(service.results_feed.closed)


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_progress_is_sent_before_prediction_finishes(self):
        service = Service()
        service.block = threading.Event()
        app, _ = application(service)
        body = json.dumps(PAYLOAD).encode()
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                 "http_version": "1.1", "method": "POST", "scheme": "https", "path": "/api/predict/stream",
                 "raw_path": b"/api/predict/stream", "query_string": b"", "root_path": "",
                 "server": ("court.example", 443), "client": ("127.0.0.1", 1),
                 "headers": [(b"host", b"court.example"), (b"content-type", b"application/json"),
                             (b"content-length", str(len(body)).encode())]}
        sent = []
        progress_seen = asyncio.Event()
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.body" and b'"type":"progress"' in message.get("body", b""):
                progress_seen.set()

        async with app.router.lifespan_context(app):
            request = asyncio.create_task(app(scope, receive, send))
            try:
                await asyncio.wait_for(progress_seen.wait(), timeout=2)
                self.assertFalse(request.done())
                self.assertFalse(service.block.is_set())
            finally:
                service.block.set()
            await asyncio.wait_for(request, timeout=2)
        chunks = b"".join(message.get("body", b"") for message in sent)
        events = [json.loads(line) for line in chunks.splitlines()]
        self.assertEqual(events[-1]["type"], "result")


if __name__ == "__main__":
    unittest.main()
