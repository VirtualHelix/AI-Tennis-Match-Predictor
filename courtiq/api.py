"""HTTP API for the CourtIQ website and saved predictor.

Run one process: uvicorn courtiq.api:app --host 0.0.0.0 --port 8000 --workers 1
Set COURTIQ_PUBLIC_ORIGIN to the exact HTTPS origin serving this application.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
import queue
import re
import threading
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect
from starlette.responses import HTMLResponse, JSONResponse, Response, StreamingResponse


LOGGER = logging.getLogger("courtiq.hosted")
BODY_LIMIT = 8192
BODY_TIMEOUT = 15
REFRESH_SECONDS = 6 * 60 * 60
GENERIC_ERROR = "The request could not be completed. Please try again shortly."
MODEL_ERROR = "The prediction model is not ready. Please try again shortly."


class PublicError(Exception):
    def __init__(self, message, code="invalid_request", status=400):
        super().__init__(message)
        self.code, self.status = code, status


class StreamDisconnected(Exception):
    pass


def error_response(error):
    return JSONResponse({"error": {"code": error.code, "message": str(error)}},
                        status_code=error.status)


def canonical_origin(value):
    """Parse configured/browser origins, never forwarded request headers."""
    try:
        parsed = urlsplit(value)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.path not in ("", "/") or parsed.query or parsed.fragment
                or not re.fullmatch(r"[a-zA-Z0-9.-]+", parsed.hostname)):
            raise ValueError("Invalid public origin")
        port = parsed.port
        host = parsed.hostname.lower()
        if port is not None and port != (443 if parsed.scheme == "https" else 80):
            host += ":" + str(port)
        return parsed.scheme + "://" + host
    except (ValueError, TypeError, AttributeError):
        raise ValueError("Use an HTTP(S) origin without credentials, a path or query.")


class RequestPolicy:
    """Exact host/origin checks and response headers, including streamed output."""
    def __init__(self, app, origins):
        self.app = app
        self.origins = frozenset(origins)
        self.hosts = frozenset(urlsplit(origin).netloc for origin in origins)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["Cache-Control"] = "no-store"
                headers["X-Content-Type-Options"] = "nosniff"
                headers["Referrer-Policy"] = "same-origin"
            await send(message)

        headers = Headers(scope=scope)
        host = headers.get("host", "").lower()
        raw_hosts = [value for key, value in scope.get("headers", []) if key.lower() == b"host"]
        health_local = scope.get("path") == "/healthz" and host in ("localhost", "127.0.0.1")
        if len(raw_hosts) != 1 or (host not in self.hosts and not health_local):
            response = error_response(PublicError("Unrecognized application host.", "invalid_host", 403))
            return await response(scope, receive, send_with_headers)
        raw_origins = [value for key, value in scope.get("headers", []) if key.lower() == b"origin"]
        if raw_origins:
            try:
                valid_origin = len(raw_origins) == 1 and canonical_origin(headers["origin"]) in self.origins
            except ValueError:
                valid_origin = False
            if not valid_origin:
                response = error_response(PublicError("Requests from other websites are not accepted.",
                                                      "invalid_origin", 403))
                return await response(scope, receive, send_with_headers)
        await self.app(scope, receive, send_with_headers)


def hosted_html(html):
    """Change deployment wording only; preserve all reviewed UI and calculations."""
    replacements = {
        "LOCAL MODEL READY": "MODEL READY",
        "LOCAL MODEL · MATCH ANALYSIS": "SAVED MODEL · MATCH ANALYSIS",
        "your local model": "the saved model",
        "your local model report": "the saved model report",
        "local model report": "saved model report",
        "Connect local player list": "Player list unavailable",
        "Connect your local model": "Connect the prediction service",
        "Cannot reach the local model. Check that the CourtIQ server is running, then reconnect.":
            "Cannot reach the prediction service. Please try again shortly.",
        "Check the local server terminal.": "Please try again shortly.",
        "Interface preview · run locally to use your trained model": "Prediction service unavailable",
        "All three views are ready to explore; Model Report includes your reported validation results.":
            "Please retry once the prediction service is available.",
    }
    for old, new in replacements.items():
        html = html.replace(old, new)
    return html


async def json_payload(request):
    content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if content_type != "application/json":
        raise PublicError("Send application/json.", "invalid_content_type", 415)
    length_header = request.headers.get("content-length")
    expected = None
    if length_header is not None:
        if not re.fullmatch(r"\d+", length_header):
            raise PublicError("Invalid request length.")
        expected = int(length_header)
        if not 0 < expected <= BODY_LIMIT:
            raise PublicError("Request body must be between 1 and 8192 bytes.", "invalid_body", 413)

    async def collect():
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > BODY_LIMIT:
                raise PublicError("Request body must be between 1 and 8192 bytes.", "invalid_body", 413)
            body.extend(chunk)
        return bytes(body)

    try:
        body = await asyncio.wait_for(collect(), timeout=BODY_TIMEOUT)
    except asyncio.TimeoutError:
        raise PublicError("The request body timed out.", "timeout", 408)
    except ClientDisconnect:
        raise PublicError("The request was disconnected.", "invalid_body", 400)
    if not body:
        raise PublicError("Request body must be between 1 and 8192 bytes.", "invalid_body", 413)
    if expected is not None and len(body) != expected:
        raise PublicError("The request length does not match its body.", "invalid_body", 400)
    try:
        payload = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeError):
        raise PublicError("Request body must contain valid JSON.")
    if not isinstance(payload, dict):
        raise PublicError("Send a JSON object with two players and match conditions.")
    return payload


def create_app(service_factory=None, *, html=None, api_error_type=None,
               public_origin=None, port=None, max_inflight=4, refresh_seconds=REFRESH_SECONDS):
    """Factory keeps tests independent of model files and public provider calls."""
    port = int(port if port is not None else os.environ.get("PORT", "8000"))
    if not 1 <= port <= 65535 or not 1 <= max_inflight <= 32 or refresh_seconds <= 0:
        raise ValueError("Invalid hosting port, request limit, or refresh interval.")
    configured = public_origin if public_origin is not None else (
        os.environ.get("COURTIQ_PUBLIC_ORIGIN"))
    origins = {"http://127.0.0.1:" + str(port), "http://localhost:" + str(port)}
    if configured:
        origins.add(canonical_origin(configured))

    @asynccontextmanager
    async def lifespan(app):
        app.state.service = None
        app.state.api_error_type = api_error_type or ()
        app.state.html = hosted_html(html or "")
        app.state.gate = threading.BoundedSemaphore(max_inflight)
        app.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="courtiq-inference")
        app.state.stream_stops = set()
        stop = asyncio.Event()
        periodic = None
        try:
            factory = service_factory
            if factory is None:
                from courtiq import server as runtime
                app.state.api_error_type = runtime.APIError
                app.state.html = hosted_html(runtime.HTML_TEMPLATE)
                root = Path(os.environ.get("COURTIQ_ROOT", Path(__file__).resolve().parents[1]))
                factory = lambda: runtime.CourtIQService(
                    root, model_path=os.environ.get("COURTIQ_MODEL_PATH"),
                    source_path=os.environ.get("COURTIQ_MODEL_CODE"))
            app.state.service = await run_in_threadpool(factory)
            if not app.state.service.status().get("ready"):
                LOGGER.error("CourtIQ model could not load; readiness remains unavailable.")
            elif app.state.service.results_status().get("configured"):
                await run_in_threadpool(app.state.service.results_feed.refresh)

                async def refresh_loop():
                    while not stop.is_set():
                        try:
                            await asyncio.wait_for(stop.wait(), timeout=refresh_seconds)
                        except asyncio.TimeoutError:
                            try:
                                await run_in_threadpool(app.state.service.results_feed.refresh)
                            except Exception:
                                LOGGER.exception("Results refresh failed; keeping recorded history.")
                periodic = asyncio.create_task(refresh_loop())
        except Exception:
            LOGGER.exception("CourtIQ startup did not complete; readiness remains unavailable.")
        try:
            yield
        finally:
            stop.set()
            for disconnected in list(app.state.stream_stops):
                disconnected.set()
            if periodic is not None:
                await periodic
            if app.state.service is not None:
                await run_in_threadpool(app.state.service.results_feed.close)
            app.state.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(RequestPolicy, origins=origins)

    def ready_service():
        service = getattr(app.state, "service", None)
        if service is None or not service.status().get("ready"):
            raise PublicError(MODEL_ERROR, "model_unavailable", 503)
        return service

    def invoke(method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except StreamDisconnected:
            raise
        except PublicError:
            raise
        except Exception as error:
            if isinstance(error, app.state.api_error_type):
                message = MODEL_ERROR if error.code == "model_unavailable" else str(error)
                raise PublicError(message, error.code, error.status)
            LOGGER.exception("CourtIQ request failed.")
            raise PublicError(GENERIC_ERROR, "server_error", 500)

    def submit_inference(function):
        if not app.state.gate.acquire(blocking=False):
            raise PublicError("The prediction service is busy. Please try again shortly.", "busy", 429)
        try:
            future = asyncio.get_running_loop().run_in_executor(app.state.executor, function)
        except BaseException:
            app.state.gate.release()
            raise
        # A disconnected caller must not release capacity while its worker is
        # still calculating. Shielding consumers leaves this future alive.
        future.add_done_callback(lambda completed: app.state.gate.release())
        return future

    @app.exception_handler(PublicError)
    async def public_exception(request, error):
        return error_response(error)

    @app.exception_handler(HTTPException)
    async def http_exception(request, error):
        code = "not_found" if error.status_code == 404 else "method_not_allowed"
        return error_response(PublicError(str(error.detail), code, error.status_code))

    @app.exception_handler(Exception)
    async def unexpected_exception(request, error):
        LOGGER.error("Unexpected application error.", exc_info=(type(error), error, error.__traceback__))
        return error_response(PublicError(GENERIC_ERROR, "server_error", 500))

    @app.get("/healthz")
    async def health():
        ready_service()
        return {"status": "ready", "model_ready": True}

    @app.get("/", response_class=HTMLResponse)
    @app.get("/index.html", response_class=HTMLResponse)
    async def home():
        ready_service()
        if not app.state.html:
            raise PublicError("The application interface is unavailable.", "interface_unavailable", 503)
        return HTMLResponse(app.state.html)

    @app.get("/favicon.ico")
    async def favicon():
        return Response(status_code=204)

    @app.get("/api/status")
    async def status():
        return await run_in_threadpool(invoke, ready_service().status)

    @app.get("/api/players")
    async def players():
        return await run_in_threadpool(invoke, ready_service().players)

    @app.get("/api/evaluation")
    async def evaluation():
        return await run_in_threadpool(invoke, ready_service().evaluation)

    @app.get("/api/methodology")
    async def methodology():
        return await run_in_threadpool(invoke, ready_service().methodology)

    @app.get("/api/results/status")
    async def results_status():
        return await run_in_threadpool(invoke, ready_service().results_status)

    @app.post("/api/results/sync")
    async def results_sync(request: Request):
        payload = await json_payload(request)
        return await run_in_threadpool(invoke, ready_service().sync_results, payload)

    @app.get("/api/compare")
    async def compare(request: Request):
        service = ready_service()
        items = list(request.query_params.multi_items())
        if len({key for key, value in items}) != len(items):
            raise PublicError("Each query field may appear only once.")
        payload = dict(items)
        if "best_of" in payload:
            if payload["best_of"] not in ("3", "5"):
                raise PublicError("Match format must be best of 3 or best of 5.")
            payload["best_of"] = int(payload["best_of"])
        return await asyncio.shield(submit_inference(lambda: invoke(service.compare, payload)))

    @app.post("/api/predict")
    async def predict(request: Request):
        payload = await json_payload(request)
        service = ready_service()
        return await asyncio.shield(submit_inference(lambda: invoke(service.predict, payload)))

    @app.post("/api/predict/stream")
    async def stream_predict(request: Request):
        payload = await json_payload(request)
        service = ready_service()
        events = queue.Queue(maxsize=16)
        disconnected = threading.Event()

        def emit(event):
            encoded = (json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
            while not disconnected.is_set():
                try:
                    events.put(encoded, timeout=0.1)
                    return
                except queue.Full:
                    continue
            raise StreamDisconnected()

        def work():
            try:
                if disconnected.is_set():
                    return
                result = invoke(service.predict, payload, progress=emit)
                emit({"type": "result", "data": result})
            except StreamDisconnected:
                pass
            except PublicError as error:
                try:
                    emit({"type": "error", "error": str(error), "code": error.code})
                except StreamDisconnected:
                    pass
            except Exception:
                # A response serialization error also needs a terminal event;
                # headers may already have reached the visitor.
                LOGGER.exception("Could not encode prediction stream.")
                try:
                    emit({"type": "error", "error": GENERIC_ERROR, "code": "server_error"})
                except StreamDisconnected:
                    pass

        future = submit_inference(work)
        app.state.stream_stops.add(disconnected)

        async def stream():
            try:
                while True:
                    try:
                        event = await run_in_threadpool(events.get, True, 0.25)
                    except queue.Empty:
                        if future.done():
                            # Read worker exceptions, if any, so none are lost.
                            await asyncio.shield(future)
                            break
                        continue
                    yield event
            finally:
                disconnected.set()
                app.state.stream_stops.discard(disconnected)

        return StreamingResponse(stream(), media_type="application/x-ndjson",
                                 headers={"X-Accel-Buffering": "no"})

    return app


app = create_app()
