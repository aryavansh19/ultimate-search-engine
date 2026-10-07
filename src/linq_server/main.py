"""FastAPI app for the LinQ enrichment + embedding service.

Run it:

    python -m linq_server                      # loopback, no auth, for development
    LINQ_REQUIRE_AUTH=1 python -m linq_server --host 0.0.0.0

Endpoints:

    GET  /health        liveness + what models are configured. No auth, no user data.
    POST /v1/analyze    URL -> Gemini enrichment + Nemotron vectors. The save path.
    POST /v1/embed      text -> vectors. The search path.

Every route that spends money is authenticated and rate limited. Extraction, enrichment and
embedding are all synchronous blocking I/O, so they run in a worker thread — doing them
inline would block the event loop and make one slow video upload freeze every other request.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .auth import AccessPolicy
from .diagnostics import (
    analyze_request_payload,
    configure_logging,
    log_json,
    payload_logging_enabled,
)
from .models import AnalyzeRequest, AnalyzeResponse, EmbedRequest, EmbedResponse
from .service import AnalysisService, ServiceError

log = logging.getLogger("linq.main")

service: AnalysisService | None = None
policy = AccessPolicy.from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    configure_logging()
    service = AnalysisService()
    caps = service.capabilities()
    log.info("linq service ready: %s", caps)
    if not policy.configured and policy.require_auth:
        log.error(
            "No credentials configured. Set SUPABASE_URL, SUPABASE_JWT_SECRET, "
            "or LINQ_API_TOKEN; or set LINQ_REQUIRE_AUTH=0 for loopback development."
        )
    try:
        yield
    finally:
        if service is not None:
            service.close()
            service = None


app = FastAPI(title="LinQ Enrichment", version="1.0.0", lifespan=lifespan)

# The iOS app is not a browser and sends no Origin, so CORS is irrelevant to it. It is
# restricted here anyway so that a stray web page cannot drive this service from a user's
# browser using ambient credentials.
_origins = [o for o in (os.getenv("LINQ_CORS_ORIGINS") or "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or [],
    allow_methods=["POST", "GET"],
    allow_headers=["authorization", "content-type", "x-access-token"],
)


@app.middleware("http")
async def trace_requests(request: Request, call_next):
    """Give every API call one ID shared by all Render log lines."""
    trace_id = uuid4().hex[:12]
    request.state.trace_id = trace_id
    started = time.perf_counter()
    is_api = request.url.path.startswith("/v1/")
    if is_api:
        log.info(
            "[search-pipeline:%s] request started method=%s path=%s",
            trace_id,
            request.method,
            request.url.path,
        )
    try:
        response = await call_next(request)
    except Exception:  # noqa: BLE001
        log.exception("[search-pipeline:%s] unhandled request failure", trace_id)
        raise
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    response.headers["X-LinQ-Trace-ID"] = trace_id
    if is_api:
        log.info(
            "[search-pipeline:%s] request finished status=%s duration_ms=%s",
            trace_id,
            response.status_code,
            elapsed_ms,
        )
    return response


def _service() -> AnalysisService:
    if service is None:  # pragma: no cover - only during startup/shutdown
        raise HTTPException(status_code=503, detail="service not ready")
    return service


@app.exception_handler(ServiceError)
async def _service_error(request: Request, exc: ServiceError) -> JSONResponse:
    # 502: we are a healthy gateway reporting that an upstream model failed. Distinguishing
    # this from a 4xx matters to the client, which should retry a 502 but not a 400.
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe. Carries configuration, never user data."""
    payload: dict[str, Any] = {
        "ok": True,
        "auth": policy.describe(),
        "diagnostics": {
            "payload_logging": payload_logging_enabled(),
            "logger_level": logging.getLevelName(logging.getLogger("linq").level),
        },
    }
    if service is not None:
        payload["capabilities"] = service.capabilities()
    return payload


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, http: Request) -> AnalyzeResponse:
    """Extract a link, understand it with Gemini, and return vectors for it.

    This is the expensive one — a video pass costs real money and can take a minute — so it
    is both authenticated and rate limited per caller.
    """
    trace_id = getattr(http.state, "trace_id", "unknown")
    caller = policy.identify(http)
    policy.charge(caller)
    svc = _service()
    log.info(
        "[search-pipeline:%s] authenticated caller=%s user_id=%s url=%s "
        "force_media=%s embed=%s",
        trace_id,
        caller.kind,
        caller.user_id or "shared-token",
        request.url,
        request.force_media,
        request.embed,
    )
    log_json(
        log,
        trace_id,
        "1. request JSON (authorization and HTML excluded)",
        analyze_request_payload(request),
    )
    try:
        result = await run_in_threadpool(lambda: svc.analyze(request, trace_id=trace_id))
    except Exception:  # noqa: BLE001
        log.exception("[search-pipeline:%s] analyze failed url=%s", trace_id, request.url)
        raise
    log_json(
        log,
        trace_id,
        "6. response JSON returned to LinQ",
        result.model_dump(mode="json"),
    )
    log.info(
        "[search-pipeline:%s] response ready degraded=%s details=%s tags=%s entities=%s",
        trace_id,
        result.degraded,
        len(result.details),
        len(result.tags),
        sum(len(values) for values in result.entities.model_dump().values()),
    )
    return result


@app.post("/v1/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest, http: Request) -> EmbedResponse:
    """Vectorise text. Used for search queries, and for re-embedding after an edit.

    Not charged against the analysis budget: this is cheap, free on the current model, and
    rate limiting the search path would make the app feel broken.
    """
    trace_id = getattr(http.state, "trace_id", "unknown")
    caller = policy.identify(http)
    svc = _service()
    log.info(
        "[search-pipeline:%s] embed request caller=%s kind=%s texts=%s",
        trace_id,
        caller.kind,
        request.kind,
        len(request.texts),
    )
    result, cached = await run_in_threadpool(
        lambda: svc.embed(request.texts, kind=request.kind)
    )
    log.info(
        "[search-pipeline:%s] embed complete model=%s dimension=%s vectors=%s cached=%s",
        trace_id,
        result.model,
        result.dim,
        len(result.vectors),
        cached,
    )
    return EmbedResponse(
        model=result.model, dim=result.dim, vectors=result.vectors, cached=cached
    )
