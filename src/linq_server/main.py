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
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .auth import AccessPolicy
from .models import AnalyzeRequest, AnalyzeResponse, EmbedRequest, EmbedResponse
from .service import AnalysisService, ServiceError

log = logging.getLogger("linq.main")

service: AnalysisService | None = None
policy = AccessPolicy.from_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = AnalysisService()
    caps = service.capabilities()
    log.info("linq service ready: %s", caps)
    if not policy.configured and policy.require_auth:
        log.error(
            "No credentials configured. Set SUPABASE_JWT_SECRET or LINQ_API_TOKEN, "
            "or LINQ_REQUIRE_AUTH=0 for loopback development."
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
    payload: dict[str, Any] = {"ok": True, "auth": policy.describe()}
    if service is not None:
        payload["capabilities"] = service.capabilities()
    return payload


@app.post("/v1/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest, http: Request) -> AnalyzeResponse:
    """Extract a link, understand it with Gemini, and return vectors for it.

    This is the expensive one — a video pass costs real money and can take a minute — so it
    is both authenticated and rate limited per caller.
    """
    caller = policy.identify(http)
    policy.charge(caller)
    svc = _service()
    log.info("analyze url=%s force_media=%s caller=%s", request.url, request.force_media, caller.kind)
    return await run_in_threadpool(svc.analyze, request)


@app.post("/v1/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest, http: Request) -> EmbedResponse:
    """Vectorise text. Used for search queries, and for re-embedding after an edit.

    Not charged against the analysis budget: this is cheap, free on the current model, and
    rate limiting the search path would make the app feel broken.
    """
    policy.identify(http)
    svc = _service()
    result, cached = await run_in_threadpool(
        lambda: svc.embed(request.texts, kind=request.kind)
    )
    return EmbedResponse(
        model=result.model, dim=result.dim, vectors=result.vectors, cached=cached
    )
