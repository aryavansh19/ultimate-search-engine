"""HTTP API for the link-memory search engine.

Endpoint shapes mirror the CLI, with one addition that exists purely for
search-as-you-type: `mode=keyword` skips embedding entirely and answers from FTS in about
a millisecond, so the UI can render on every keystroke and upgrade to hybrid results once
typing settles.

SECURITY -- read before exposing this anywhere:

* There is no authentication. Anyone who can reach the port can read the whole library and
  add links that spend your Gemini quota.
* `POST /api/items` fetches a caller-supplied URL server-side, which is textbook SSRF
  exposure. The HTML tier resolves hostnames and refuses private, loopback, link-local and
  reserved addresses, but that check does not stop DNS rebinding.
* The server therefore binds to 127.0.0.1 by default. Putting it on 0.0.0.0 without adding
  auth and a request allow-list would be a genuine mistake, not a rough edge.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from extractor import looks_like_url
from search import SearchMode

from .pipeline import Pipeline
from .security import TOKEN_HEADER, AccessPolicy

log = logging.getLogger("api.app")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

pipeline: Pipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = Pipeline()
    log.info("pipeline ready")
    try:
        yield
    finally:
        if pipeline is not None:
            pipeline.close()
            pipeline = None


app = FastAPI(title="Link Memory", version="0.1.0", lifespan=lifespan)

policy = AccessPolicy.from_env()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*", TOKEN_HEADER],
)


@app.middleware("http")
async def guard(request: Request, call_next):
    """Enforce the access token and write limits before anything touches the pipeline.

    Middleware rather than per-route dependencies so a route added later cannot forget to
    protect itself -- the failure mode of an unguarded new endpoint on a publicly tunnelled
    server is exactly the one worth designing out.
    """
    try:
        policy.check(request)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, Any]:
    """Unauthenticated liveness probe. Deliberately carries no library data."""
    return {"ok": True, "auth_required": policy.enabled}


def _pipeline() -> Pipeline:
    if pipeline is None:  # pragma: no cover - only during shutdown
        raise HTTPException(status_code=503, detail="pipeline not ready")
    return pipeline


class AddItemRequest(BaseModel):
    url: str = Field(min_length=4)
    # Pre-extracted content, as an iOS share extension or browser extension would send.
    # Present means the client already did the extraction with the user's own session.
    payload: dict[str, Any] | None = None


@app.post("/api/items", status_code=202)
def add_item(request: AddItemRequest) -> dict[str, Any]:
    """Queue a link. Returns immediately; poll /api/jobs for progress."""
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    # Reject non-URLs before queueing rather than after. Unvalidated text reached the
    # worker and became a real item -- a stray search term "ramen" was saved as
    # `https://ramen/` and burned an extraction plus a tagging call before failing.
    if not looks_like_url(url):
        raise HTTPException(
            status_code=400,
            detail="That does not look like a link. Paste a full URL, e.g. "
            "https://www.instagram.com/reels/...",
        )
    return _pipeline().submit(url, request.payload)


@app.get("/api/items")
def list_items(limit: int = Query(200, ge=1, le=1000)) -> dict[str, Any]:
    p = _pipeline()
    return {"items": [_present(doc) for doc in p.library(limit=limit)]}


@app.get("/api/items/{url_hash}")
def item_detail(url_hash: str) -> dict[str, Any]:
    """Everything extracted for one item, for the detail drawer.

    Joins all three stages: the extractor's envelope (raw captured content), the
    enrichment (model-derived understanding), and the search document (what is indexed).
    Kept as one call so opening the drawer is a single round trip.
    """
    p = _pipeline()
    doc = p.search_store.get_documents([url_hash]).get(url_hash)
    if doc is None:
        raise HTTPException(status_code=404, detail="unknown item")

    envelope = p.cache.get(url_hash)
    enrichment = p.enricher.store.get(url_hash)

    payload = _present(doc)
    payload["description"] = doc.get("description")

    if envelope is not None:
        payload["extraction"] = {
            "tier": envelope.tier.value,
            "note": envelope.extractor_note,
            "signal": envelope.signal.value,
            "word_count": envelope.prose_word_count,
            "duration_s": envelope.duration_s,
            "media_kind": envelope.media_kind.value,
            "degraded": envelope.degraded,
            "caption": envelope.caption,
            "platform_tags": envelope.platform_tags,
            "has_transcript": bool(envelope.transcript),
            "transcript_chars": len(envelope.transcript or ""),
            "transcript_excerpt": (envelope.transcript or "")[:1500] or None,
            "article_chars": len(envelope.article_text or ""),
            "like_count": envelope.like_count,
            "view_count": envelope.view_count,
        }

    if enrichment is not None:
        payload["enrichment"] = {
            "summary": enrichment.summary,
            "description": enrichment.description,
            "details": enrichment.details,
            "tags": enrichment.tags,
            "category": enrichment.category,
            "content_type": enrichment.content_type,
            "entities": {
                "people": enrichment.entities.people,
                "organizations": enrichment.entities.organizations,
                "places": enrichment.entities.places,
                "products": enrichment.entities.products,
            },
            "language": enrichment.language,
            "mode": enrichment.mode.value,
            "provider": enrichment.provider,
            "model": enrichment.model,
            "input_tokens": enrichment.input_tokens,
            "output_tokens": enrichment.output_tokens,
            "cost_usd": enrichment.cost_usd,
            "duration_ms": enrichment.duration_ms,
            "degraded": enrichment.degraded,
            "note": enrichment.note,
        }

    return payload


@app.delete("/api/items/{url_hash}")
def delete_item(url_hash: str) -> Response:
    """Remove an item from the search index.

    Returns 204 via an explicit Response: FastAPI refuses a 204 route that declares a
    response model, since a 204 must not carry a body.
    """
    _pipeline().forget(url_hash)
    return Response(status_code=204)


@app.post("/api/items/{url_hash}/deepen", status_code=202)
def deepen_item(url_hash: str) -> dict[str, Any]:
    """Force a full visual pass on an item that only has a transcript or a caption.

    A captioned YouTube video is deliberately not analyzed visually -- the transcript is a
    richer and free record of a talking-head video. That is the wrong trade for anything
    where the picture carries the meaning, so this exists to override it per item rather
    than forcing the expensive path on everything.
    """
    p = _pipeline()
    doc = p.search_store.get_documents([url_hash]).get(url_hash)
    if doc is None:
        raise HTTPException(status_code=404, detail="unknown item")
    return p.submit_deepen(url_hash, str(doc["canonical_url"]))


@app.get("/api/jobs")
def list_jobs(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
    p = _pipeline()
    return {
        "active": [job.as_dict() for job in p.jobs.active()],
        "recent": [job.as_dict() for job in p.jobs.recent(limit=limit)],
    }


@app.post("/api/jobs/clear")
def clear_jobs() -> dict[str, int]:
    return {"cleared": _pipeline().jobs.clear_finished()}


@app.get("/api/search")
def search(
    q: str = Query("", description="query text"),
    mode: str = Query("hybrid"),
    limit: int = Query(20, ge=1, le=100),
    category: str | None = None,
    content_type: str | None = None,
    platform: str | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    """Search the library.

    `mode=keyword` is the fast path used for per-keystroke rendering: no embedding call, so
    it costs nothing and returns in about a millisecond. `mode=hybrid` adds the semantic
    leg and is what the UI requests once typing pauses.
    """
    query = q.strip()
    if not query:
        return {"query": "", "mode": mode, "hits": [], "notes": []}

    try:
        search_mode = SearchMode(mode)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"unknown mode {mode!r}") from None

    p = _pipeline()
    response = p.searcher.search(
        query,
        limit=limit,
        mode=search_mode,
        category=category,
        content_type=content_type,
        platform=platform,
        tag=tag,
    )
    return {
        "query": response.query,
        "mode": response.mode.value,
        "vector_available": response.vector_available,
        "notes": response.notes,
        "keyword_candidates": response.keyword_candidates,
        "vector_candidates": response.vector_candidates,
        "hits": [
            {
                "url_hash": hit.url_hash,
                "url": hit.canonical_url,
                "title": hit.title,
                "summary": hit.summary,
                "author": hit.author,
                "platform": hit.platform,
                "category": hit.category,
                "content_type": hit.content_type,
                "tags": hit.tags,
                "thumbnail_url": hit.thumbnail_url,
                "score": hit.score,
                "found_by": hit.found_by,
                "keyword_rank": hit.keyword_rank,
                "vector_rank": hit.vector_rank,
                "cosine": hit.vector_similarity,
                "matched_text": hit.matched_text,
                # Same coverage fields as /api/items so the card renders identically in
                # both views.
                "analysis": _coverage_label(
                    {
                        "audio_covered": hit.audio_covered,
                        "visual_covered": hit.visual_covered,
                        "is_video": hit.is_video,
                    }
                ),
                "analysis_mode": hit.analysis_mode,
                "detail_count": hit.detail_count,
                "is_video": hit.is_video,
                "can_deepen": hit.is_video and not hit.visual_covered,
            }
            for hit in response.hits
        ],
    }


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    p = _pipeline()
    index_stats = p.search_store.stats()
    return {
        "documents": index_stats["documents"],
        "enriched": index_stats["enriched"],
        "chunks": index_stats["chunks"],
        "vectors": index_stats["vectors"],
        "platforms": index_stats["platforms"],
        "embed_model": p.search_config.embed_model,
        "dimensions": p.search_config.dimensions,
        "vector_available": p.searcher.embedder.availability()[0],
        "query_cache_hits": p.searcher.embedder.query_cache_hits,
    }


def _coverage_label(doc: dict[str, Any]) -> str:
    """Plain-language statement of how deeply this item was examined.

    Exists because the distinction was invisible in the UI. An item tagged from a caption
    and an item where every frame was described looked identical in the list, and the only
    way to tell them apart was querying SQLite by hand.
    """
    audio = bool(doc.get("audio_covered"))
    visual = bool(doc.get("visual_covered"))
    if not doc.get("is_video"):
        return "text"
    if audio and visual:
        return "video + audio"
    if visual:
        return "video"
    if audio:
        return "transcript only"
    return "caption only"


def _present(doc: dict[str, Any]) -> dict[str, Any]:
    tags_text = doc.get("tags_text") or ""
    return {
        "url_hash": doc.get("url_hash"),
        "url": doc.get("canonical_url"),
        "title": doc.get("title"),
        "summary": doc.get("summary"),
        "author": doc.get("author"),
        "platform": doc.get("platform"),
        "category": doc.get("category"),
        "content_type": doc.get("content_type"),
        "tags": [t.strip() for t in tags_text.split(",") if t.strip()],
        "thumbnail_url": doc.get("thumbnail_url"),
        "enriched": bool(doc.get("enriched")),
        "indexed_at": doc.get("indexed_at"),
        "analysis": _coverage_label(doc),
        "analysis_mode": doc.get("analysis_mode"),
        "detail_count": doc.get("detail_count") or 0,
        "is_video": bool(doc.get("is_video")),
        "can_deepen": bool(doc.get("is_video")) and not bool(doc.get("visual_covered")),
    }


# ------------------------------------------------------------------------ static UI
if WEB_DIR.is_dir():
    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
