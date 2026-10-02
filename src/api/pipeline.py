"""Shared pipeline wiring for the API process.

One important detail: the `Indexer` and the `Searcher` are given the *same* `SearchStore`
instance. The store keeps every vector in one in-memory numpy matrix for fast scanning and
invalidates that cache on write. Two separate store objects would each hold their own
cache, so indexing a new link would never invalidate the searcher's copy and a
freshly-added item would be invisible to semantic search until the server restarted -- with
no error anywhere.

Ingestion runs on a small thread pool. The extraction and enrichment code is synchronous
and does blocking network I/O, so running it inline would stall the event loop and freeze
search for everyone while one video uploads.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from enrichment import Enricher, EnrichmentConfig
from enrichment.schema import EnrichmentMode
from extractor import EnvelopeCache, ExtractionCascade, ExtractorConfig
from search import Indexer, SearchConfig, Searcher, SearchStore

from .jobs import JobStage, JobStatus, JobStore

log = logging.getLogger("api.pipeline")


class Pipeline:
    def __init__(self, max_workers: int = 2) -> None:
        self.extractor_config = ExtractorConfig.from_env()
        self.enrichment_config = EnrichmentConfig.from_env()
        self.search_config = SearchConfig.from_env()

        self.cascade = ExtractionCascade(self.extractor_config)
        self.enricher = Enricher(self.enrichment_config)

        # Single shared store; see the module docstring for why this is not optional.
        self.search_store = SearchStore(self.search_config.store_path)
        self.indexer = Indexer(self.search_config, store=self.search_store)
        self.searcher = Searcher(self.search_config, store=self.search_store)

        self.jobs = JobStore(self.search_config.store_path)
        self.cache = EnvelopeCache(self.search_config.store_path)

        # Ingestion is serialized enough to avoid hammering the APIs, but not so much
        # that one slow video blocks every other link indefinitely.
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ingest")
        self._ingest_lock = threading.Lock()

    # ------------------------------------------------------------------- ingestion
    def submit(self, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        job = self.jobs.create(url)
        self._pool.submit(self._run_job, job.id, url, payload)
        return job.as_dict()

    def submit_deepen(self, url_hash: str, url: str) -> dict[str, Any]:
        """Re-run one item forcing media mode, regardless of how strong its text looks."""
        job = self.jobs.create(url)
        self._pool.submit(self._run_job, job.id, url, None, True)
        return job.as_dict()

    def _run_job(
        self,
        job_id: int,
        url: str,
        payload: dict[str, Any] | None,
        force_media: bool = False,
    ) -> None:
        try:
            self.jobs.update(job_id, status=JobStatus.RUNNING, stage=JobStage.EXTRACTING)
            result = self.cascade.extract(url, client_payload=payload)
            envelope = result.envelope
            self.jobs.update(job_id, url_hash=envelope.url_hash, stage=JobStage.TAGGING)

            enrichment = self.enricher.enrich(
                envelope,
                force=force_media,
                mode=EnrichmentMode.MEDIA if force_media else None,
            )
            self.jobs.update(job_id, stage=JobStage.INDEXING)

            # Serialize the index write. SQLite handles concurrent writers poorly enough
            # that two simultaneous ingests can hit "database is locked", and the work
            # here is short compared with the network calls above.
            with self._ingest_lock:
                self.indexer.index_one(envelope, enrichment, force=True)

            note = None
            if envelope.degraded:
                note = "extraction degraded; nothing useful was retrieved"
            elif enrichment.degraded:
                note = "tagged without a language model"
            self.jobs.update(
                job_id, status=JobStatus.READY, stage=JobStage.DONE, error=note
            )
        except Exception as exc:  # noqa: BLE001 - a failed link must not kill the worker
            log.exception("ingest failed for %s", url)
            self.jobs.update(
                job_id,
                status=JobStatus.FAILED,
                stage=JobStage.DONE,
                error=f"{type(exc).__name__}: {exc}",
            )

    # ---------------------------------------------------------------------- queries
    def library(self, limit: int = 200) -> list[dict[str, Any]]:
        """Everything indexed, newest first, joined with its enrichment."""
        return self.search_store.list_documents(limit=limit)

    def forget(self, url_hash: str) -> dict[str, bool]:
        """Remove an item from every stage.

        Deleting only the search document is not enough, and is actively misleading: the
        envelope and enrichment survive, so the next `search index` run rebuilds the
        document and the item reappears as though the delete never happened.
        """
        removed = {
            "enrichment": self.enricher.store.delete(url_hash),
            "envelope": self.cache.delete(url_hash),
        }
        self.search_store.delete(url_hash)
        removed["search"] = True
        return removed

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self.jobs.close()
        self.cache.close()
        self.enricher.close()
        self.cascade.close()
        # indexer and searcher share search_store; close it once.
        self.search_store.close()
