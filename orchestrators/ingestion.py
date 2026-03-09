"""Ingestion orchestrator: composes ingest → parse → chunk → index."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from libs.chunking.protocols import ChunkingStrategy
from libs.contracts.chunks import Chunk
from libs.contracts.common import RunId
from libs.contracts.documents import CanonicalDocument
from libs.indexing.models import IndexingResult
from libs.indexing.service import IndexingService
from libs.ingestion.models import IngestionOutcome, IngestionResult
from libs.ingestion.service import IngestionService
from libs.observability.context import ObservabilityContext
from libs.observability.stage_instruments import pipeline_span, record_stage_result
from libs.observability.tracer import Tracer
from libs.parsing.protocols import DocumentParser


@dataclass(frozen=True)
class IngestionPipelineResult:
    """Summary of a complete ingest → parse → chunk → index run."""

    run_id: str
    documents_ingested: int
    chunks_produced: int
    chunks_indexed: int
    errors: list[str] = field(default_factory=list)
    total_latency_ms: float = 0.0
    trace_id: str = ""
    indexing_result: IndexingResult | None = None


class IngestionOrchestrator:
    """Orchestrates the full ingestion pipeline."""

    def __init__(
        self,
        tracer: Tracer,
        ingestion_service: IngestionService,
        parser_registry: dict[str, DocumentParser],
        chunking_strategy: ChunkingStrategy,
        indexing_service: IndexingService,
    ) -> None:
        self._tracer = tracer
        self._ingestion = ingestion_service
        self._parsers = parser_registry
        self._chunker = chunking_strategy
        self._indexer = indexing_service

    def run(
        self,
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> IngestionPipelineResult:
        """Execute ingest → parse → chunk → index pipeline."""
        start = time.monotonic()
        ctx = self._tracer.create_context(trace_id=trace_id)
        rid: RunId = run_id or f"ingest-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        errors: list[str] = []

        # --- Stage 1: Ingestion ---
        ingestion_results = self._run_ingestion(ctx, rid)

        # --- Stage 2: Parse ---
        canonical_docs: list[CanonicalDocument] = []
        for result in ingestion_results:
            if result.outcome != IngestionOutcome.SUCCESS:
                if result.error:
                    errors.append(f"Ingestion {result.work_item.source_id}: {result.error}")
                continue
            if result.raw_document is None:
                continue
            mime = result.raw_document.mime_type
            parser = self._parsers.get(mime)
            if parser is None:
                errors.append(f"No parser for MIME type {mime}")
                continue
            try:
                doc = parser.parse(result.raw_document)
                canonical_docs.append(doc)
            except Exception as exc:
                errors.append(f"Parse error for {result.work_item.source_id}: {exc}")

        # --- Stage 3: Chunk ---
        all_chunks: list[Chunk] = []
        for doc in canonical_docs:
            with pipeline_span(self._tracer, ctx, "chunking") as span:
                chunks = self._chunker.chunk(doc)
                record_stage_result(
                    span, outcome="success",
                    input_count=len(doc.blocks), output_count=len(chunks),
                )
                all_chunks.extend(chunks)

        # --- Stage 4: Index ---
        indexing_result: IndexingResult | None = None
        chunks_indexed = 0
        if all_chunks:
            indexing_result = self._run_indexing(ctx, all_chunks, rid)
            chunks_indexed = (
                indexing_result.chunks_indexed_vector
                + indexing_result.chunks_indexed_lexical
            )
            errors.extend(indexing_result.errors)

        return IngestionPipelineResult(
            run_id=rid,
            documents_ingested=len(canonical_docs),
            chunks_produced=len(all_chunks),
            chunks_indexed=chunks_indexed,
            errors=errors,
            total_latency_ms=_elapsed_ms(start),
            trace_id=ctx.trace_id,
            indexing_result=indexing_result,
        )

    def _run_ingestion(
        self, ctx: ObservabilityContext, run_id: RunId,
    ) -> list[IngestionResult]:
        with pipeline_span(self._tracer, ctx, "ingestion") as span:
            results = self._ingestion.run(run_id)
            successes = sum(1 for r in results if r.outcome == IngestionOutcome.SUCCESS)
            record_stage_result(
                span, outcome="success",
                input_count=len(results), output_count=successes,
            )
            return results

    def _run_indexing(
        self, ctx: ObservabilityContext,
        chunks: list[Chunk], run_id: RunId,
    ) -> IndexingResult:
        with pipeline_span(self._tracer, ctx, "indexing") as span:
            result = self._indexer.run(chunks, run_id)
            record_stage_result(
                span, outcome=result.outcome.value,
                input_count=len(chunks),
                output_count=result.chunks_indexed_vector,
            )
            return result


def _elapsed_ms(start: float) -> float:
    return (time.monotonic() - start) * 1000
