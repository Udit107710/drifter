"""Tests for orchestrators/ingestion.py — IngestionOrchestrator."""

from __future__ import annotations

from datetime import UTC, datetime

from libs.chunking.strategies.fixed_window import FixedWindowChunker
from libs.contracts.chunks import Chunk
from libs.contracts.common import BlockType
from libs.contracts.documents import Block, CanonicalDocument, RawDocument, SourceDocumentRef
from libs.ingestion.models import (
    IngestionOutcome,
    IngestionResult,
    WorkAction,
    WorkItem,
)
from libs.observability.collector import InMemoryCollector
from libs.observability.tracer import Tracer
from orchestrators.ingestion import IngestionOrchestrator, IngestionPipelineResult


class _MockIngestionService:
    """Minimal mock that returns pre-configured ingestion results."""

    def __init__(self, results: list[IngestionResult]) -> None:
        self._results = results

    def run(self, run_id: str) -> list[IngestionResult]:
        return self._results


class _MockParser:
    """Converts raw bytes to a simple CanonicalDocument."""

    def parse(self, raw: RawDocument) -> CanonicalDocument:
        text = raw.raw_bytes.decode("utf-8", errors="replace")
        block = Block(
            block_id="b1",
            block_type=BlockType.PARAGRAPH,
            content=text,
            position=0,
        )
        return CanonicalDocument(
            document_id=f"doc-{raw.source_ref.source_id}",
            source_ref=raw.source_ref,
            blocks=[block],
            parser_version="mock:1.0",
            parsed_at=datetime.now(UTC),
        )

    def supported_mime_types(self) -> list[str]:
        return ["text/plain"]


class _MockIndexingService:
    """Tracks what was indexed."""

    def __init__(self) -> None:
        self.indexed_chunks: list[Chunk] = []

    def run(self, chunks: list[Chunk], run_id: str):  # type: ignore[no-untyped-def]
        from libs.embeddings.models import EmbeddingModelInfo
        from libs.indexing.models import IndexingOutcome, IndexingResult

        self.indexed_chunks = chunks
        return IndexingResult(
            run_id=run_id,
            chunks_received=len(chunks),
            chunks_embedded=len(chunks),
            chunks_indexed_vector=len(chunks),
            chunks_indexed_lexical=len(chunks),
            outcome=IndexingOutcome.SUCCESS,
            model_info=EmbeddingModelInfo(
                model_id="test", model_version="1.0",
                dimensions=64, max_tokens=512,
            ),
            completed_at=datetime.now(UTC),
        )


def _make_source_ref(source_id: str = "src-1") -> SourceDocumentRef:
    return SourceDocumentRef(
        source_id=source_id,
        uri=f"file:///{source_id}.txt",
        content_hash="abc123",
        fetched_at=datetime.now(UTC),
        version=1,
    )


def _make_raw_doc(content: str, source_id: str = "src-1") -> RawDocument:
    ref = _make_source_ref(source_id)
    raw = content.encode("utf-8")
    return RawDocument(
        source_ref=ref,
        raw_bytes=raw,
        mime_type="text/plain",
        size_bytes=len(raw),
    )


def _make_work_item(source_id: str = "src-1") -> WorkItem:
    return WorkItem(
        source_id=source_id,
        action=WorkAction.INGEST,
        run_id="run-1",
    )


class TestIngestionOrchestrator:
    """Test the ingestion pipeline with mocks."""

    def test_successful_pipeline(self) -> None:
        raw_doc = _make_raw_doc("This is test content for chunking")
        ingestion_results = [
            IngestionResult(
                work_item=_make_work_item(),
                outcome=IngestionOutcome.SUCCESS,
                raw_document=raw_doc,
                source_ref=raw_doc.source_ref,
            )
        ]

        collector = InMemoryCollector()
        tracer = Tracer(collector=collector)
        indexer = _MockIndexingService()

        orchestrator = IngestionOrchestrator(
            tracer=tracer,
            ingestion_service=_MockIngestionService(ingestion_results),  # type: ignore[arg-type]
            parser_registry={"text/plain": _MockParser()},  # type: ignore[dict-item]
            chunking_strategy=FixedWindowChunker(),  # type: ignore[arg-type]
            indexing_service=indexer,  # type: ignore[arg-type]
        )

        result = orchestrator.run(run_id="test-run")
        assert isinstance(result, IngestionPipelineResult)
        assert result.run_id == "test-run"
        assert result.documents_ingested == 1
        assert result.chunks_produced > 0
        assert result.trace_id
        assert result.total_latency_ms > 0
        assert not result.errors

    def test_skipped_ingestion(self) -> None:
        ingestion_results = [
            IngestionResult(
                work_item=_make_work_item(),
                outcome=IngestionOutcome.SKIPPED,
            )
        ]

        orchestrator = IngestionOrchestrator(
            tracer=Tracer(),
            ingestion_service=_MockIngestionService(ingestion_results),  # type: ignore[arg-type]
            parser_registry={},
            chunking_strategy=FixedWindowChunker(),  # type: ignore[arg-type]
            indexing_service=_MockIndexingService(),  # type: ignore[arg-type]
        )

        result = orchestrator.run()
        assert result.documents_ingested == 0
        assert result.chunks_produced == 0

    def test_missing_parser(self) -> None:
        raw_doc = _make_raw_doc("test")
        # Override mime type to something unregistered
        raw_doc_pdf = RawDocument(
            source_ref=raw_doc.source_ref,
            raw_bytes=raw_doc.raw_bytes,
            mime_type="application/pdf",
            size_bytes=raw_doc.size_bytes,
        )
        ingestion_results = [
            IngestionResult(
                work_item=_make_work_item(),
                outcome=IngestionOutcome.SUCCESS,
                raw_document=raw_doc_pdf,
                source_ref=raw_doc_pdf.source_ref,
            )
        ]

        orchestrator = IngestionOrchestrator(
            tracer=Tracer(),
            ingestion_service=_MockIngestionService(ingestion_results),  # type: ignore[arg-type]
            parser_registry={"text/plain": _MockParser()},  # type: ignore[dict-item]
            chunking_strategy=FixedWindowChunker(),  # type: ignore[arg-type]
            indexing_service=_MockIndexingService(),  # type: ignore[arg-type]
        )

        result = orchestrator.run()
        assert result.documents_ingested == 0
        assert len(result.errors) == 1
        assert "No parser" in result.errors[0]

    def test_trace_id_propagation(self) -> None:
        collector = InMemoryCollector()
        tracer = Tracer(collector=collector)

        orchestrator = IngestionOrchestrator(
            tracer=tracer,
            ingestion_service=_MockIngestionService([]),  # type: ignore[arg-type]
            parser_registry={},
            chunking_strategy=FixedWindowChunker(),  # type: ignore[arg-type]
            indexing_service=_MockIndexingService(),  # type: ignore[arg-type]
        )

        result = orchestrator.run(trace_id="custom-trace")
        assert result.trace_id == "custom-trace"
