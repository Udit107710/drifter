"""Tests for orchestrators/query.py — QueryOrchestrator."""

from __future__ import annotations

from datetime import UTC, datetime

from libs.chunking.token_counter import WhitespaceTokenCounter
from libs.context_builder.greedy_builder import GreedyContextBuilder
from libs.context_builder.service import ContextBuilderService
from libs.contracts.chunks import Chunk, ChunkLineage
from libs.generation.citation_validator import DefaultCitationValidator
from libs.generation.mock_generator import MockGenerator
from libs.generation.request_builder import GenerationRequestBuilder
from libs.generation.service import GenerationService
from libs.observability.collector import InMemoryCollector
from libs.observability.tracer import Tracer
from libs.reranking.feature_reranker import FeatureBasedReranker
from libs.reranking.service import RerankerService
from libs.retrieval.broker.service import RetrievalBroker
from libs.retrieval.stores.memory_lexical_store import MemoryLexicalStore
from libs.retrieval.stores.memory_vector_store import MemoryVectorStore
from orchestrators.query import QueryOrchestrator, QueryResult


def _make_chunk(chunk_id: str, content: str) -> Chunk:
    """Create a test chunk."""
    now = datetime.now(UTC)
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_id="src-1",
        block_ids=["b1"],
        content=content,
        content_hash=f"hash-{chunk_id}",
        token_count=len(content.split()),
        strategy="fixed_window",
        byte_offset_start=0,
        byte_offset_end=len(content),
        lineage=ChunkLineage(
            source_id="src-1",
            document_id="doc-1",
            block_ids=["b1"],
            chunk_strategy="fixed_window",
            parser_version="test:1.0",
            created_at=now,
        ),
    )


def _make_orchestrator(
    collector: InMemoryCollector | None = None,
) -> tuple[QueryOrchestrator, MemoryVectorStore, MemoryLexicalStore, InMemoryCollector]:
    """Build a QueryOrchestrator with in-memory stores."""
    coll = collector or InMemoryCollector()
    tracer = Tracer(collector=coll)

    vector_store = MemoryVectorStore()
    lexical_store = MemoryLexicalStore()

    # Mock query embedder
    class _MockEmbedder:
        def embed_query(self, text: str) -> list[float]:
            return [0.1] * 64

    broker = RetrievalBroker(
        vector_store=vector_store,
        lexical_store=lexical_store,
        query_embedder=_MockEmbedder(),
    )

    reranker = FeatureBasedReranker()
    reranker_service = RerankerService(reranker=reranker)

    counter = WhitespaceTokenCounter()
    builder = GreedyContextBuilder(token_counter=counter)
    context_service = ContextBuilderService(builder=builder)

    generator = MockGenerator()
    request_builder = GenerationRequestBuilder()
    citation_validator = DefaultCitationValidator()
    gen_service = GenerationService(
        generator=generator,
        request_builder=request_builder,
        citation_validator=citation_validator,
    )

    orchestrator = QueryOrchestrator(
        tracer=tracer,
        retrieval_broker=broker,
        reranker_service=reranker_service,
        context_builder_service=context_service,
        generation_service=gen_service,
        token_budget=3000,
    )

    return orchestrator, vector_store, lexical_store, coll


class TestQueryOrchestratorFullPipeline:
    """Test the full query pipeline with in-memory stores."""

    def test_empty_store_returns_no_results(self) -> None:
        orchestrator, _, _, _ = _make_orchestrator()
        result = orchestrator.run("test query")
        assert isinstance(result, QueryResult)
        assert result.trace_id
        assert result.query == "test query"
        assert result.outcome == "no_results"
        assert result.total_latency_ms > 0

    def test_with_lexical_data_returns_success(self) -> None:
        orchestrator, _, lexical_store, _ = _make_orchestrator()

        # Add some chunks to the lexical store
        chunk = _make_chunk("c1", "machine learning is a branch of AI")
        lexical_store.add(chunk)

        result = orchestrator.run("machine learning")
        assert result.trace_id
        assert result.broker_result is not None
        assert result.broker_result.candidate_count > 0

    def test_trace_id_propagation(self) -> None:
        orchestrator, _, _, _ = _make_orchestrator()
        result = orchestrator.run("test", trace_id="custom-trace-123")
        assert result.trace_id == "custom-trace-123"

    def test_generates_trace_id_when_not_provided(self) -> None:
        orchestrator, _, _, _ = _make_orchestrator()
        result = orchestrator.run("test")
        assert result.trace_id
        assert len(result.trace_id) > 0


class TestQueryOrchestratorStages:
    """Test individual pipeline stage methods."""

    def test_retrieve_only(self) -> None:
        orchestrator, _, _, _ = _make_orchestrator()
        result = orchestrator.run_retrieve_only("test query")
        assert result.broker_result is not None
        assert result.reranker_result is None
        assert result.builder_result is None
        assert result.generation_result is None

    def test_through_rerank(self) -> None:
        orchestrator, _, lexical_store, _ = _make_orchestrator()
        chunk = _make_chunk("c1", "test query content here")
        lexical_store.add(chunk)

        result = orchestrator.run_through_rerank("test query")
        assert result.broker_result is not None
        assert result.builder_result is None
        assert result.generation_result is None

    def test_through_context(self) -> None:
        orchestrator, _, lexical_store, _ = _make_orchestrator()
        chunk = _make_chunk("c1", "test query content here")
        lexical_store.add(chunk)

        result = orchestrator.run_through_context("test query")
        assert result.broker_result is not None
        assert result.builder_result is not None
        assert result.generation_result is None


class TestQueryOrchestratorObservability:
    """Test span collection during pipeline execution."""

    def test_spans_are_collected(self) -> None:
        collector = InMemoryCollector()
        orchestrator, _, _, _ = _make_orchestrator(collector)
        orchestrator.run("test query")

        assert collector.count > 0
        span_names = [s.name for s in collector.spans]
        assert "retrieval" in span_names

    def test_trace_id_in_all_spans(self) -> None:
        collector = InMemoryCollector()
        orchestrator, _, _, _ = _make_orchestrator(collector)
        orchestrator.run("test", trace_id="trace-abc")

        for span in collector.spans:
            assert span.trace_id == "trace-abc"
