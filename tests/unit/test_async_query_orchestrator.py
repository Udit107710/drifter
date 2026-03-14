"""Tests for orchestrators/async_query.py — AsyncQueryOrchestrator."""

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
from libs.retrieval.broker.async_service import AsyncRetrievalBroker
from libs.retrieval.broker.models import BrokerConfig, RetrievalMode
from libs.retrieval.stores.async_memory_lexical_store import (
    AsyncMemoryLexicalStore,
)
from libs.retrieval.stores.async_memory_vector_store import (
    AsyncMemoryVectorStore,
)
from libs.retrieval.stores.memory_lexical_store import MemoryLexicalStore
from libs.retrieval.stores.memory_vector_store import MemoryVectorStore
from orchestrators.async_query import AsyncQueryOrchestrator
from orchestrators.query import QueryResult


def _make_chunk(chunk_id: str, content: str) -> Chunk:
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


class _AsyncMockEmbedder:
    async def async_embed_query(self, text: str) -> list[float]:
        return [0.1] * 64


def _make_async_orchestrator() -> tuple[
    AsyncQueryOrchestrator, MemoryVectorStore, MemoryLexicalStore,
]:
    vs = MemoryVectorStore()
    ls = MemoryLexicalStore()

    collector = InMemoryCollector()
    tracer = Tracer(collector=collector)

    broker = AsyncRetrievalBroker(
        vector_store=AsyncMemoryVectorStore(vs),
        lexical_store=AsyncMemoryLexicalStore(ls),
        query_embedder=_AsyncMockEmbedder(),
        config=BrokerConfig(mode=RetrievalMode.HYBRID),
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

    orchestrator = AsyncQueryOrchestrator(
        tracer=tracer,
        async_retrieval_broker=broker,
        reranker_service=reranker_service,
        context_builder_service=context_service,
        generation_service=gen_service,
        token_budget=3000,
    )
    return orchestrator, vs, ls


class TestAsyncQueryOrchestrator:
    async def test_empty_stores_returns_no_results(self) -> None:
        orchestrator, _, _ = _make_async_orchestrator()
        result = await orchestrator.async_run("test query")
        assert isinstance(result, QueryResult)
        assert result.outcome == "no_results"

    async def test_with_data_returns_success(self) -> None:
        orchestrator, _, ls = _make_async_orchestrator()
        chunk = _make_chunk("c1", "machine learning is a branch of AI")
        ls.add(chunk)

        result = await orchestrator.async_run("machine learning")
        assert result.broker_result is not None
        assert result.broker_result.candidate_count > 0

    async def test_trace_id_propagation(self) -> None:
        orchestrator, _, _ = _make_async_orchestrator()
        result = await orchestrator.async_run("test", trace_id="async-trace-1")
        assert result.trace_id == "async-trace-1"
