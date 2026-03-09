"""Tests for AsyncRetrievalBroker: parallel fanout, failure handling."""

from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import UTC, datetime

from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.retrieval import RetrievalCandidate, RetrievalQuery
from libs.retrieval.broker.async_service import AsyncRetrievalBroker
from libs.retrieval.broker.models import BrokerConfig, BrokerOutcome, RetrievalMode
from libs.retrieval.stores.async_memory_lexical_store import AsyncMemoryLexicalStore
from libs.retrieval.stores.async_memory_vector_store import AsyncMemoryVectorStore
from libs.retrieval.stores.memory_lexical_store import MemoryLexicalStore
from libs.retrieval.stores.memory_vector_store import MemoryVectorStore


def _make_chunk(content: str = "hello", chunk_id: str = "chk-1") -> Chunk:
    content_hash = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_id="src-1",
        block_ids=["blk-1"],
        content=content,
        content_hash=content_hash,
        token_count=len(content.split()),
        strategy="fixed_window",
        byte_offset_start=0,
        byte_offset_end=len(content.encode()),
        lineage=ChunkLineage(
            source_id="src-1",
            document_id="doc-1",
            block_ids=["blk-1"],
            chunk_strategy="fixed_window",
            parser_version="test:1.0",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
    )


def _make_embedding(
    chunk_id: str = "chk-1",
    vector: list[float] | None = None,
) -> tuple[str, list[float]]:
    from libs.contracts.embeddings import ChunkEmbedding

    v = vector or [1.0, 0.0, 0.0, 0.0]
    emb = ChunkEmbedding(
        embedding_id=f"emb-{chunk_id}",
        chunk_id=chunk_id,
        vector=v,
        model_id="test-model",
        model_version="1.0",
        dimensions=len(v),
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    return chunk_id, emb  # type: ignore[return-value]


def _make_query(text: str = "test") -> RetrievalQuery:
    return RetrievalQuery(
        raw_query=text,
        normalized_query=text,
        trace_id="trace-async",
        top_k=10,
    )


class _AsyncMockEmbedder:
    async def async_embed_query(self, text: str) -> list[float]:
        return [0.1] * 4


class TestAsyncBrokerHybrid:
    async def test_hybrid_retrieval(self) -> None:
        """Both stores populated → SUCCESS with candidates."""
        vs = MemoryVectorStore(store_id="mem-v")
        ls = MemoryLexicalStore(store_id="mem-l")

        from libs.contracts.embeddings import ChunkEmbedding

        c1 = _make_chunk(content="dense data point", chunk_id="chk-1")
        emb = ChunkEmbedding(
            embedding_id="emb-1",
            chunk_id="chk-1",
            vector=[1.0, 0.0, 0.0, 0.0],
            model_id="test",
            model_version="1.0",
            dimensions=4,
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        vs.add(emb, c1)

        c2 = _make_chunk(content="lexical data point", chunk_id="chk-2")
        ls.add(c2)

        broker = AsyncRetrievalBroker(
            vector_store=AsyncMemoryVectorStore(vs),
            lexical_store=AsyncMemoryLexicalStore(ls),
            query_embedder=_AsyncMockEmbedder(),
            config=BrokerConfig(mode=RetrievalMode.HYBRID),
        )

        result = await broker.run(_make_query("data point"))
        assert result.outcome == BrokerOutcome.SUCCESS
        assert result.candidate_count >= 1
        assert result.debug.get("async") is True


class TestAsyncBrokerSingleStore:
    async def test_lexical_only(self) -> None:
        """LEXICAL mode with async broker."""
        ls = MemoryLexicalStore(store_id="mem-l")
        c = _make_chunk(content="single store test", chunk_id="chk-1")
        ls.add(c)

        broker = AsyncRetrievalBroker(
            vector_store=AsyncMemoryVectorStore(),
            lexical_store=AsyncMemoryLexicalStore(ls),
            query_embedder=_AsyncMockEmbedder(),
            config=BrokerConfig(mode=RetrievalMode.LEXICAL),
        )

        result = await broker.run(_make_query("single store"))
        assert result.outcome == BrokerOutcome.SUCCESS
        assert result.candidate_count >= 1


class TestAsyncBrokerParallel:
    async def test_parallel_execution_is_faster(self) -> None:
        """Hybrid fanout should run in parallel (~max latency, not sum)."""

        class _SlowAsyncVectorStore:
            @property
            def store_id(self) -> str:
                return "slow-v"

            async def async_search(
                self, query: RetrievalQuery, query_vector: list[float],
            ) -> list[RetrievalCandidate]:
                await asyncio.sleep(0.05)
                return []

        class _SlowAsyncLexicalStore:
            @property
            def store_id(self) -> str:
                return "slow-l"

            async def async_search(
                self, query: RetrievalQuery,
            ) -> list[RetrievalCandidate]:
                await asyncio.sleep(0.05)
                return []

        broker = AsyncRetrievalBroker(
            vector_store=_SlowAsyncVectorStore(),
            lexical_store=_SlowAsyncLexicalStore(),
            query_embedder=_AsyncMockEmbedder(),
            config=BrokerConfig(mode=RetrievalMode.HYBRID),
        )

        t0 = time.monotonic()
        await broker.run(_make_query("parallel"))
        elapsed = time.monotonic() - t0

        # If sequential, would be ~0.1s. Parallel should be ~0.05s.
        assert elapsed < 0.09, f"Expected parallel execution, got {elapsed:.3f}s"


class TestAsyncBrokerFailure:
    async def test_both_fail(self) -> None:
        """Both stores fail → FAILED outcome."""

        class _FailingAsyncVectorStore:
            @property
            def store_id(self) -> str:
                return "fail-v"

            async def async_search(
                self, query: RetrievalQuery, query_vector: list[float],
            ) -> list[RetrievalCandidate]:
                raise RuntimeError("vector fail")

        class _FailingAsyncLexicalStore:
            @property
            def store_id(self) -> str:
                return "fail-l"

            async def async_search(
                self, query: RetrievalQuery,
            ) -> list[RetrievalCandidate]:
                raise RuntimeError("lexical fail")

        broker = AsyncRetrievalBroker(
            vector_store=_FailingAsyncVectorStore(),
            lexical_store=_FailingAsyncLexicalStore(),
            query_embedder=_AsyncMockEmbedder(),
            config=BrokerConfig(mode=RetrievalMode.HYBRID),
        )

        result = await broker.run(_make_query("fail"))
        assert result.outcome == BrokerOutcome.FAILED
        assert len(result.errors) == 2
