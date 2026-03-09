"""Tests for retry integration in RetrievalBroker."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.common import ChunkId
from libs.contracts.retrieval import RetrievalCandidate, RetrievalQuery
from libs.resilience import RetryConfig
from libs.retrieval.broker.models import BrokerConfig, BrokerOutcome, RetrievalMode
from libs.retrieval.broker.service import RetrievalBroker
from libs.retrieval.stores.memory_lexical_store import MemoryLexicalStore


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


def _make_query(text: str = "test") -> RetrievalQuery:
    return RetrievalQuery(
        raw_query=text,
        normalized_query=text,
        trace_id="trace-retry",
        top_k=10,
    )


class _MockEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 4


class _FailOnceLexicalStore:
    """Lexical store that fails once then succeeds."""

    def __init__(self) -> None:
        self._call_count = 0
        self._inner = MemoryLexicalStore(store_id="fail-once-lexical")

    @property
    def store_id(self) -> str:
        return "fail-once-lexical"

    def add(self, chunk: Chunk) -> None:
        self._inner.add(chunk)

    def add_batch(self, chunks: list[Chunk]) -> None:
        self._inner.add_batch(chunks)

    def delete(self, chunk_ids: list[ChunkId]) -> int:
        return 0

    def search(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        self._call_count += 1
        if self._call_count == 1:
            raise ConnectionError("transient failure")
        return self._inner.search(query)

    def count(self) -> int:
        return self._inner.count()


class _AlwaysFailLexicalStore:
    """Lexical store that always fails."""

    @property
    def store_id(self) -> str:
        return "always-fail-lexical"

    def add(self, chunk: Chunk) -> None:
        pass

    def add_batch(self, chunks: list[Chunk]) -> None:
        pass

    def delete(self, chunk_ids: list[ChunkId]) -> int:
        return 0

    def search(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        raise ConnectionError("always fails")

    def count(self) -> int:
        return 0


class TestBrokerRetry:
    def test_retry_recovers_from_transient_failure(self) -> None:
        """Store fails once, retry succeeds on second attempt."""
        fail_once = _FailOnceLexicalStore()
        fail_once.add(_make_chunk(content="retry test data", chunk_id="chk-1"))

        from libs.retrieval.stores.memory_vector_store import MemoryVectorStore

        broker = RetrievalBroker(
            vector_store=MemoryVectorStore(),
            lexical_store=fail_once,  # type: ignore[arg-type]
            query_embedder=_MockEmbedder(),
            config=BrokerConfig(mode=RetrievalMode.LEXICAL),
            retry_config=RetryConfig(
                max_retries=2, base_delay_s=0.01, jitter_factor=0.0,
            ),
        )

        result = broker.run(_make_query("retry test"))
        assert result.outcome == BrokerOutcome.SUCCESS
        assert result.candidate_count >= 1

    def test_retry_exhausted_still_fails(self) -> None:
        """Store always fails → FAILED after retries exhausted."""
        from libs.retrieval.stores.memory_vector_store import MemoryVectorStore

        broker = RetrievalBroker(
            vector_store=MemoryVectorStore(),
            lexical_store=_AlwaysFailLexicalStore(),  # type: ignore[arg-type]
            query_embedder=_MockEmbedder(),
            config=BrokerConfig(mode=RetrievalMode.LEXICAL),
            retry_config=RetryConfig(
                max_retries=2, base_delay_s=0.01, jitter_factor=0.0,
            ),
        )

        result = broker.run(_make_query("fail"))
        assert result.outcome == BrokerOutcome.FAILED
        assert len(result.errors) >= 1
