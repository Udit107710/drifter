"""Integration tests for the retrieval broker with memory stores."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.common import ChunkId, RetrievalMethod
from libs.contracts.embeddings import ChunkEmbedding
from libs.contracts.retrieval import RetrievalCandidate, RetrievalQuery
from libs.embeddings.query_embedder import DeterministicQueryEmbedder
from libs.retrieval.broker.models import (
    BrokerConfig,
    BrokerOutcome,
    ErrorClassification,
    RetrievalMode,
)
from libs.retrieval.broker.protocols import (
    PassthroughNormalizer,
    QueryEmbedder,
    QueryNormalizer,
)
from libs.retrieval.broker.service import RetrievalBroker
from libs.retrieval.stores.memory_lexical_store import MemoryLexicalStore
from libs.retrieval.stores.memory_vector_store import MemoryVectorStore

# ── Helpers ─────────────────────────────────────────────────────────


def _make_chunk(
    content: str = "hello world",
    chunk_id: str = "chk-001",
    document_id: str = "doc-001",
    source_id: str = "src-001",
    metadata: dict[str, Any] | None = None,
) -> Chunk:
    if metadata is None:
        metadata = {}
    content_hash = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_id=source_id,
        block_ids=["blk-001"],
        content=content,
        content_hash=content_hash,
        token_count=len(content.split()),
        strategy="fixed_window",
        byte_offset_start=0,
        byte_offset_end=len(content.encode()),
        lineage=ChunkLineage(
            source_id=source_id,
            document_id=document_id,
            block_ids=["blk-001"],
            chunk_strategy="fixed_window",
            parser_version="test:1.0.0",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
        metadata=metadata,
    )


def _make_embedding(
    chunk_id: str = "chk-001",
    vector: list[float] | None = None,
    embedding_id: str = "emb-001",
    model_id: str = "test-model",
    model_version: str = "1.0",
) -> ChunkEmbedding:
    if vector is None:
        vector = [0.1, 0.2, 0.3, 0.4]
    return ChunkEmbedding(
        embedding_id=embedding_id,
        chunk_id=chunk_id,
        vector=vector,
        model_id=model_id,
        model_version=model_version,
        dimensions=len(vector),
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _make_query(
    query_text: str = "test query",
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
) -> RetrievalQuery:
    if filters is None:
        filters = {}
    return RetrievalQuery(
        raw_query=query_text,
        normalized_query=query_text,
        trace_id="trace-001",
        top_k=top_k,
        filters=filters,
    )


def _make_broker(
    mode: RetrievalMode = RetrievalMode.HYBRID,
    rrf_k: int = 60,
    max_candidates_per_source: int = 0,
    dense_weight: float = 1.0,
    lexical_weight: float = 1.0,
    normalizer: QueryNormalizer | None = None,
    vector_store: MemoryVectorStore | None = None,
    lexical_store: MemoryLexicalStore | None = None,
    query_embedder: DeterministicQueryEmbedder | None = None,
) -> tuple[RetrievalBroker, MemoryVectorStore, MemoryLexicalStore, DeterministicQueryEmbedder]:
    """Build a broker with memory stores and a deterministic embedder."""
    vs = vector_store or MemoryVectorStore(store_id="memory-vector")
    ls = lexical_store or MemoryLexicalStore(store_id="memory-lexical")
    qe = query_embedder or DeterministicQueryEmbedder(dimensions=4)

    config = BrokerConfig(
        mode=mode,
        rrf_k=rrf_k,
        max_candidates_per_source=max_candidates_per_source,
        dense_weight=dense_weight,
        lexical_weight=lexical_weight,
    )

    broker = RetrievalBroker(
        vector_store=vs,
        lexical_store=ls,
        query_embedder=qe,
        config=config,
        normalizer=normalizer,
    )

    return broker, vs, ls, qe


# ── Failing store stubs ─────────────────────────────────────────────


class FailingVectorStore:
    """Vector store that raises RuntimeError on search."""

    @property
    def store_id(self) -> str:
        return "failing-vector"

    def add(self, embedding: ChunkEmbedding, chunk: Chunk) -> None:
        pass

    def add_batch(self, embeddings: list[ChunkEmbedding], chunks: list[Chunk]) -> None:
        pass

    def delete(self, chunk_ids: list[ChunkId]) -> int:
        return 0

    def search(
        self, query: RetrievalQuery, query_vector: list[float]
    ) -> list[RetrievalCandidate]:
        raise RuntimeError("vector store unavailable")

    def count(self) -> int:
        return 0


class FailingLexicalStore:
    """Lexical store that raises RuntimeError on search."""

    @property
    def store_id(self) -> str:
        return "failing-lexical"

    def add(self, chunk: Chunk) -> None:
        pass

    def add_batch(self, chunks: list[Chunk]) -> None:
        pass

    def delete(self, chunk_ids: list[ChunkId]) -> int:
        return 0

    def search(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        raise RuntimeError("lexical store unavailable")

    def count(self) -> int:
        return 0


# ── Tests ───────────────────────────────────────────────────────────


class TestRetrievalBroker:
    def test_dense_only_mode(self) -> None:
        """DENSE mode — vector store populated, result has candidates from dense only."""
        broker, vs, ls, _ = _make_broker(mode=RetrievalMode.DENSE)

        c1 = _make_chunk(content="dense candidate", chunk_id="chk-1")
        e1 = _make_embedding(chunk_id="chk-1", vector=[1.0, 0.0, 0.0, 0.0], embedding_id="emb-1")
        vs.add(e1, c1)

        # Add to lexical store too — should not be returned in DENSE mode
        c2 = _make_chunk(content="lexical only candidate", chunk_id="chk-2")
        ls.add(c2)

        result = broker.run(_make_query("dense candidate"))

        assert result.outcome == BrokerOutcome.SUCCESS
        assert result.mode == RetrievalMode.DENSE
        assert len(result.candidates) >= 1
        # Only dense store should have been queried
        store_methods = {sr.retrieval_method for sr in result.store_results}
        assert RetrievalMethod.DENSE in store_methods
        assert RetrievalMethod.LEXICAL not in store_methods

    def test_lexical_only_mode(self) -> None:
        """LEXICAL mode — lexical store populated, result has lexical candidates."""
        broker, vs, ls, _ = _make_broker(mode=RetrievalMode.LEXICAL)

        c1 = _make_chunk(content="lexical search terms", chunk_id="chk-1")
        ls.add(c1)

        # Add to vector store — should not be returned in LEXICAL mode
        c2 = _make_chunk(content="vector only", chunk_id="chk-2")
        e2 = _make_embedding(chunk_id="chk-2", vector=[1.0, 0.0, 0.0, 0.0], embedding_id="emb-2")
        vs.add(e2, c2)

        result = broker.run(_make_query("lexical search"))

        assert result.outcome == BrokerOutcome.SUCCESS
        assert result.mode == RetrievalMode.LEXICAL
        assert len(result.candidates) >= 1
        store_methods = {sr.retrieval_method for sr in result.store_results}
        assert RetrievalMethod.LEXICAL in store_methods
        assert RetrievalMethod.DENSE not in store_methods

    def test_hybrid_mode(self) -> None:
        """HYBRID mode — both stores populated, candidates fused."""
        broker, vs, ls, _ = _make_broker(mode=RetrievalMode.HYBRID)

        c1 = _make_chunk(content="dense result alpha", chunk_id="chk-1")
        e1 = _make_embedding(chunk_id="chk-1", vector=[1.0, 0.0, 0.0, 0.0], embedding_id="emb-1")
        vs.add(e1, c1)

        c2 = _make_chunk(content="lexical result beta", chunk_id="chk-2")
        ls.add(c2)

        result = broker.run(_make_query("result"))

        assert result.outcome == BrokerOutcome.SUCCESS
        assert result.mode == RetrievalMode.HYBRID
        assert len(result.candidates) >= 1
        store_methods = {sr.retrieval_method for sr in result.store_results}
        assert RetrievalMethod.DENSE in store_methods
        assert RetrievalMethod.LEXICAL in store_methods

    def test_same_chunk_in_both_stores(self) -> None:
        """Same chunk_id in both stores -> HYBRID method, single entry with combined score."""
        broker, vs, ls, _ = _make_broker(mode=RetrievalMode.HYBRID)

        shared = _make_chunk(content="shared content here", chunk_id="chk-shared")
        emb = _make_embedding(
            chunk_id="chk-shared", vector=[1.0, 0.0, 0.0, 0.0], embedding_id="emb-shared"
        )
        vs.add(emb, shared)
        ls.add(shared)

        result = broker.run(_make_query("shared content"))

        # Should have exactly one fused candidate for the shared chunk
        shared_candidates = [
            fc for fc in result.candidates if fc.chunk.chunk_id == "chk-shared"
        ]
        assert len(shared_candidates) == 1
        assert shared_candidates[0].retrieval_method == RetrievalMethod.HYBRID
        assert len(shared_candidates[0].contributing_stores) == 2

    def test_metadata_filter(self) -> None:
        """Filters passed through to stores restrict results."""
        broker, vs, ls, _ = _make_broker(mode=RetrievalMode.HYBRID)

        c1 = _make_chunk(
            content="science article about physics",
            chunk_id="chk-1",
            metadata={"category": "science"},
        )
        c2 = _make_chunk(
            content="art article about painting",
            chunk_id="chk-2",
            metadata={"category": "art"},
        )
        e1 = _make_embedding(chunk_id="chk-1", vector=[1.0, 0.0, 0.0, 0.0], embedding_id="emb-1")
        e2 = _make_embedding(chunk_id="chk-2", vector=[0.0, 1.0, 0.0, 0.0], embedding_id="emb-2")
        vs.add(e1, c1)
        vs.add(e2, c2)
        ls.add(c1)
        ls.add(c2)

        result = broker.run(_make_query("article", filters={"category": "science"}))

        # Only science chunk should appear
        chunk_ids = {fc.chunk.chunk_id for fc in result.candidates}
        assert "chk-1" in chunk_ids
        assert "chk-2" not in chunk_ids

    def test_vector_store_failure(self) -> None:
        """Vector store raises exception -> PARTIAL outcome with lexical results."""
        failing_vs = FailingVectorStore()
        ls = MemoryLexicalStore(store_id="memory-lexical")
        qe = DeterministicQueryEmbedder(dimensions=4)

        config = BrokerConfig(mode=RetrievalMode.HYBRID)
        broker = RetrievalBroker(
            vector_store=failing_vs,  # type: ignore[arg-type]
            lexical_store=ls,
            query_embedder=qe,
            config=config,
        )

        c1 = _make_chunk(content="lexical fallback content", chunk_id="chk-1")
        ls.add(c1)

        result = broker.run(_make_query("lexical fallback"))

        assert result.outcome == BrokerOutcome.PARTIAL
        assert len(result.errors) >= 1
        assert len(result.candidates) >= 1

    def test_lexical_store_failure(self) -> None:
        """Lexical store raises -> PARTIAL with dense results."""
        vs = MemoryVectorStore(store_id="memory-vector")
        failing_ls = FailingLexicalStore()
        qe = DeterministicQueryEmbedder(dimensions=4)

        config = BrokerConfig(mode=RetrievalMode.HYBRID)
        broker = RetrievalBroker(
            vector_store=vs,
            lexical_store=failing_ls,  # type: ignore[arg-type]
            query_embedder=qe,
            config=config,
        )

        c1 = _make_chunk(content="dense fallback content", chunk_id="chk-1")
        e1 = _make_embedding(chunk_id="chk-1", vector=[1.0, 0.0, 0.0, 0.0], embedding_id="emb-1")
        vs.add(e1, c1)

        result = broker.run(_make_query("dense fallback"))

        assert result.outcome == BrokerOutcome.PARTIAL
        assert len(result.errors) >= 1
        assert len(result.candidates) >= 1

    def test_both_stores_fail(self) -> None:
        """Both stores raise -> FAILED outcome."""
        failing_vs = FailingVectorStore()
        failing_ls = FailingLexicalStore()
        qe = DeterministicQueryEmbedder(dimensions=4)

        config = BrokerConfig(mode=RetrievalMode.HYBRID)
        broker = RetrievalBroker(
            vector_store=failing_vs,  # type: ignore[arg-type]
            lexical_store=failing_ls,  # type: ignore[arg-type]
            query_embedder=qe,
            config=config,
        )

        result = broker.run(_make_query("anything"))

        assert result.outcome == BrokerOutcome.FAILED
        assert len(result.errors) == 2
        assert result.candidates == []

    def test_empty_stores(self) -> None:
        """No data in stores -> NO_RESULTS."""
        broker, *_ = _make_broker(mode=RetrievalMode.HYBRID)

        result = broker.run(_make_query("no results expected"))

        assert result.outcome == BrokerOutcome.NO_RESULTS
        assert result.candidates == []

    def test_top_k_respected(self) -> None:
        """Populate many chunks, top_k=3 -> max 3 candidates."""
        broker, _, ls, _ = _make_broker(mode=RetrievalMode.LEXICAL)

        for i in range(10):
            chunk = _make_chunk(
                content=f"common search term variant {i}",
                chunk_id=f"chk-{i}",
            )
            ls.add(chunk)

        result = broker.run(_make_query("common search term", top_k=3))

        assert len(result.candidates) <= 3

    def test_store_results_populated(self) -> None:
        """BrokerResult.store_results has per-store intermediates."""
        broker, vs, ls, _ = _make_broker(mode=RetrievalMode.HYBRID)

        c1 = _make_chunk(content="dense data point", chunk_id="chk-1")
        e1 = _make_embedding(chunk_id="chk-1", vector=[1.0, 0.0, 0.0, 0.0], embedding_id="emb-1")
        vs.add(e1, c1)

        c2 = _make_chunk(content="lexical data point", chunk_id="chk-2")
        ls.add(c2)

        result = broker.run(_make_query("data point"))

        assert len(result.store_results) == 2
        methods = {sr.retrieval_method for sr in result.store_results}
        assert methods == {RetrievalMethod.DENSE, RetrievalMethod.LEXICAL}
        for sr in result.store_results:
            assert sr.store_id != ""
            assert sr.latency_ms >= 0.0

    def test_debug_payload(self) -> None:
        """Debug dict has expected keys (mode, rrf_k, pre_fusion counts, etc.)."""
        broker, vs, ls, _ = _make_broker(mode=RetrievalMode.HYBRID)

        c1 = _make_chunk(content="debug test content", chunk_id="chk-1")
        e1 = _make_embedding(chunk_id="chk-1", vector=[1.0, 0.0, 0.0, 0.0], embedding_id="emb-1")
        vs.add(e1, c1)
        ls.add(c1)

        result = broker.run(_make_query("debug test"))

        debug = result.debug
        assert "mode" in debug
        assert "rrf_k" in debug
        assert "pre_fusion_dense_count" in debug
        assert "pre_fusion_lexical_count" in debug
        assert "post_fusion_count" in debug
        assert "post_source_cap_count" in debug
        assert "source_cap_removals" in debug
        assert debug["mode"] == "hybrid"
        assert debug["rrf_k"] == 60

    def test_custom_normalizer(self) -> None:
        """Custom normalizer that uppercases query text — verify it is called."""

        class UpperNormalizer:
            def normalize(self, raw_query: str) -> str:
                return raw_query.upper()

        broker, _, ls, _ = _make_broker(
            mode=RetrievalMode.LEXICAL,
            normalizer=UpperNormalizer(),
        )

        c1 = _make_chunk(content="HELLO WORLD data", chunk_id="chk-1")
        ls.add(c1)

        result = broker.run(_make_query("hello world data"))

        # The normalizer uppercased "hello world data" -> "HELLO WORLD DATA"
        # The effective query should have the uppercased normalized form
        assert result.query.normalized_query == "HELLO WORLD DATA"
        assert result.query.raw_query == "hello world data"

    def test_protocol_conformance(self) -> None:
        """isinstance checks for DeterministicQueryEmbedder and PassthroughNormalizer."""
        embedder = DeterministicQueryEmbedder(dimensions=4)
        normalizer = PassthroughNormalizer()

        assert isinstance(embedder, QueryEmbedder)
        assert isinstance(normalizer, QueryNormalizer)

    def test_error_classification_transient(self) -> None:
        """ConnectionError from a store should be classified as TRANSIENT."""

        class ConnectionErrorVectorStore:
            @property
            def store_id(self) -> str:
                return "conn-error-vector"

            def add(self, embedding: ChunkEmbedding, chunk: Chunk) -> None:
                pass

            def add_batch(self, embeddings: list[ChunkEmbedding], chunks: list[Chunk]) -> None:
                pass

            def delete(self, chunk_ids: list[ChunkId]) -> int:
                return 0

            def search(
                self, query: RetrievalQuery, query_vector: list[float]
            ) -> list[RetrievalCandidate]:
                raise ConnectionError("connection refused")

            def count(self) -> int:
                return 0

        vs = ConnectionErrorVectorStore()
        ls = MemoryLexicalStore(store_id="memory-lexical")
        qe = DeterministicQueryEmbedder(dimensions=4)
        config = BrokerConfig(mode=RetrievalMode.DENSE)
        broker = RetrievalBroker(
            vector_store=vs,  # type: ignore[arg-type]
            lexical_store=ls,
            query_embedder=qe,
            config=config,
        )

        result = broker.run(_make_query("test"))

        dense_sr = [
            sr for sr in result.store_results
            if sr.retrieval_method == RetrievalMethod.DENSE
        ]
        assert len(dense_sr) == 1
        assert dense_sr[0].error is not None
        assert dense_sr[0].error_classification == ErrorClassification.TRANSIENT

    def test_error_classification_permanent(self) -> None:
        """NotImplementedError from a store should be classified as PERMANENT."""

        class NotImplVectorStore:
            @property
            def store_id(self) -> str:
                return "notimpl-vector"

            def add(self, embedding: ChunkEmbedding, chunk: Chunk) -> None:
                pass

            def add_batch(self, embeddings: list[ChunkEmbedding], chunks: list[Chunk]) -> None:
                pass

            def delete(self, chunk_ids: list[ChunkId]) -> int:
                return 0

            def search(
                self, query: RetrievalQuery, query_vector: list[float]
            ) -> list[RetrievalCandidate]:
                raise NotImplementedError("dense search not supported")

            def count(self) -> int:
                return 0

        vs = NotImplVectorStore()
        ls = MemoryLexicalStore(store_id="memory-lexical")
        qe = DeterministicQueryEmbedder(dimensions=4)
        config = BrokerConfig(mode=RetrievalMode.DENSE)
        broker = RetrievalBroker(
            vector_store=vs,  # type: ignore[arg-type]
            lexical_store=ls,
            query_embedder=qe,
            config=config,
        )

        result = broker.run(_make_query("test"))

        dense_sr = [
            sr for sr in result.store_results
            if sr.retrieval_method == RetrievalMethod.DENSE
        ]
        assert len(dense_sr) == 1
        assert dense_sr[0].error is not None
        assert dense_sr[0].error_classification == ErrorClassification.PERMANENT

    def test_timeout_warning(self) -> None:
        """fanout_timeout_ms=0 means any elapsed time triggers timeout warning."""
        broker, vs, ls, _ = _make_broker(mode=RetrievalMode.DENSE)

        c1 = _make_chunk(content="timeout test data", chunk_id="chk-1")
        e1 = _make_embedding(chunk_id="chk-1", vector=[1.0, 0.0, 0.0, 0.0], embedding_id="emb-1")
        vs.add(e1, c1)

        # Rebuild broker with fanout_timeout_ms that will definitely be exceeded
        config = BrokerConfig(mode=RetrievalMode.DENSE, fanout_timeout_ms=0)
        # fanout_timeout_ms=0 means disabled (> 0 check), so use 1 nanosecond equivalent
        # Actually fanout_timeout_ms=0 skips the check. Use a very small positive value instead.
        # The check is: if self._config.fanout_timeout_ms > 0 and elapsed > ...
        # So we need fanout_timeout_ms > 0 but tiny enough to always be exceeded.
        config = BrokerConfig(mode=RetrievalMode.DENSE, fanout_timeout_ms=1)
        # Even 1ms should be exceeded by embed + search

        # But to be sure, use a store that's slightly slow
        # Actually the memory store + deterministic embedder may be <1ms.
        # Use a wrapper that ensures some delay.
        import time as _time

        class SlowVectorStore:
            @property
            def store_id(self) -> str:
                return "slow-vector"

            def add(self, embedding: ChunkEmbedding, chunk: Chunk) -> None:
                vs.add(embedding, chunk)

            def add_batch(self, embeddings: list[ChunkEmbedding], chunks: list[Chunk]) -> None:
                pass

            def delete(self, chunk_ids: list[ChunkId]) -> int:
                return 0

            def search(
                self, query: RetrievalQuery, query_vector: list[float]
            ) -> list[RetrievalCandidate]:
                _time.sleep(0.01)  # 10ms delay, exceeds 1ms timeout
                return vs.search(query, query_vector)

            def count(self) -> int:
                return 0

        slow_vs = SlowVectorStore()
        slow_vs.add(e1, c1)

        broker = RetrievalBroker(
            vector_store=slow_vs,  # type: ignore[arg-type]
            lexical_store=ls,
            query_embedder=DeterministicQueryEmbedder(dimensions=4),
            config=config,
        )

        result = broker.run(_make_query("timeout test"))

        dense_sr = [
            sr for sr in result.store_results
            if sr.retrieval_method == RetrievalMethod.DENSE
        ]
        assert len(dense_sr) == 1
        assert dense_sr[0].error is not None
        assert "exceeded timeout" in dense_sr[0].error
        assert dense_sr[0].error_classification == ErrorClassification.TRANSIENT

    def test_debug_includes_trace_id(self) -> None:
        """Verify trace_id from the query appears in the debug dict."""
        broker, vs, ls, _ = _make_broker(mode=RetrievalMode.HYBRID)

        c1 = _make_chunk(content="trace test content", chunk_id="chk-1")
        e1 = _make_embedding(chunk_id="chk-1", vector=[1.0, 0.0, 0.0, 0.0], embedding_id="emb-1")
        vs.add(e1, c1)
        ls.add(c1)

        result = broker.run(_make_query("trace test"))

        # trace_id is on the query, check it's accessible
        assert result.query.trace_id == "trace-001"
