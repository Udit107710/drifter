"""Tests for retrieval store adapters: vector stores, lexical stores, and stubs."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import pytest

from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.common import RetrievalMethod
from libs.contracts.embeddings import ChunkEmbedding
from libs.contracts.retrieval import RetrievalCandidate, RetrievalQuery
from libs.retrieval.stores.memory_lexical_store import MemoryLexicalStore
from libs.retrieval.stores.memory_vector_store import MemoryVectorStore
from libs.adapters.config import OpenSearchConfig, QdrantConfig
from libs.retrieval.stores.opensearch_lexical_store import OpenSearchLexicalStore
from libs.retrieval.stores.opensearch_vector_store import OpenSearchVectorStore
from libs.retrieval.stores.protocols import LexicalStore, VectorStore
from libs.retrieval.stores.qdrant_vector_store import QdrantVectorStore

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


# ── Protocol conformance ──────────────────────────────────────────


class TestProtocolConformance:
    def test_memory_vector_store_is_vector_store(self) -> None:
        assert isinstance(MemoryVectorStore(), VectorStore)

    def test_memory_lexical_store_is_lexical_store(self) -> None:
        assert isinstance(MemoryLexicalStore(), LexicalStore)

    def test_qdrant_store_is_vector_store(self) -> None:
        assert isinstance(QdrantVectorStore(QdrantConfig()), VectorStore)

    def test_opensearch_lexical_is_lexical_store(self) -> None:
        assert isinstance(OpenSearchLexicalStore(OpenSearchConfig()), LexicalStore)

    def test_opensearch_vector_is_vector_store(self) -> None:
        assert isinstance(OpenSearchVectorStore(OpenSearchConfig()), VectorStore)


# ── MemoryVectorStore ─────────────────────────────────────────────


class TestMemoryVectorStore:
    def test_search_returns_candidates(self) -> None:
        store = MemoryVectorStore(store_id="vec-test")
        c1 = _make_chunk(content="first chunk", chunk_id="chk-1")
        c2 = _make_chunk(content="second chunk", chunk_id="chk-2")
        e1 = _make_embedding(chunk_id="chk-1", vector=[1.0, 0.0, 0.0], embedding_id="emb-1")
        e2 = _make_embedding(chunk_id="chk-2", vector=[0.0, 1.0, 0.0], embedding_id="emb-2")
        store.add(e1, c1)
        store.add(e2, c2)

        query = _make_query("search test")
        results = store.search(query, query_vector=[1.0, 0.0, 0.0])

        assert len(results) == 2
        for r in results:
            assert isinstance(r, RetrievalCandidate)
            assert r.retrieval_method == RetrievalMethod.DENSE
            assert r.store_id == "vec-test"

    def test_search_cosine_similarity_ordering(self) -> None:
        store = MemoryVectorStore()
        c1 = _make_chunk(content="matching chunk", chunk_id="chk-1")
        c2 = _make_chunk(content="other chunk", chunk_id="chk-2")
        e1 = _make_embedding(chunk_id="chk-1", vector=[1.0, 0.0, 0.0], embedding_id="emb-1")
        e2 = _make_embedding(chunk_id="chk-2", vector=[0.0, 1.0, 0.0], embedding_id="emb-2")
        store.add(e1, c1)
        store.add(e2, c2)

        results = store.search(_make_query(), query_vector=[1.0, 0.0, 0.0])

        assert len(results) == 2
        assert results[0].chunk.chunk_id == "chk-1"
        assert results[0].score > results[1].score

    def test_search_top_k_limits_results(self) -> None:
        store = MemoryVectorStore()
        for i in range(5):
            cid = f"chk-{i}"
            chunk = _make_chunk(content=f"chunk number {i}", chunk_id=cid)
            emb = _make_embedding(
                chunk_id=cid,
                vector=[float(i), 1.0, 0.0],
                embedding_id=f"emb-{i}",
            )
            store.add(emb, chunk)

        query = _make_query(top_k=2)
        results = store.search(query, query_vector=[1.0, 1.0, 0.0])

        assert len(results) == 2

    def test_search_metadata_filter(self) -> None:
        store = MemoryVectorStore()
        c1 = _make_chunk(
            content="science article", chunk_id="chk-1", metadata={"category": "science"}
        )
        c2 = _make_chunk(
            content="art article", chunk_id="chk-2", metadata={"category": "art"}
        )
        e1 = _make_embedding(chunk_id="chk-1", vector=[1.0, 0.0, 0.0], embedding_id="emb-1")
        e2 = _make_embedding(chunk_id="chk-2", vector=[0.0, 1.0, 0.0], embedding_id="emb-2")
        store.add(e1, c1)
        store.add(e2, c2)

        query = _make_query(filters={"category": "science"})
        results = store.search(query, query_vector=[0.5, 0.5, 0.0])

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "chk-1"

    def test_search_empty_store(self) -> None:
        store = MemoryVectorStore()
        results = store.search(_make_query(), query_vector=[1.0, 0.0, 0.0])
        assert results == []

    def test_count(self) -> None:
        store = MemoryVectorStore()
        assert store.count() == 0
        c1 = _make_chunk(content="first", chunk_id="chk-1")
        e1 = _make_embedding(chunk_id="chk-1", vector=[1.0, 0.0], embedding_id="emb-1")
        store.add(e1, c1)
        assert store.count() == 1
        c2 = _make_chunk(content="second", chunk_id="chk-2")
        e2 = _make_embedding(chunk_id="chk-2", vector=[0.0, 1.0], embedding_id="emb-2")
        store.add(e2, c2)
        assert store.count() == 2

    def test_delete(self) -> None:
        store = MemoryVectorStore()
        c1 = _make_chunk(content="first", chunk_id="chk-1")
        c2 = _make_chunk(content="second", chunk_id="chk-2")
        e1 = _make_embedding(chunk_id="chk-1", vector=[1.0, 0.0], embedding_id="emb-1")
        e2 = _make_embedding(chunk_id="chk-2", vector=[0.0, 1.0], embedding_id="emb-2")
        store.add(e1, c1)
        store.add(e2, c2)
        assert store.count() == 2

        deleted = store.delete(["chk-1"])
        assert deleted == 1
        assert store.count() == 1

        # Deleting non-existent returns 0
        assert store.delete(["chk-999"]) == 0

    def test_search_different_vectors_produce_different_scores(self) -> None:
        store = MemoryVectorStore()
        c1 = _make_chunk(content="chunk one", chunk_id="chk-1")
        c2 = _make_chunk(content="chunk two", chunk_id="chk-2")
        e1 = _make_embedding(chunk_id="chk-1", vector=[1.0, 0.0, 0.0], embedding_id="emb-1")
        e2 = _make_embedding(chunk_id="chk-2", vector=[0.0, 0.7, 0.7], embedding_id="emb-2")
        store.add(e1, c1)
        store.add(e2, c2)

        results = store.search(_make_query(), query_vector=[0.8, 0.2, 0.0])

        assert len(results) == 2
        assert results[0].score != results[1].score


# ── MemoryLexicalStore ────────────────────────────────────────────


class TestMemoryLexicalStore:
    def test_search_exact_term_match(self) -> None:
        store = MemoryLexicalStore()
        c1 = _make_chunk(content="the quick brown fox", chunk_id="chk-1")
        c2 = _make_chunk(content="a lazy dog sleeps", chunk_id="chk-2")
        store.add(c1)
        store.add(c2)

        results = store.search(_make_query("fox"))

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "chk-1"
        assert results[0].score > 0.0

    def test_search_term_overlap_scoring(self) -> None:
        store = MemoryLexicalStore()
        c1 = _make_chunk(content="python machine learning data", chunk_id="chk-1")
        c2 = _make_chunk(content="python programming basics", chunk_id="chk-2")
        store.add(c1)
        store.add(c2)

        results = store.search(_make_query("python machine learning"))

        assert len(results) == 2
        # c1 matches all three query terms, c2 matches only one
        assert results[0].chunk.chunk_id == "chk-1"
        assert results[0].score > results[1].score

    def test_search_no_match_returns_empty(self) -> None:
        store = MemoryLexicalStore()
        c1 = _make_chunk(content="the quick brown fox", chunk_id="chk-1")
        store.add(c1)

        results = store.search(_make_query("elephants"))

        assert results == []

    def test_search_top_k_limits_results(self) -> None:
        store = MemoryLexicalStore()
        for i in range(5):
            chunk = _make_chunk(content=f"common term variant {i}", chunk_id=f"chk-{i}")
            store.add(chunk)

        query = _make_query("common term", top_k=2)
        results = store.search(query)

        assert len(results) == 2

    def test_search_metadata_filter(self) -> None:
        store = MemoryLexicalStore()
        c1 = _make_chunk(
            content="python tutorial", chunk_id="chk-1", metadata={"category": "science"}
        )
        c2 = _make_chunk(
            content="python tutorial", chunk_id="chk-2", metadata={"category": "art"}
        )
        store.add(c1)
        store.add(c2)

        query = _make_query("python", filters={"category": "science"})
        results = store.search(query)

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "chk-1"

    def test_search_empty_store(self) -> None:
        store = MemoryLexicalStore()
        results = store.search(_make_query("anything"))
        assert results == []

    def test_count(self) -> None:
        store = MemoryLexicalStore()
        assert store.count() == 0
        c1 = _make_chunk(content="first chunk", chunk_id="chk-1")
        store.add(c1)
        assert store.count() == 1
        c2 = _make_chunk(content="second chunk", chunk_id="chk-2")
        store.add(c2)
        assert store.count() == 2

    def test_delete(self) -> None:
        store = MemoryLexicalStore()
        c1 = _make_chunk(content="first chunk", chunk_id="chk-1")
        c2 = _make_chunk(content="second chunk", chunk_id="chk-2")
        store.add(c1)
        store.add(c2)
        assert store.count() == 2

        deleted = store.delete(["chk-1"])
        assert deleted == 1
        assert store.count() == 1

        assert store.delete(["chk-999"]) == 0

    def test_search_case_insensitive(self) -> None:
        store = MemoryLexicalStore()
        c1 = _make_chunk(content="Python Machine Learning", chunk_id="chk-1")
        store.add(c1)

        results = store.search(_make_query("python machine"))

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "chk-1"
        assert results[0].score > 0.0


# ── Adapter stubs ─────────────────────────────────────────────────


class TestAdapterStubs:
    """Real adapters raise RuntimeError when called without connect()."""

    def test_qdrant_search_raises(self) -> None:
        store = QdrantVectorStore(QdrantConfig())
        with pytest.raises(RuntimeError, match="not connected"):
            store.search(_make_query(), query_vector=[1.0, 0.0])

    def test_opensearch_lexical_search_raises(self) -> None:
        store = OpenSearchLexicalStore(OpenSearchConfig())
        with pytest.raises(RuntimeError, match="not connected"):
            store.search(_make_query())

    def test_opensearch_vector_search_raises(self) -> None:
        store = OpenSearchVectorStore(OpenSearchConfig())
        with pytest.raises(RuntimeError, match="not connected"):
            store.search(_make_query(), query_vector=[1.0, 0.0])
