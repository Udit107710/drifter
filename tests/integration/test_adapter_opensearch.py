"""Integration tests for OpenSearch adapters against a real OpenSearch instance.

Run with: uv run pytest tests/integration/test_adapter_opensearch.py -v
Requires: docker compose up -d opensearch
"""

from __future__ import annotations

import hashlib
import socket
import time
import uuid
from datetime import UTC, datetime

import pytest

from libs.adapters.config import OpenSearchConfig
from libs.adapters.opensearch import OpenSearchLexicalStore, OpenSearchVectorStore
from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.common import RetrievalMethod
from libs.contracts.embeddings import ChunkEmbedding
from libs.contracts.retrieval import RetrievalQuery


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _make_chunk(
    content: str = "hello world",
    chunk_id: str | None = None,
    document_id: str = "doc-001",
    source_id: str = "src-001",
    metadata: dict | None = None,
) -> Chunk:
    cid = chunk_id or str(uuid.uuid4())
    return Chunk(
        chunk_id=cid,
        document_id=document_id,
        source_id=source_id,
        block_ids=["b1"],
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        token_count=len(content.split()),
        strategy="fixed",
        byte_offset_start=0,
        byte_offset_end=len(content),
        lineage=ChunkLineage(
            source_id=source_id,
            document_id=document_id,
            block_ids=["b1"],
            chunk_strategy="fixed",
            parser_version="test-1.0",
            created_at=datetime.now(UTC),
        ),
        metadata=metadata or {},
    )


def _make_embedding(chunk: Chunk, vector: list[float]) -> ChunkEmbedding:
    return ChunkEmbedding(
        embedding_id=str(uuid.uuid4()),
        chunk_id=chunk.chunk_id,
        vector=vector,
        model_id="test-model",
        model_version="1.0",
        dimensions=len(vector),
        created_at=datetime.now(UTC),
    )


def _unique_config() -> OpenSearchConfig:
    """Return a config with a unique index prefix per call."""
    prefix = f"drifter_test_{uuid.uuid4().hex[:8]}"
    return OpenSearchConfig(
        hosts=["localhost:9200"],
        username="admin",
        password="admin",
        use_ssl=False,
        index_prefix=prefix,
    )


def _cleanup_indices(client, prefix: str) -> None:
    """Delete all test indices."""
    try:
        indices = client.indices.get(f"{prefix}-*")
        for idx in indices:
            client.indices.delete(idx)
    except Exception:
        pass


# ── Lexical Store Tests ──────────────────────────────────────────────


@pytest.fixture()
def lexical_store():
    if not _port_open("localhost", 9200):
        pytest.skip("OpenSearch not running on localhost:9200")

    config = _unique_config()
    store = OpenSearchLexicalStore(config)
    store.connect()

    yield store

    _cleanup_indices(store._client, config.index_prefix)
    store.close()


class TestOpenSearchLexicalStore:
    def test_health_check(self, lexical_store: OpenSearchLexicalStore) -> None:
        assert lexical_store.health_check() is True

    def test_add_and_count(self, lexical_store: OpenSearchLexicalStore) -> None:
        chunk = _make_chunk("the quick brown fox jumps over the lazy dog")
        lexical_store.add(chunk)
        assert lexical_store.count() == 1

    def test_add_batch_and_count(self, lexical_store: OpenSearchLexicalStore) -> None:
        chunks = [_make_chunk(f"batch document number {i}") for i in range(5)]
        lexical_store.add_batch(chunks)
        assert lexical_store.count() == 5

    def test_search_bm25(self, lexical_store: OpenSearchLexicalStore) -> None:
        lexical_store.add(_make_chunk("machine learning algorithms"))
        lexical_store.add(_make_chunk("cooking recipes for dinner"))

        query = RetrievalQuery(
            raw_query="machine learning",
            normalized_query="machine learning",
            trace_id="t1",
            top_k=10,
        )
        results = lexical_store.search(query)
        assert len(results) >= 1
        assert "machine" in results[0].chunk.content
        assert results[0].retrieval_method == RetrievalMethod.LEXICAL
        assert results[0].score > 0.0

    def test_search_reconstructs_chunk(self, lexical_store: OpenSearchLexicalStore) -> None:
        chunk = _make_chunk("full reconstruction via opensearch", metadata={"lang": "en"})
        lexical_store.add(chunk)

        query = RetrievalQuery(
            raw_query="reconstruction",
            normalized_query="reconstruction",
            trace_id="t2",
            top_k=5,
        )
        results = lexical_store.search(query)
        assert len(results) == 1
        result_chunk = results[0].chunk
        assert result_chunk.chunk_id == chunk.chunk_id
        assert result_chunk.document_id == chunk.document_id
        assert result_chunk.content == chunk.content
        assert result_chunk.metadata == {"lang": "en"}
        assert result_chunk.lineage.parser_version == "test-1.0"

    def test_delete(self, lexical_store: OpenSearchLexicalStore) -> None:
        chunk = _make_chunk("to be deleted from opensearch")
        lexical_store.add(chunk)
        assert lexical_store.count() == 1

        deleted = lexical_store.delete([chunk.chunk_id])
        assert deleted == 1
        assert lexical_store.count() == 0

    def test_search_empty_index(self, lexical_store: OpenSearchLexicalStore) -> None:
        query = RetrievalQuery(
            raw_query="anything",
            normalized_query="anything",
            trace_id="t3",
            top_k=5,
        )
        results = lexical_store.search(query)
        assert results == []


# ── Vector Store Tests ───────────────────────────────────────────────


@pytest.fixture()
def vector_store():
    if not _port_open("localhost", 9200):
        pytest.skip("OpenSearch not running on localhost:9200")

    config = _unique_config()
    store = OpenSearchVectorStore(config)
    store.connect()

    yield store

    _cleanup_indices(store._client, config.index_prefix)
    store.close()


class TestOpenSearchVectorStore:
    def test_health_check(self, vector_store: OpenSearchVectorStore) -> None:
        assert vector_store.health_check() is True

    def test_add_and_count(self, vector_store: OpenSearchVectorStore) -> None:
        chunk = _make_chunk("vector document")
        emb = _make_embedding(chunk, [0.1, 0.2, 0.3, 0.4])
        vector_store.add(emb, chunk)
        assert vector_store.count() == 1

    def test_add_batch_and_count(self, vector_store: OpenSearchVectorStore) -> None:
        chunks = [_make_chunk(f"vector batch {i}") for i in range(5)]
        embeddings = [
            _make_embedding(c, [0.1 + float(i) * 0.1] * 4) for i, c in enumerate(chunks)
        ]
        vector_store.add_batch(embeddings, chunks)
        assert vector_store.count() == 5

    def test_search_knn(self, vector_store: OpenSearchVectorStore) -> None:
        chunk = _make_chunk("neural networks and deep learning")
        emb = _make_embedding(chunk, [1.0, 0.0, 0.0, 0.0])
        vector_store.add(emb, chunk)

        # Allow index to refresh
        time.sleep(1)

        query = RetrievalQuery(
            raw_query="neural nets",
            normalized_query="neural nets",
            trace_id="t1",
            top_k=5,
        )
        results = vector_store.search(query, query_vector=[1.0, 0.0, 0.0, 0.0])
        assert len(results) == 1
        assert results[0].chunk.content == "neural networks and deep learning"
        assert results[0].retrieval_method == RetrievalMethod.DENSE
        assert results[0].score > 0.0

    def test_search_reconstructs_chunk(self, vector_store: OpenSearchVectorStore) -> None:
        chunk = _make_chunk("opensearch vector reconstruction", metadata={"topic": "ml"})
        emb = _make_embedding(chunk, [0.5, 0.5, 0.0, 0.0])
        vector_store.add(emb, chunk)

        time.sleep(1)

        query = RetrievalQuery(
            raw_query="test",
            normalized_query="test",
            trace_id="t2",
            top_k=5,
        )
        results = vector_store.search(query, query_vector=[0.5, 0.5, 0.0, 0.0])
        assert len(results) == 1
        result_chunk = results[0].chunk
        assert result_chunk.chunk_id == chunk.chunk_id
        assert result_chunk.content == chunk.content
        assert result_chunk.metadata == {"topic": "ml"}

    def test_delete(self, vector_store: OpenSearchVectorStore) -> None:
        chunk = _make_chunk("vector to delete")
        emb = _make_embedding(chunk, [0.1, 0.1, 0.1, 0.1])
        vector_store.add(emb, chunk)
        assert vector_store.count() == 1

        deleted = vector_store.delete([chunk.chunk_id])
        assert deleted == 1
        assert vector_store.count() == 0
