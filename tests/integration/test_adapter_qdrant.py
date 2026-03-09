"""Integration tests for QdrantVectorStore against a real Qdrant instance.

Run with: uv run pytest tests/integration/test_adapter_qdrant.py -v
Requires: docker compose up -d qdrant
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

import pytest

from libs.adapters.config import QdrantConfig
from libs.adapters.qdrant import QdrantVectorStore
from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.common import RetrievalMethod
from libs.contracts.embeddings import ChunkEmbedding
from libs.contracts.retrieval import RetrievalQuery


def _port_open(host: str, port: int) -> bool:
    import socket

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


@pytest.fixture()
def qdrant_store():
    """Provide a connected QdrantVectorStore, cleaned up after the test."""
    if not _port_open("localhost", 6333):
        pytest.skip("Qdrant not running on localhost:6333")

    collection = f"drifter_test_{uuid.uuid4().hex[:8]}"
    config = QdrantConfig(collection_name=collection)
    store = QdrantVectorStore(config)
    store.connect()

    yield store

    # Cleanup: delete the test collection
    try:
        if store._client is not None:
            store._client.delete_collection(collection)
    except Exception:
        pass
    store.close()


class TestQdrantVectorStore:
    def test_health_check(self, qdrant_store: QdrantVectorStore) -> None:
        assert qdrant_store.health_check() is True

    def test_add_and_count(self, qdrant_store: QdrantVectorStore) -> None:
        chunk = _make_chunk("the quick brown fox")
        emb = _make_embedding(chunk, [0.1, 0.2, 0.3, 0.4])
        qdrant_store.add(emb, chunk)
        assert qdrant_store.count() == 1

    def test_add_batch_and_count(self, qdrant_store: QdrantVectorStore) -> None:
        chunks = [_make_chunk(f"chunk number {i}") for i in range(5)]
        embeddings = [
            _make_embedding(c, [0.1 + float(i) * 0.1] * 4) for i, c in enumerate(chunks)
        ]
        qdrant_store.add_batch(embeddings, chunks)
        assert qdrant_store.count() == 5

    def test_search_returns_candidates(self, qdrant_store: QdrantVectorStore) -> None:
        chunk = _make_chunk("machine learning is great")
        emb = _make_embedding(chunk, [1.0, 0.0, 0.0, 0.0])
        qdrant_store.add(emb, chunk)

        query = RetrievalQuery(
            raw_query="machine learning",
            normalized_query="machine learning",
            trace_id="t1",
            top_k=5,
        )
        results = qdrant_store.search(query, query_vector=[1.0, 0.0, 0.0, 0.0])
        assert len(results) == 1
        assert results[0].chunk.content == "machine learning is great"
        assert results[0].retrieval_method == RetrievalMethod.DENSE
        assert results[0].store_id == qdrant_store.store_id
        assert results[0].score > 0.0

    def test_search_reconstructs_chunk_fully(self, qdrant_store: QdrantVectorStore) -> None:
        chunk = _make_chunk("full reconstruction test", metadata={"lang": "en"})
        emb = _make_embedding(chunk, [0.5, 0.5, 0.0, 0.0])
        qdrant_store.add(emb, chunk)

        query = RetrievalQuery(
            raw_query="test",
            normalized_query="test",
            trace_id="t2",
            top_k=5,
        )
        results = qdrant_store.search(query, query_vector=[0.5, 0.5, 0.0, 0.0])
        assert len(results) == 1
        result_chunk = results[0].chunk
        assert result_chunk.chunk_id == chunk.chunk_id
        assert result_chunk.document_id == chunk.document_id
        assert result_chunk.source_id == chunk.source_id
        assert result_chunk.content == chunk.content
        assert result_chunk.token_count == chunk.token_count
        assert result_chunk.metadata == {"lang": "en"}
        assert result_chunk.lineage.chunk_strategy == "fixed"

    def test_delete(self, qdrant_store: QdrantVectorStore) -> None:
        chunk = _make_chunk("to be deleted")
        emb = _make_embedding(chunk, [0.1, 0.1, 0.1, 0.1])
        qdrant_store.add(emb, chunk)
        assert qdrant_store.count() == 1

        deleted = qdrant_store.delete([chunk.chunk_id])
        assert deleted == 1
        assert qdrant_store.count() == 0

    def test_search_empty_collection(self, qdrant_store: QdrantVectorStore) -> None:
        query = RetrievalQuery(
            raw_query="anything",
            normalized_query="anything",
            trace_id="t3",
            top_k=5,
        )
        results = qdrant_store.search(query, query_vector=[1.0, 0.0, 0.0, 0.0])
        assert results == []

    def test_search_top_k(self, qdrant_store: QdrantVectorStore) -> None:
        chunks = [_make_chunk(f"item {i}") for i in range(10)]
        embeddings = [
            _make_embedding(c, [0.1 + float(i) / 10] * 4) for i, c in enumerate(chunks)
        ]
        qdrant_store.add_batch(embeddings, chunks)

        query = RetrievalQuery(
            raw_query="items",
            normalized_query="items",
            trace_id="t4",
            top_k=3,
        )
        results = qdrant_store.search(query, query_vector=[0.9, 0.9, 0.9, 0.9])
        assert len(results) == 3
