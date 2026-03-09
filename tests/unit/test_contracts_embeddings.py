"""Tests for embedding contracts: ChunkEmbedding."""

from datetime import UTC, datetime

import pytest

from libs.contracts.embeddings import ChunkEmbedding


def _make_embedding(**overrides: object) -> ChunkEmbedding:
    defaults: dict[str, object] = {
        "embedding_id": "emb-001",
        "chunk_id": "chk-001",
        "vector": [0.1, 0.2, 0.3],
        "model_id": "bge-base-en-v1.5",
        "model_version": "1.0",
        "dimensions": 3,
        "created_at": datetime(2025, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return ChunkEmbedding(**defaults)  # type: ignore[arg-type]


class TestChunkEmbedding:
    def test_create_valid(self) -> None:
        emb = _make_embedding()
        assert emb.embedding_id == "emb-001"
        assert emb.dimensions == 3
        assert len(emb.vector) == 3

    def test_schema_version_default(self) -> None:
        emb = _make_embedding()
        assert emb.schema_version == 1

    def test_schema_version_custom(self) -> None:
        emb = _make_embedding(schema_version=2)
        assert emb.schema_version == 2

    def test_empty_embedding_id_raises(self) -> None:
        with pytest.raises(ValueError, match="embedding_id"):
            _make_embedding(embedding_id="")

    def test_empty_chunk_id_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_id"):
            _make_embedding(chunk_id="")

    def test_empty_vector_raises(self) -> None:
        with pytest.raises(ValueError, match="vector must not be empty"):
            _make_embedding(vector=[], dimensions=0)

    def test_empty_model_id_raises(self) -> None:
        with pytest.raises(ValueError, match="model_id"):
            _make_embedding(model_id="")

    def test_empty_model_version_raises(self) -> None:
        with pytest.raises(ValueError, match="model_version"):
            _make_embedding(model_version="")

    def test_zero_dimensions_raises(self) -> None:
        with pytest.raises(ValueError, match="dimensions must be >= 1"):
            _make_embedding(vector=[0.1], dimensions=0)

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match=r"vector length.*must match dimensions"):
            _make_embedding(vector=[0.1, 0.2], dimensions=3)

    def test_high_dimensional_vector(self) -> None:
        vec = [0.0] * 768
        emb = _make_embedding(vector=vec, dimensions=768)
        assert emb.dimensions == 768
        assert len(emb.vector) == 768
