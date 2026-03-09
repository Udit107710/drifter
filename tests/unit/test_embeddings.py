"""Tests for the embeddings subsystem: models, protocols, and mock provider."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from libs.contracts.chunks import Chunk, ChunkLineage
from libs.embeddings.mock_provider import DeterministicEmbeddingProvider
from libs.embeddings.models import EmbeddingModelInfo
from libs.embeddings.protocols import EmbeddingProvider

# ── Helpers ─────────────────────────────────────────────────────────


def _make_chunk(
    content: str = "hello world",
    chunk_id: str = "chk-001",
    document_id: str = "doc-001",
    source_id: str = "src-001",
) -> Chunk:
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
    )


# ── Protocol conformance ──────────────────────────────────────────


class TestProtocolConformance:
    def test_deterministic_provider_is_embedding_provider(self) -> None:
        assert isinstance(DeterministicEmbeddingProvider(), EmbeddingProvider)


# ── EmbeddingModelInfo validation ─────────────────────────────────


class TestEmbeddingModelInfo:
    def test_valid_defaults(self) -> None:
        info = EmbeddingModelInfo(
            model_id="test-model",
            model_version="1.0",
            dimensions=128,
            max_tokens=512,
        )
        assert info.model_id == "test-model"
        assert info.dimensions == 128

    def test_empty_model_id_raises(self) -> None:
        with pytest.raises(ValueError, match="model_id must not be empty"):
            EmbeddingModelInfo(
                model_id="",
                model_version="1.0",
                dimensions=128,
                max_tokens=512,
            )

    def test_dimensions_less_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="dimensions must be >= 1"):
            EmbeddingModelInfo(
                model_id="test",
                model_version="1.0",
                dimensions=0,
                max_tokens=512,
            )

    def test_max_tokens_less_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            EmbeddingModelInfo(
                model_id="test",
                model_version="1.0",
                dimensions=128,
                max_tokens=0,
            )


# ── DeterministicEmbeddingProvider ────────────────────────────────


class TestDeterministicEmbeddingProvider:
    def test_model_info_returned(self) -> None:
        provider = DeterministicEmbeddingProvider(
            model_id="my-model", model_version="2.0", dimensions=32
        )
        info = provider.model_info()
        assert info.model_id == "my-model"
        assert info.model_version == "2.0"
        assert info.dimensions == 32
        assert info.max_tokens == 8192

    def test_deterministic_same_input_same_output(self) -> None:
        provider = DeterministicEmbeddingProvider()
        chunk = _make_chunk("reproducible content")
        result_a = provider.embed_chunks([chunk])
        result_b = provider.embed_chunks([chunk])
        assert result_a[0].vector == result_b[0].vector

    def test_different_content_different_vectors(self) -> None:
        provider = DeterministicEmbeddingProvider()
        chunk_a = _make_chunk("alpha content", chunk_id="chk-a")
        chunk_b = _make_chunk("beta content", chunk_id="chk-b")
        emb_a = provider.embed_chunks([chunk_a])[0]
        emb_b = provider.embed_chunks([chunk_b])[0]
        assert emb_a.vector != emb_b.vector

    def test_model_version_change_different_vectors(self) -> None:
        chunk = _make_chunk("same content")
        provider_v1 = DeterministicEmbeddingProvider(model_version="1.0")
        provider_v2 = DeterministicEmbeddingProvider(model_version="2.0")
        emb_v1 = provider_v1.embed_chunks([chunk])[0]
        emb_v2 = provider_v2.embed_chunks([chunk])[0]
        assert emb_v1.vector != emb_v2.vector

    def test_empty_batch_returns_empty_list(self) -> None:
        provider = DeterministicEmbeddingProvider()
        result = provider.embed_chunks([])
        assert result == []

    def test_batch_preserves_order(self) -> None:
        provider = DeterministicEmbeddingProvider()
        chunks = [
            _make_chunk("first", chunk_id="chk-1"),
            _make_chunk("second", chunk_id="chk-2"),
            _make_chunk("third", chunk_id="chk-3"),
        ]
        embeddings = provider.embed_chunks(chunks)
        assert len(embeddings) == 3
        for chunk, emb in zip(chunks, embeddings, strict=True):
            assert emb.chunk_id == chunk.chunk_id

    def test_vector_dimensions_match_model_info(self) -> None:
        dims = 48
        provider = DeterministicEmbeddingProvider(dimensions=dims)
        chunk = _make_chunk("dimension test")
        emb = provider.embed_chunks([chunk])[0]
        assert len(emb.vector) == dims
        assert emb.dimensions == dims
