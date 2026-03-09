"""Tests for chunk contracts: Chunk, ChunkLineage."""

from datetime import UTC, datetime

import pytest

from libs.contracts.chunks import Chunk, ChunkLineage


def _make_lineage(**overrides: object) -> ChunkLineage:
    defaults: dict[str, object] = {
        "source_id": "src-001",
        "document_id": "doc-001",
        "block_ids": ["blk-001"],
        "chunk_strategy": "fixed_size",
        "parser_version": "1.0.0",
        "created_at": datetime(2025, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return ChunkLineage(**defaults)  # type: ignore[arg-type]


def _make_chunk(**overrides: object) -> Chunk:
    defaults: dict[str, object] = {
        "chunk_id": "chk-001",
        "document_id": "doc-001",
        "source_id": "src-001",
        "block_ids": ["blk-001"],
        "content": "Chunk text content.",
        "content_hash": "sha256:abc123",
        "token_count": 5,
        "strategy": "fixed_size",
        "byte_offset_start": 0,
        "byte_offset_end": 19,
        "lineage": _make_lineage(),
    }
    defaults.update(overrides)
    return Chunk(**defaults)  # type: ignore[arg-type]


# ── ChunkLineage ────────────────────────────────────────────────────


class TestChunkLineage:
    def test_create_valid(self) -> None:
        lineage = _make_lineage()
        assert lineage.source_id == "src-001"
        assert lineage.chunk_strategy == "fixed_size"

    def test_schema_version_default(self) -> None:
        lineage = _make_lineage()
        assert lineage.schema_version == 1

    def test_schema_version_custom(self) -> None:
        lineage = _make_lineage(schema_version=2)
        assert lineage.schema_version == 2

    def test_empty_source_id_raises(self) -> None:
        with pytest.raises(ValueError, match="source_id"):
            _make_lineage(source_id="")

    def test_empty_document_id_raises(self) -> None:
        with pytest.raises(ValueError, match="document_id"):
            _make_lineage(document_id="")

    def test_empty_block_ids_raises(self) -> None:
        with pytest.raises(ValueError, match="block_ids"):
            _make_lineage(block_ids=[])

    def test_empty_chunk_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_strategy"):
            _make_lineage(chunk_strategy="")

    def test_empty_parser_version_raises(self) -> None:
        with pytest.raises(ValueError, match="parser_version"):
            _make_lineage(parser_version="")


# ── Chunk ────────────────────────────────────────────────────────────


class TestChunk:
    def test_create_valid(self) -> None:
        chunk = _make_chunk()
        assert chunk.chunk_id == "chk-001"
        assert chunk.token_count == 5
        assert chunk.acl == []

    def test_schema_version_default(self) -> None:
        chunk = _make_chunk()
        assert chunk.schema_version == 1

    def test_schema_version_custom(self) -> None:
        chunk = _make_chunk(schema_version=2)
        assert chunk.schema_version == 2

    def test_with_acl(self) -> None:
        chunk = _make_chunk(acl=["role:admin"])
        assert chunk.acl == ["role:admin"]

    def test_lineage_integrity(self) -> None:
        """Chunk's source_id/document_id should match its lineage."""
        chunk = _make_chunk()
        assert chunk.source_id == chunk.lineage.source_id
        assert chunk.document_id == chunk.lineage.document_id

    def test_empty_chunk_id_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_id"):
            _make_chunk(chunk_id="")

    def test_empty_content_raises(self) -> None:
        with pytest.raises(ValueError, match="content"):
            _make_chunk(content="")

    def test_empty_content_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="content_hash"):
            _make_chunk(content_hash="")

    def test_zero_token_count_raises(self) -> None:
        with pytest.raises(ValueError, match="token_count"):
            _make_chunk(token_count=0)

    def test_negative_byte_offset_start_raises(self) -> None:
        with pytest.raises(ValueError, match="byte_offset_start"):
            _make_chunk(byte_offset_start=-1)

    def test_byte_offset_end_not_greater_raises(self) -> None:
        with pytest.raises(ValueError, match="byte_offset_end"):
            _make_chunk(byte_offset_start=10, byte_offset_end=10)

    def test_empty_block_ids_raises(self) -> None:
        with pytest.raises(ValueError, match="block_ids"):
            _make_chunk(block_ids=[])

    def test_multi_block_chunk(self) -> None:
        chunk = _make_chunk(
            block_ids=["blk-001", "blk-002"],
            lineage=_make_lineage(block_ids=["blk-001", "blk-002"]),
        )
        assert len(chunk.block_ids) == 2
