"""Tests for context contracts: ContextItem, ContextPack."""

from datetime import UTC, datetime

import pytest

from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.common import SelectionReason
from libs.contracts.context import ContextItem, ContextPack


def _make_chunk() -> Chunk:
    return Chunk(
        chunk_id="chk-001",
        document_id="doc-001",
        source_id="src-001",
        block_ids=["blk-001"],
        content="Test content.",
        content_hash="sha256:abc123",
        token_count=3,
        strategy="fixed_size",
        byte_offset_start=0,
        byte_offset_end=13,
        lineage=ChunkLineage(
            source_id="src-001",
            document_id="doc-001",
            block_ids=["blk-001"],
            chunk_strategy="fixed_size",
            parser_version="1.0.0",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
    )


def _make_context_item(**overrides: object) -> ContextItem:
    defaults: dict[str, object] = {
        "chunk": _make_chunk(),
        "rank": 1,
        "token_count": 3,
        "selection_reason": SelectionReason.TOP_RANKED,
    }
    defaults.update(overrides)
    return ContextItem(**defaults)  # type: ignore[arg-type]


# ── ContextItem ─────────────────────────────────────────────────────


class TestContextItem:
    def test_create_valid(self) -> None:
        item = _make_context_item()
        assert item.rank == 1
        assert item.selection_reason == SelectionReason.TOP_RANKED

    def test_zero_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            _make_context_item(rank=0)

    def test_zero_token_count_raises(self) -> None:
        with pytest.raises(ValueError, match="token_count"):
            _make_context_item(token_count=0)


# ── ContextPack ─────────────────────────────────────────────────────


class TestContextPack:
    def test_create_valid(self) -> None:
        pack = ContextPack(
            query="What is RAG?",
            evidence=[_make_context_item()],
            total_tokens=3,
            token_budget=1000,
            diversity_score=0.5,
        )
        assert len(pack.evidence) == 1
        assert pack.total_tokens == 3

    def test_schema_version_default(self) -> None:
        pack = ContextPack(
            query="What is RAG?",
            evidence=[_make_context_item()],
            total_tokens=3,
            token_budget=1000,
            diversity_score=0.5,
        )
        assert pack.schema_version == 1

    def test_schema_version_custom(self) -> None:
        pack = ContextPack(
            query="What is RAG?",
            evidence=[_make_context_item()],
            total_tokens=3,
            token_budget=1000,
            diversity_score=0.5,
            schema_version=2,
        )
        assert pack.schema_version == 2

    def test_empty_evidence_allowed(self) -> None:
        pack = ContextPack(
            query="query",
            evidence=[],
            total_tokens=0,
            token_budget=100,
            diversity_score=0.0,
        )
        assert len(pack.evidence) == 0

    def test_empty_query_raises(self) -> None:
        with pytest.raises(ValueError, match="query"):
            ContextPack(
                query="",
                evidence=[],
                total_tokens=0,
                token_budget=100,
                diversity_score=0.0,
            )

    def test_negative_total_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="total_tokens"):
            ContextPack(
                query="q",
                evidence=[],
                total_tokens=-1,
                token_budget=100,
                diversity_score=0.0,
            )

    def test_zero_token_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="token_budget"):
            ContextPack(
                query="q",
                evidence=[],
                total_tokens=0,
                token_budget=0,
                diversity_score=0.0,
            )

    def test_tokens_exceed_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="total_tokens must not exceed"):
            ContextPack(
                query="q",
                evidence=[],
                total_tokens=101,
                token_budget=100,
                diversity_score=0.0,
            )

    def test_diversity_score_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="diversity_score"):
            ContextPack(
                query="q",
                evidence=[],
                total_tokens=0,
                token_budget=100,
                diversity_score=1.5,
            )

    def test_chunk_ids_property(self) -> None:
        pack = ContextPack(
            query="test",
            evidence=[_make_context_item()],
            total_tokens=3,
            token_budget=1000,
            diversity_score=0.5,
        )
        assert pack.chunk_ids == ["chk-001"]

    def test_source_ids_property(self) -> None:
        pack = ContextPack(
            query="test",
            evidence=[_make_context_item(), _make_context_item()],
            total_tokens=6,
            token_budget=1000,
            diversity_score=0.5,
        )
        # Duplicate source_ids should be deduplicated
        assert pack.source_ids == ["src-001"]

    def test_empty_evidence_properties(self) -> None:
        pack = ContextPack(
            query="test",
            evidence=[],
            total_tokens=0,
            token_budget=100,
            diversity_score=0.0,
        )
        assert pack.chunk_ids == []
        assert pack.source_ids == []
