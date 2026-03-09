"""Tests for source-cap deduplication of fused retrieval candidates."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.common import RetrievalMethod
from libs.retrieval.broker.dedup import apply_source_caps
from libs.retrieval.broker.models import FusedCandidate

# ── Helpers ─────────────────────────────────────────────────────────


def _make_fused(
    chunk_id: str,
    source_id: str,
    score: float,
) -> FusedCandidate:
    """Create a FusedCandidate with minimal fields."""
    content = f"content for {chunk_id}"
    content_hash = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc-001",
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
            document_id="doc-001",
            block_ids=["blk-001"],
            chunk_strategy="fixed_window",
            parser_version="test:1.0.0",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
    )
    return FusedCandidate(
        chunk=chunk,
        fused_score=score,
        retrieval_method=RetrievalMethod.HYBRID,
        contributing_stores=["vec-store", "lex-store"],
        per_store_ranks={"vec-store": 1, "lex-store": 1},
        per_store_scores={"vec-store": score, "lex-store": score},
    )


# ── Tests ───────────────────────────────────────────────────────────


class TestSourceCaps:
    def test_caps_enforced(self) -> None:
        """5 candidates from same source, max_per_source=2 -> 2 remain."""
        candidates = [
            _make_fused(f"chk-{i}", "src-A", score=1.0 - i * 0.1)
            for i in range(5)
        ]

        result = apply_source_caps(candidates, max_per_source=2)

        assert len(result) == 2
        # Top 2 by score should survive
        assert result[0].chunk.chunk_id == "chk-0"
        assert result[1].chunk.chunk_id == "chk-1"

    def test_zero_cap_is_passthrough(self) -> None:
        """max_per_source=0 -> all candidates returned."""
        candidates = [
            _make_fused(f"chk-{i}", "src-A", score=1.0 - i * 0.1)
            for i in range(5)
        ]

        result = apply_source_caps(candidates, max_per_source=0)

        assert len(result) == 5

    def test_mixed_sources(self) -> None:
        """Some sources capped, others not."""
        candidates = [
            _make_fused("chk-a1", "src-A", score=0.9),
            _make_fused("chk-a2", "src-A", score=0.85),
            _make_fused("chk-a3", "src-A", score=0.8),
            _make_fused("chk-b1", "src-B", score=0.75),
            _make_fused("chk-c1", "src-C", score=0.7),
        ]

        result = apply_source_caps(candidates, max_per_source=2)

        assert len(result) == 4
        result_ids = [fc.chunk.chunk_id for fc in result]
        # src-A capped to 2, src-B and src-C unaffected
        assert result_ids == ["chk-a1", "chk-a2", "chk-b1", "chk-c1"]

    def test_preserves_order(self) -> None:
        """Output order matches input order (by score)."""
        candidates = [
            _make_fused("chk-a1", "src-A", score=0.9),
            _make_fused("chk-b1", "src-B", score=0.85),
            _make_fused("chk-a2", "src-A", score=0.8),
            _make_fused("chk-b2", "src-B", score=0.75),
            _make_fused("chk-a3", "src-A", score=0.7),
        ]

        result = apply_source_caps(candidates, max_per_source=2)

        result_ids = [fc.chunk.chunk_id for fc in result]
        assert result_ids == ["chk-a1", "chk-b1", "chk-a2", "chk-b2"]

    def test_empty_input(self) -> None:
        """Empty list -> empty result."""
        result = apply_source_caps([], max_per_source=2)
        assert result == []
