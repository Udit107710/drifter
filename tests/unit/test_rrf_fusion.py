"""Tests for reciprocal rank fusion (RRF) merging of ranked retrieval lists."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

import pytest

from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.common import RetrievalMethod
from libs.contracts.retrieval import RetrievalCandidate
from libs.retrieval.broker.fusion import reciprocal_rank_fusion

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


def _make_candidate(
    chunk: Chunk,
    score: float,
    method: RetrievalMethod,
    store_id: str,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=chunk,
        score=score,
        retrieval_method=method,
        store_id=store_id,
    )


# ── Tests ───────────────────────────────────────────────────────────


class TestRRFFusion:
    def test_no_overlap(self) -> None:
        """Two lists with different chunks — all get single-store RRF scores."""
        c1 = _make_chunk(content="alpha chunk", chunk_id="chk-1")
        c2 = _make_chunk(content="beta chunk", chunk_id="chk-2")
        c3 = _make_chunk(content="gamma chunk", chunk_id="chk-3")
        c4 = _make_chunk(content="delta chunk", chunk_id="chk-4")

        list_a = [
            _make_candidate(c1, 0.9, RetrievalMethod.DENSE, "vec-store"),
            _make_candidate(c2, 0.7, RetrievalMethod.DENSE, "vec-store"),
        ]
        list_b = [
            _make_candidate(c3, 0.8, RetrievalMethod.LEXICAL, "lex-store"),
            _make_candidate(c4, 0.6, RetrievalMethod.LEXICAL, "lex-store"),
        ]

        fused = reciprocal_rank_fusion([list_a, list_b], weights=[1.0, 1.0])

        assert len(fused) == 4
        chunk_ids = [fc.chunk.chunk_id for fc in fused]
        assert set(chunk_ids) == {"chk-1", "chk-2", "chk-3", "chk-4"}

        # Each candidate should appear in exactly one store
        for fc in fused:
            assert len(fc.contributing_stores) == 1
            assert fc.retrieval_method in (RetrievalMethod.DENSE, RetrievalMethod.LEXICAL)

    def test_full_overlap(self) -> None:
        """Same chunk in both lists — gets combined score and HYBRID method."""
        c1 = _make_chunk(content="shared chunk", chunk_id="chk-shared")

        list_a = [_make_candidate(c1, 0.9, RetrievalMethod.DENSE, "vec-store")]
        list_b = [_make_candidate(c1, 0.8, RetrievalMethod.LEXICAL, "lex-store")]

        fused = reciprocal_rank_fusion([list_a, list_b], weights=[1.0, 1.0])

        assert len(fused) == 1
        fc = fused[0]
        assert fc.chunk.chunk_id == "chk-shared"
        assert fc.retrieval_method == RetrievalMethod.HYBRID
        assert set(fc.contributing_stores) == {"vec-store", "lex-store"}

        # Combined score = 1/(60+1) + 1/(60+1)
        expected = 1.0 / 61 + 1.0 / 61
        assert abs(fc.fused_score - expected) < 1e-9

    def test_partial_overlap(self) -> None:
        """Mix of overlapping and unique chunks."""
        shared = _make_chunk(content="shared chunk", chunk_id="chk-shared")
        unique_a = _make_chunk(content="only in dense", chunk_id="chk-dense")
        unique_b = _make_chunk(content="only in lexical", chunk_id="chk-lex")

        list_a = [
            _make_candidate(shared, 0.9, RetrievalMethod.DENSE, "vec-store"),
            _make_candidate(unique_a, 0.5, RetrievalMethod.DENSE, "vec-store"),
        ]
        list_b = [
            _make_candidate(shared, 0.8, RetrievalMethod.LEXICAL, "lex-store"),
            _make_candidate(unique_b, 0.4, RetrievalMethod.LEXICAL, "lex-store"),
        ]

        fused = reciprocal_rank_fusion([list_a, list_b], weights=[1.0, 1.0])

        assert len(fused) == 3
        # Shared chunk should be first (two contributions)
        assert fused[0].chunk.chunk_id == "chk-shared"
        assert fused[0].retrieval_method == RetrievalMethod.HYBRID

        # Unique chunks have single-method attribution
        unique_ids = {fc.chunk.chunk_id: fc for fc in fused[1:]}
        assert unique_ids["chk-dense"].retrieval_method == RetrievalMethod.DENSE
        assert unique_ids["chk-lex"].retrieval_method == RetrievalMethod.LEXICAL

    def test_deterministic_ordering(self) -> None:
        """Same fused scores tie-break by max individual score."""
        c1 = _make_chunk(content="high raw score", chunk_id="chk-1")
        c2 = _make_chunk(content="low raw score", chunk_id="chk-2")

        # Both at rank 1 in their respective lists → same RRF score
        list_a = [_make_candidate(c1, 0.95, RetrievalMethod.DENSE, "vec-store")]
        list_b = [_make_candidate(c2, 0.30, RetrievalMethod.LEXICAL, "lex-store")]

        fused = reciprocal_rank_fusion([list_a, list_b], weights=[1.0, 1.0])

        assert len(fused) == 2
        # Same fused score (1/(60+1)), but c1 has higher raw score
        assert abs(fused[0].fused_score - fused[1].fused_score) < 1e-9
        assert fused[0].chunk.chunk_id == "chk-1"
        assert fused[1].chunk.chunk_id == "chk-2"

    def test_asymmetric_weights(self) -> None:
        """Weight [2.0, 1.0] favors first list's top candidates."""
        c1 = _make_chunk(content="dense top", chunk_id="chk-dense")
        c2 = _make_chunk(content="lexical top", chunk_id="chk-lex")

        # Both at rank 1 in their lists
        list_a = [_make_candidate(c1, 0.9, RetrievalMethod.DENSE, "vec-store")]
        list_b = [_make_candidate(c2, 0.9, RetrievalMethod.LEXICAL, "lex-store")]

        fused = reciprocal_rank_fusion([list_a, list_b], weights=[2.0, 1.0])

        assert len(fused) == 2
        # c1 should have higher fused score due to 2x weight
        assert fused[0].chunk.chunk_id == "chk-dense"
        assert fused[0].fused_score > fused[1].fused_score

        expected_dense = 2.0 / 61
        expected_lex = 1.0 / 61
        assert abs(fused[0].fused_score - expected_dense) < 1e-9
        assert abs(fused[1].fused_score - expected_lex) < 1e-9

    def test_one_empty_list(self) -> None:
        """One empty list — other list's candidates survive."""
        c1 = _make_chunk(content="only candidate", chunk_id="chk-1")

        list_a = [_make_candidate(c1, 0.9, RetrievalMethod.DENSE, "vec-store")]
        list_b: list[RetrievalCandidate] = []

        fused = reciprocal_rank_fusion([list_a, list_b], weights=[1.0, 1.0])

        assert len(fused) == 1
        assert fused[0].chunk.chunk_id == "chk-1"
        assert fused[0].retrieval_method == RetrievalMethod.DENSE

    def test_both_empty_lists(self) -> None:
        """Both lists empty — returns empty."""
        fused = reciprocal_rank_fusion(
            [[], []], weights=[1.0, 1.0]
        )
        assert fused == []

    def test_k_parameter(self) -> None:
        """k=1 vs k=60 produces different score distributions."""
        c1 = _make_chunk(content="rank one", chunk_id="chk-1")
        c2 = _make_chunk(content="rank two", chunk_id="chk-2")

        candidates = [
            _make_candidate(c1, 0.9, RetrievalMethod.DENSE, "vec-store"),
            _make_candidate(c2, 0.7, RetrievalMethod.DENSE, "vec-store"),
        ]

        fused_k1 = reciprocal_rank_fusion([candidates], weights=[1.0], k=1)
        fused_k60 = reciprocal_rank_fusion([candidates], weights=[1.0], k=60)

        assert len(fused_k1) == 2
        assert len(fused_k60) == 2

        # k=1: rank1 score = 1/2 = 0.5, rank2 score = 1/3 ≈ 0.333
        # k=60: rank1 score = 1/61 ≈ 0.0164, rank2 score = 1/62 ≈ 0.0161
        # k=1 should have larger gap between rank 1 and rank 2
        gap_k1 = fused_k1[0].fused_score - fused_k1[1].fused_score
        gap_k60 = fused_k60[0].fused_score - fused_k60[1].fused_score
        assert gap_k1 > gap_k60

    def test_mismatched_lengths_raises(self) -> None:
        """ranked_lists and weights length mismatch raises ValueError."""
        c1 = _make_chunk(content="some chunk", chunk_id="chk-1")
        candidates = [_make_candidate(c1, 0.9, RetrievalMethod.DENSE, "vec-store")]

        with pytest.raises(ValueError, match="same length"):
            reciprocal_rank_fusion([candidates], weights=[1.0, 2.0])
