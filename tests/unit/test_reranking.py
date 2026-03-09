"""Tests for the reranking subsystem."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.common import RetrievalMethod
from libs.contracts.retrieval import RetrievalCandidate, RetrievalQuery
from libs.reranking.converters import (
    fused_list_to_retrieval_candidates,
    fused_to_retrieval_candidate,
)
from libs.reranking.cross_encoder_stub import CrossEncoderReranker
from libs.reranking.feature_reranker import FeatureBasedReranker
from libs.reranking.mock_reranker import PassthroughReranker
from libs.reranking.models import FeatureWeights, RerankerOutcome
from libs.reranking.protocols import Reranker
from libs.reranking.service import RerankerService
from libs.retrieval.broker.models import FusedCandidate

# ── Helpers ──────────────────────────────────────────────────────────


def _make_lineage() -> ChunkLineage:
    return ChunkLineage(
        source_id="src-1",
        document_id="doc-1",
        block_ids=["b1"],
        chunk_strategy="fixed",
        parser_version="1.0",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _make_chunk(
    chunk_id: str = "c1",
    content: str = "some chunk content",
    metadata: dict | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_id="src-1",
        block_ids=["b1"],
        content=content,
        content_hash="hash-" + chunk_id,
        token_count=10,
        strategy="fixed",
        byte_offset_start=0,
        byte_offset_end=100,
        lineage=_make_lineage(),
        metadata=metadata or {},
    )


def _make_candidate(
    chunk_id: str = "c1",
    score: float = 0.8,
    content: str = "some chunk content",
    metadata: dict | None = None,
    store_id: str = "store-1",
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=_make_chunk(chunk_id=chunk_id, content=content, metadata=metadata),
        score=score,
        retrieval_method=RetrievalMethod.DENSE,
        store_id=store_id,
    )


def _make_query(text: str = "machine learning") -> RetrievalQuery:
    return RetrievalQuery(
        raw_query=text,
        normalized_query=text,
        trace_id="trace-1",
    )


# ── Protocol conformance ────────────────────────────────────────────


class TestProtocolConformance:
    def test_passthrough_is_reranker(self) -> None:
        assert isinstance(PassthroughReranker(), Reranker)

    def test_feature_based_is_reranker(self) -> None:
        assert isinstance(FeatureBasedReranker(), Reranker)

    def test_cross_encoder_is_reranker(self) -> None:
        assert isinstance(CrossEncoderReranker("ms-marco"), Reranker)


# ── PassthroughReranker ─────────────────────────────────────────────


class TestPassthroughReranker:
    def test_preserves_score_order(self) -> None:
        reranker = PassthroughReranker()
        candidates = [
            _make_candidate("c1", 0.5),
            _make_candidate("c2", 0.9),
            _make_candidate("c3", 0.7),
        ]
        ranked = reranker.rerank(candidates, _make_query())
        scores = [r.rerank_score for r in ranked]
        assert scores == [0.9, 0.7, 0.5]

    def test_sequential_ranks(self) -> None:
        reranker = PassthroughReranker()
        candidates = [
            _make_candidate("c1", 0.5),
            _make_candidate("c2", 0.9),
        ]
        ranked = reranker.rerank(candidates, _make_query())
        ranks = [r.rank for r in ranked]
        assert ranks == [1, 2]

    def test_reranker_id(self) -> None:
        assert PassthroughReranker().reranker_id == "passthrough-v1"

    def test_empty_input(self) -> None:
        ranked = PassthroughReranker().rerank([], _make_query())
        assert ranked == []


# ── FeatureBasedReranker ─────────────────────────────────────────────


class TestFeatureBasedReranker:
    def test_authority_reorders(self) -> None:
        """Higher authority should boost a lower-scored candidate."""
        reranker = FeatureBasedReranker(
            weights=FeatureWeights(
                retrieval_score=0.1,
                lexical_overlap=0.0,
                source_authority=2.0,
                freshness=0.0,
                title_match=0.0,
                source_type=0.0,
            ),
        )
        candidates = [
            _make_candidate("c1", 0.9, metadata={"authority": 0.1}),
            _make_candidate("c2", 0.8, metadata={"authority": 1.0}),
        ]
        ranked = reranker.rerank(candidates, _make_query())
        assert ranked[0].candidate.chunk.chunk_id == "c2"

    def test_lexical_overlap_boost(self) -> None:
        """Candidate with query terms in content should rank higher."""
        reranker = FeatureBasedReranker(
            weights=FeatureWeights(
                retrieval_score=0.0,
                lexical_overlap=2.0,
                source_authority=0.0,
                freshness=0.0,
                title_match=0.0,
                source_type=0.0,
            ),
        )
        candidates = [
            _make_candidate("c1", 0.9, content="unrelated content here"),
            _make_candidate("c2", 0.8, content="machine learning is great"),
        ]
        ranked = reranker.rerank(candidates, _make_query("machine learning"))
        assert ranked[0].candidate.chunk.chunk_id == "c2"

    def test_title_match_boost(self) -> None:
        """Candidate with query as title substring should rank higher."""
        reranker = FeatureBasedReranker(
            weights=FeatureWeights(
                retrieval_score=0.0,
                lexical_overlap=0.0,
                source_authority=0.0,
                freshness=0.0,
                title_match=2.0,
                source_type=0.0,
            ),
        )
        candidates = [
            _make_candidate("c1", 0.9, metadata={"title": "unrelated"}),
            _make_candidate("c2", 0.8, metadata={"title": "intro to machine learning"}),
        ]
        ranked = reranker.rerank(candidates, _make_query("machine learning"))
        assert ranked[0].candidate.chunk.chunk_id == "c2"

    def test_freshness_boost(self) -> None:
        """More recent candidate should rank higher with freshness weight."""
        ref_time = datetime(2025, 6, 1, tzinfo=UTC)
        reranker = FeatureBasedReranker(
            weights=FeatureWeights(
                retrieval_score=0.0,
                lexical_overlap=0.0,
                source_authority=0.0,
                freshness=2.0,
                title_match=0.0,
                source_type=0.0,
            ),
            reference_time=ref_time,
        )
        candidates = [
            _make_candidate(
                "c1",
                0.9,
                metadata={"updated_at": ref_time - timedelta(days=730)},
            ),
            _make_candidate(
                "c2",
                0.8,
                metadata={"updated_at": ref_time - timedelta(days=1)},
            ),
        ]
        ranked = reranker.rerank(candidates, _make_query())
        assert ranked[0].candidate.chunk.chunk_id == "c2"

    def test_combined_signals_reorder_vs_passthrough(self) -> None:
        """Default weights should produce different order than passthrough for crafted input."""
        ref_time = datetime(2025, 6, 1, tzinfo=UTC)
        reranker = FeatureBasedReranker(reference_time=ref_time)
        passthrough = PassthroughReranker()

        # c1: high retrieval score but poor features
        # c2: lower retrieval score but great features
        candidates = [
            _make_candidate(
                "c1",
                0.95,
                content="unrelated stuff",
                metadata={"authority": 0.1, "updated_at": ref_time - timedelta(days=1000)},
            ),
            _make_candidate(
                "c2",
                0.90,
                content="machine learning overview",
                metadata={
                    "authority": 1.0,
                    "title": "machine learning guide",
                    "updated_at": ref_time - timedelta(days=1),
                },
            ),
        ]
        query = _make_query("machine learning")
        feature_ranked = reranker.rerank(candidates, query)
        pass_ranked = passthrough.rerank(candidates, query)

        feature_order = [r.candidate.chunk.chunk_id for r in feature_ranked]
        pass_order = [r.candidate.chunk.chunk_id for r in pass_ranked]
        assert feature_order != pass_order

    def test_score_breakdown_inspectability(self) -> None:
        """Skill: rag_trace_analysis — per-candidate feature breakdown for debugging."""
        reranker = FeatureBasedReranker()
        candidate = _make_candidate(
            "c1", 0.8, content="machine learning", metadata={"authority": 0.9},
        )
        query = _make_query("machine learning")
        breakdown = reranker.score_breakdown(candidate, query, min_score=0.5, score_range=0.5)
        assert "retrieval_score" in breakdown
        assert "retrieval_score_weighted" in breakdown
        assert "lexical_overlap" in breakdown
        assert "source_authority" in breakdown
        assert breakdown["lexical_overlap"] == 1.0  # all query terms present
        assert breakdown["source_authority"] == 0.9

    def test_custom_weights_retrieval_score_zero(self) -> None:
        """With retrieval_score=0, retrieval score should not affect ranking."""
        reranker = FeatureBasedReranker(
            weights=FeatureWeights(
                retrieval_score=0.0,
                lexical_overlap=0.0,
                source_authority=1.0,
                freshness=0.0,
                title_match=0.0,
                source_type=0.0,
            ),
        )
        # c1 has higher retrieval score but lower authority
        candidates = [
            _make_candidate("c1", 0.99, metadata={"authority": 0.1}),
            _make_candidate("c2", 0.01, metadata={"authority": 0.9}),
        ]
        ranked = reranker.rerank(candidates, _make_query())
        assert ranked[0].candidate.chunk.chunk_id == "c2"


# ── CrossEncoderReranker ─────────────────────────────────────────────


class TestCrossEncoderReranker:
    def test_raises_not_implemented(self) -> None:
        reranker = CrossEncoderReranker("ms-marco-MiniLM")
        with pytest.raises(NotImplementedError):
            reranker.rerank([_make_candidate()], _make_query())


# ── RerankerService ──────────────────────────────────────────────────


class TestRerankerService:
    def test_success_outcome(self) -> None:
        service = RerankerService(PassthroughReranker())
        result = service.run([_make_candidate()], _make_query())
        assert result.outcome == RerankerOutcome.SUCCESS
        assert result.candidate_count == 1

    def test_empty_no_candidates(self) -> None:
        service = RerankerService(PassthroughReranker())
        result = service.run([], _make_query())
        assert result.outcome == RerankerOutcome.NO_CANDIDATES
        assert result.candidate_count == 0

    def test_top_n_truncation(self) -> None:
        service = RerankerService(PassthroughReranker(), top_n=2)
        candidates = [
            _make_candidate("c1", 0.5),
            _make_candidate("c2", 0.9),
            _make_candidate("c3", 0.7),
        ]
        result = service.run(candidates, _make_query())
        assert result.candidate_count == 2
        assert result.ranked_candidates[0].candidate.chunk.chunk_id == "c2"

    def test_error_handling_failed(self) -> None:
        service = RerankerService(CrossEncoderReranker("fail-model"))
        result = service.run([_make_candidate()], _make_query())
        assert result.outcome == RerankerOutcome.FAILED
        assert len(result.errors) == 1
        assert result.candidate_count == 0

    def test_debug_payload_keys(self) -> None:
        service = RerankerService(PassthroughReranker(), top_n=5)
        result = service.run([_make_candidate()], _make_query())
        assert "reranker_id" in result.debug
        assert "top_n" in result.debug
        assert "input_count" in result.debug
        assert "output_count" in result.debug
        # Skill: rag_trace_analysis — trace context for diagnostics
        assert "trace_id" in result.debug
        assert result.debug["trace_id"] == "trace-1"
        # Skill: reranker_design — pre-rerank baseline for comparison
        assert "pre_rerank_score_min" in result.debug
        assert "pre_rerank_score_max" in result.debug

    def test_error_context_includes_trace_id(self) -> None:
        """Skill: failure_mode_analysis — errors must include identifying context."""
        service = RerankerService(CrossEncoderReranker("fail-model"))
        result = service.run([_make_candidate()], _make_query())
        assert "trace_id=trace-1" in result.errors[0]
        assert "reranker=cross-encoder:fail-model" in result.errors[0]

    def test_truncation_debug(self) -> None:
        """Skill: latency_budgeting — track how many candidates were truncated."""
        service = RerankerService(PassthroughReranker(), top_n=1)
        candidates = [_make_candidate("c1", 0.9), _make_candidate("c2", 0.5)]
        result = service.run(candidates, _make_query())
        assert result.debug["truncated_count"] == 1


# ── Converters ───────────────────────────────────────────────────────


class TestConverters:
    def test_single_conversion(self) -> None:
        fc = FusedCandidate(
            chunk=_make_chunk("c1"),
            fused_score=0.75,
            retrieval_method=RetrievalMethod.HYBRID,
            contributing_stores=["store-a", "store-b"],
            per_store_ranks={"store-a": 1, "store-b": 3},
            per_store_scores={"store-a": 0.9, "store-b": 0.6},
        )
        rc = fused_to_retrieval_candidate(fc)
        assert rc.score == 0.75
        assert rc.store_id == "store-a"
        assert rc.retrieval_method == RetrievalMethod.HYBRID

    def test_batch_conversion(self) -> None:
        fused = [
            FusedCandidate(
                chunk=_make_chunk(f"c{i}"),
                fused_score=0.5 + i * 0.1,
                retrieval_method=RetrievalMethod.DENSE,
                contributing_stores=[f"store-{i}"],
                per_store_ranks={f"store-{i}": 1},
                per_store_scores={f"store-{i}": 0.5 + i * 0.1},
            )
            for i in range(3)
        ]
        candidates = fused_list_to_retrieval_candidates(fused)
        assert len(candidates) == 3
        assert all(isinstance(c, RetrievalCandidate) for c in candidates)


# ── Integration ──────────────────────────────────────────────────────


class TestIntegration:
    def test_fused_to_rerank_pipeline(self) -> None:
        """FusedCandidate -> convert -> rerank -> verify order changes."""
        ref_time = datetime(2025, 6, 1, tzinfo=UTC)
        fused = [
            FusedCandidate(
                chunk=_make_chunk(
                    "c1",
                    content="unrelated content",
                    metadata={"authority": 0.1},
                ),
                fused_score=0.9,
                retrieval_method=RetrievalMethod.HYBRID,
                contributing_stores=["store-a"],
                per_store_ranks={"store-a": 1},
                per_store_scores={"store-a": 0.9},
            ),
            FusedCandidate(
                chunk=_make_chunk(
                    "c2",
                    content="machine learning tutorial",
                    metadata={
                        "authority": 1.0,
                        "title": "machine learning guide",
                        "updated_at": ref_time - timedelta(days=1),
                    },
                ),
                fused_score=0.7,
                retrieval_method=RetrievalMethod.HYBRID,
                contributing_stores=["store-b"],
                per_store_ranks={"store-b": 2},
                per_store_scores={"store-b": 0.7},
            ),
        ]

        candidates = fused_list_to_retrieval_candidates(fused)
        reranker = FeatureBasedReranker(reference_time=ref_time)
        ranked = reranker.rerank(candidates, _make_query("machine learning"))

        # c2 should be promoted despite lower fused_score due to better features
        assert ranked[0].candidate.chunk.chunk_id == "c2"
        assert ranked[0].rank == 1
        assert ranked[1].candidate.chunk.chunk_id == "c1"
        assert ranked[1].rank == 2
