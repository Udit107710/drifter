"""Tests for retrieval contracts: RetrievalQuery, RetrievalCandidate, RankedCandidate."""

from datetime import UTC, datetime

import pytest

from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.common import RetrievalMethod
from libs.contracts.retrieval import RankedCandidate, RetrievalCandidate, RetrievalQuery


def _make_lineage() -> ChunkLineage:
    return ChunkLineage(
        source_id="src-001",
        document_id="doc-001",
        block_ids=["blk-001"],
        chunk_strategy="fixed_size",
        parser_version="1.0.0",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _make_chunk() -> Chunk:
    return Chunk(
        chunk_id="chk-001",
        document_id="doc-001",
        source_id="src-001",
        block_ids=["blk-001"],
        content="Test chunk content.",
        content_hash="sha256:abc123",
        token_count=4,
        strategy="fixed_size",
        byte_offset_start=0,
        byte_offset_end=19,
        lineage=_make_lineage(),
    )


def _make_candidate() -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=_make_chunk(),
        score=0.85,
        retrieval_method=RetrievalMethod.DENSE,
        store_id="qdrant-main",
    )


# ── RetrievalQuery ──────────────────────────────────────────────────


class TestRetrievalQuery:
    def test_create_valid(self) -> None:
        q = RetrievalQuery(
            raw_query="What is RAG?",
            normalized_query="what is rag",
            trace_id="trace-001",
        )
        assert q.top_k == 50
        assert q.filters == {}

    def test_custom_top_k(self) -> None:
        q = RetrievalQuery(
            raw_query="query",
            normalized_query="query",
            trace_id="t-001",
            top_k=10,
        )
        assert q.top_k == 10

    def test_empty_raw_query_raises(self) -> None:
        with pytest.raises(ValueError, match="raw_query"):
            RetrievalQuery(raw_query="", normalized_query="q", trace_id="t")

    def test_empty_normalized_query_raises(self) -> None:
        with pytest.raises(ValueError, match="normalized_query"):
            RetrievalQuery(raw_query="q", normalized_query="", trace_id="t")

    def test_empty_trace_id_raises(self) -> None:
        with pytest.raises(ValueError, match="trace_id"):
            RetrievalQuery(raw_query="q", normalized_query="q", trace_id="")

    def test_zero_top_k_raises(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            RetrievalQuery(raw_query="q", normalized_query="q", trace_id="t", top_k=0)


# ── RetrievalCandidate ──────────────────────────────────────────────


class TestRetrievalCandidate:
    def test_create_valid(self) -> None:
        c = _make_candidate()
        assert c.score == 0.85
        assert c.retrieval_method == RetrievalMethod.DENSE

    def test_empty_store_id_raises(self) -> None:
        with pytest.raises(ValueError, match="store_id"):
            RetrievalCandidate(
                chunk=_make_chunk(),
                score=0.5,
                retrieval_method=RetrievalMethod.LEXICAL,
                store_id="",
            )


# ── RankedCandidate ─────────────────────────────────────────────────


class TestRankedCandidate:
    def test_create_valid(self) -> None:
        rc = RankedCandidate(
            candidate=_make_candidate(),
            rank=1,
            rerank_score=0.92,
            reranker_id="cross-encoder-v1",
        )
        assert rc.rank == 1
        assert rc.rerank_score == 0.92

    def test_zero_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            RankedCandidate(
                candidate=_make_candidate(),
                rank=0,
                rerank_score=0.5,
                reranker_id="reranker",
            )

    def test_empty_reranker_id_raises(self) -> None:
        with pytest.raises(ValueError, match="reranker_id"):
            RankedCandidate(
                candidate=_make_candidate(),
                rank=1,
                rerank_score=0.5,
                reranker_id="",
            )
