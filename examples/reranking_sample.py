"""Example: reranking pipeline from FusedCandidates to RankedCandidates."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.common import RetrievalMethod
from libs.reranking import (
    FeatureBasedReranker,
    FeatureWeights,
    PassthroughReranker,
    RerankerService,
    fused_list_to_retrieval_candidates,
)
from libs.contracts.retrieval import RetrievalQuery
from libs.retrieval.broker.models import FusedCandidate


def _make_lineage() -> ChunkLineage:
    return ChunkLineage(
        source_id="src-1",
        document_id="doc-1",
        block_ids=["b1"],
        chunk_strategy="fixed",
        parser_version="1.0",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _make_chunk(chunk_id: str, content: str, metadata: dict | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_id="src-1",
        block_ids=["b1"],
        content=content,
        content_hash=f"hash-{chunk_id}",
        token_count=10,
        strategy="fixed",
        byte_offset_start=0,
        byte_offset_end=100,
        lineage=_make_lineage(),
        metadata=metadata or {},
    )


def main() -> None:
    ref_time = datetime(2025, 6, 1, tzinfo=UTC)
    query = RetrievalQuery(
        raw_query="machine learning",
        normalized_query="machine learning",
        trace_id="trace-demo",
    )

    # Simulate broker output
    fused_candidates = [
        FusedCandidate(
            chunk=_make_chunk("c1", "Introduction to deep learning basics"),
            fused_score=0.92,
            retrieval_method=RetrievalMethod.HYBRID,
            contributing_stores=["qdrant-1", "opensearch-1"],
            per_store_ranks={"qdrant-1": 1, "opensearch-1": 3},
            per_store_scores={"qdrant-1": 0.95, "opensearch-1": 0.7},
        ),
        FusedCandidate(
            chunk=_make_chunk(
                "c2",
                "Machine learning algorithms and applications",
                metadata={
                    "authority": 0.95,
                    "title": "Machine Learning Handbook",
                    "updated_at": ref_time - timedelta(days=30),
                    "source_type_score": 0.9,
                },
            ),
            fused_score=0.85,
            retrieval_method=RetrievalMethod.HYBRID,
            contributing_stores=["qdrant-1"],
            per_store_ranks={"qdrant-1": 2},
            per_store_scores={"qdrant-1": 0.85},
        ),
        FusedCandidate(
            chunk=_make_chunk(
                "c3",
                "Statistical methods overview",
                metadata={"authority": 0.3},
            ),
            fused_score=0.78,
            retrieval_method=RetrievalMethod.DENSE,
            contributing_stores=["qdrant-1"],
            per_store_ranks={"qdrant-1": 5},
            per_store_scores={"qdrant-1": 0.78},
        ),
    ]

    # Convert to RetrievalCandidates
    candidates = fused_list_to_retrieval_candidates(fused_candidates)

    # 1. Passthrough reranker (baseline)
    print("=== Passthrough Reranker ===")
    passthrough_service = RerankerService(PassthroughReranker(), top_n=3)
    result = passthrough_service.run(candidates, query)
    print(f"Outcome: {result.outcome.value}")
    for rc in result.ranked_candidates:
        print(f"  Rank {rc.rank}: {rc.candidate.chunk.chunk_id} (score={rc.rerank_score:.4f})")

    # 2. Feature-based reranker
    print("\n=== Feature-Based Reranker ===")
    feature_service = RerankerService(
        FeatureBasedReranker(reference_time=ref_time),
        top_n=3,
    )
    result = feature_service.run(candidates, query)
    print(f"Outcome: {result.outcome.value}")
    for rc in result.ranked_candidates:
        print(f"  Rank {rc.rank}: {rc.candidate.chunk.chunk_id} (score={rc.rerank_score:.4f})")

    # 3. Custom weights emphasizing authority
    print("\n=== Custom Weights (authority-heavy) ===")
    custom_weights = FeatureWeights(
        retrieval_score=0.5,
        lexical_overlap=0.1,
        source_authority=3.0,
        freshness=0.0,
        title_match=0.0,
        source_type=0.0,
    )
    custom_service = RerankerService(
        FeatureBasedReranker(weights=custom_weights, reference_time=ref_time),
        top_n=3,
    )
    result = custom_service.run(candidates, query)
    print(f"Outcome: {result.outcome.value}")
    for rc in result.ranked_candidates:
        print(f"  Rank {rc.rank}: {rc.candidate.chunk.chunk_id} (score={rc.rerank_score:.4f})")

    print(f"\nLatency: {result.total_latency_ms:.2f}ms")
    print(f"Debug: {result.debug}")


if __name__ == "__main__":
    main()
