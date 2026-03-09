"""PassthroughReranker: deterministic score-order reranker for testing."""

from __future__ import annotations

from libs.contracts.retrieval import RankedCandidate, RetrievalCandidate, RetrievalQuery


class PassthroughReranker:
    """Sorts candidates by descending retrieval score, assigns sequential ranks.

    No model inference — purely deterministic. Useful for testing and baselines.
    """

    @property
    def reranker_id(self) -> str:
        return "passthrough-v1"

    def rerank(
        self,
        candidates: list[RetrievalCandidate],
        query: RetrievalQuery,
    ) -> list[RankedCandidate]:
        sorted_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        return [
            RankedCandidate(
                candidate=c,
                rank=i + 1,
                rerank_score=c.score,
                reranker_id=self.reranker_id,
            )
            for i, c in enumerate(sorted_candidates)
        ]
