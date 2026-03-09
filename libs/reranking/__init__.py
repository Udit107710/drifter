"""Reranking subsystem.

Responsibilities:
- Score and reorder RetrievalCandidates using cross-encoder or feature-based models
- Produce RankedCandidates with rerank scores

Boundary: consumes list[RetrievalCandidate], produces list[RankedCandidate].
Orders candidates, not prompts.
"""

from libs.reranking.converters import (
    fused_list_to_retrieval_candidates,
    fused_to_retrieval_candidate,
)
from libs.reranking.cross_encoder_stub import CrossEncoderReranker
from libs.reranking.feature_reranker import FeatureBasedReranker
from libs.reranking.mock_reranker import PassthroughReranker
from libs.reranking.models import FeatureWeights, RerankerOutcome, RerankerResult
from libs.reranking.protocols import Reranker
from libs.reranking.service import RerankerService

__all__ = [
    "CrossEncoderReranker",
    "FeatureBasedReranker",
    "FeatureWeights",
    "PassthroughReranker",
    "Reranker",
    "RerankerOutcome",
    "RerankerResult",
    "RerankerService",
    "fused_list_to_retrieval_candidates",
    "fused_to_retrieval_candidate",
]
