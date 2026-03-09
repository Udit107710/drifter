"""Retrieval broker — orchestrates dense, lexical, and hybrid retrieval."""

from libs.retrieval.broker.dedup import apply_source_caps
from libs.retrieval.broker.fusion import reciprocal_rank_fusion
from libs.retrieval.broker.models import (
    BrokerConfig,
    BrokerOutcome,
    BrokerResult,
    ErrorClassification,
    FusedCandidate,
    RetrievalMode,
    StoreResult,
)
from libs.retrieval.broker.protocols import (
    PassthroughNormalizer,
    QueryEmbedder,
    QueryNormalizer,
)
from libs.retrieval.broker.service import RetrievalBroker

__all__ = [
    "BrokerConfig",
    "BrokerOutcome",
    "BrokerResult",
    "ErrorClassification",
    "FusedCandidate",
    "PassthroughNormalizer",
    "QueryEmbedder",
    "QueryNormalizer",
    "RetrievalBroker",
    "RetrievalMode",
    "StoreResult",
    "apply_source_caps",
    "reciprocal_rank_fusion",
]
