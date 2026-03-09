"""Context builder subsystem.

Responsibilities:
- Allocate token budget for evidence
- Select top evidence from RankedCandidates
- Remove redundant passages
- Optimize for source diversity
- Produce a ContextPack within token limits

Boundary: consumes list[RankedCandidate], produces ContextPack.
"""

from libs.context_builder.dedup import deduplicate
from libs.context_builder.diversity_builder import DiversityAwareBuilder
from libs.context_builder.greedy_builder import GreedyContextBuilder
from libs.context_builder.models import (
    BuilderConfig,
    BuilderOutcome,
    BuilderResult,
    ExclusionRecord,
)
from libs.context_builder.protocols import ContextBuilder
from libs.context_builder.service import ContextBuilderService

__all__ = [
    "BuilderConfig",
    "BuilderOutcome",
    "BuilderResult",
    "ContextBuilder",
    "ContextBuilderService",
    "DiversityAwareBuilder",
    "ExclusionRecord",
    "GreedyContextBuilder",
    "deduplicate",
]
