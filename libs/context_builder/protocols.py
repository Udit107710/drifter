"""Context builder protocol: the contract every builder must satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from libs.contracts.retrieval import RankedCandidate

if TYPE_CHECKING:
    from libs.context_builder.models import BuilderResult


@runtime_checkable
class ContextBuilder(Protocol):
    """Protocol for building context packs from ranked candidates."""

    def build(
        self,
        candidates: list[RankedCandidate],
        query: str,
        token_budget: int,
    ) -> BuilderResult: ...
