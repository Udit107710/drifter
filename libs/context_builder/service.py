"""ContextBuilderService: orchestrates context building with validation and error handling."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from libs.context_builder.models import BuilderOutcome, BuilderResult
from libs.context_builder.protocols import ContextBuilder
from libs.contracts.context import ContextPack
from libs.contracts.retrieval import RankedCandidate

logger = logging.getLogger(__name__)


class ContextBuilderService:
    """Orchestrator that wraps a ContextBuilder with validation, error handling, and timing."""

    def __init__(self, builder: ContextBuilder) -> None:
        self._builder = builder

    def run(
        self,
        candidates: list[RankedCandidate],
        query: str,
        token_budget: int,
    ) -> BuilderResult:
        start = time.monotonic()
        debug: dict[str, Any] = {
            "input_count": len(candidates),
            "token_budget": token_budget,
        }

        logger.debug(
            "context_builder: entry input_count=%d budget=%d",
            len(candidates), token_budget,
        )

        # Validate inputs
        if not query:
            logger.warning("context_builder: empty query")
            return self._failed(
                "query must not be empty",
                start, debug, query or "unknown", token_budget,
            )
        if token_budget < 1:
            logger.warning("context_builder: invalid token_budget=%d", token_budget)
            return self._failed(
                f"token_budget must be >= 1, got {token_budget}",
                start, debug, query, max(token_budget, 1),
            )

        try:
            result = self._builder.build(candidates, query, token_budget)
        except Exception as exc:
            logger.error("context_builder: builder exception: %s", exc)
            return self._failed(
                f"builder failed: input_count={len(candidates)} token_budget={token_budget}: {exc}",
                start, debug, query, token_budget,
            )

        # Augment debug with service-level timing
        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "context_builder: success tokens_used=%d evidence_count=%d latency=%.1fms",
            result.context_pack.total_tokens, len(result.context_pack.evidence), elapsed,
        )
        result_debug = dict(result.debug)
        result_debug["service_latency_ms"] = elapsed
        # Return a new result with updated debug and timing
        return BuilderResult(
            context_pack=result.context_pack,
            outcome=result.outcome,
            exclusions=result.exclusions,
            input_count=result.input_count,
            dedup_removed=result.dedup_removed,
            total_latency_ms=elapsed,
            completed_at=result.completed_at,
            errors=result.errors,
            debug=result_debug,
        )

    def _failed(
        self,
        error: str,
        start: float,
        debug: dict[str, Any],
        query: str,
        token_budget: int,
    ) -> BuilderResult:
        elapsed = (time.monotonic() - start) * 1000
        pack = ContextPack(
            query=query, evidence=[], total_tokens=0,
            token_budget=token_budget, diversity_score=0.0,
        )
        return BuilderResult(
            context_pack=pack,
            outcome=BuilderOutcome.FAILED,
            exclusions=[],
            input_count=0,
            dedup_removed=0,
            total_latency_ms=elapsed,
            completed_at=datetime.now(UTC),
            errors=[error],
            debug=debug,
        )
