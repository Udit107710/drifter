"""RerankerService: orchestrates reranking with timing, error handling, and truncation."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from libs.contracts.retrieval import RetrievalCandidate, RetrievalQuery
from libs.reranking.models import RerankerOutcome, RerankerResult
from libs.reranking.protocols import Reranker

logger = logging.getLogger(__name__)


class RerankerService:
    """Orchestrator that wraps a Reranker with timing, error handling, and top-n truncation."""

    def __init__(self, reranker: Reranker, top_n: int = 0) -> None:
        self._reranker = reranker
        self._top_n = top_n

    def run(
        self,
        candidates: list[RetrievalCandidate],
        query: RetrievalQuery,
    ) -> RerankerResult:
        """Execute reranking: empty check -> rerank -> truncate -> build result."""
        start = time.monotonic()
        errors: list[str] = []
        debug: dict[str, Any] = {
            "reranker_id": self._reranker.reranker_id,
            "top_n": self._top_n,
            "input_count": len(candidates),
        }

        # Trace context for diagnostics (failure_mode_analysis skill)
        debug["trace_id"] = query.trace_id

        logger.debug(
            "reranker: entry reranker_id=%s input_count=%d",
            self._reranker.reranker_id, len(candidates),
        )

        # Empty check
        if not candidates:
            elapsed = (time.monotonic() - start) * 1000
            logger.warning(
                "reranker: no candidates provided reranker_id=%s",
                self._reranker.reranker_id,
            )
            return RerankerResult(
                query=query,
                ranked_candidates=[],
                candidate_count=0,
                outcome=RerankerOutcome.NO_CANDIDATES,
                reranker_id=self._reranker.reranker_id,
                total_latency_ms=elapsed,
                completed_at=datetime.now(UTC),
                errors=errors,
                debug=debug,
            )

        # Capture pre-rerank score range for baseline comparison (reranker_design skill)
        pre_rerank_scores = [c.score for c in candidates]
        debug["pre_rerank_score_min"] = min(pre_rerank_scores)
        debug["pre_rerank_score_max"] = max(pre_rerank_scores)

        # Rerank (catch exceptions with context — failure_mode_analysis skill)
        try:
            ranked = self._reranker.rerank(candidates, query)
        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.error(
                "reranker: exception reranker_id=%s input_count=%d: %s",
                self._reranker.reranker_id, len(candidates), exc,
            )
            errors.append(
                f"reranker={self._reranker.reranker_id} "
                f"trace_id={query.trace_id} "
                f"input_count={len(candidates)}: {exc}"
            )
            return RerankerResult(
                query=query,
                ranked_candidates=[],
                candidate_count=0,
                outcome=RerankerOutcome.FAILED,
                reranker_id=self._reranker.reranker_id,
                total_latency_ms=elapsed,
                completed_at=datetime.now(UTC),
                errors=errors,
                debug=debug,
            )

        # Truncate to top_n if configured
        if self._top_n > 0:
            pre_truncation = len(ranked)
            ranked = ranked[: self._top_n]
            debug["truncated_count"] = pre_truncation - len(ranked)

        debug["output_count"] = len(ranked)
        elapsed = (time.monotonic() - start) * 1000

        logger.info(
            "reranker: success reranker_id=%s output_count=%d latency=%.1fms",
            self._reranker.reranker_id, len(ranked), elapsed,
        )

        return RerankerResult(
            query=query,
            ranked_candidates=ranked,
            candidate_count=len(ranked),
            outcome=RerankerOutcome.SUCCESS,
            reranker_id=self._reranker.reranker_id,
            total_latency_ms=elapsed,
            completed_at=datetime.now(UTC),
            errors=errors,
            debug=debug,
        )
