"""DiversityAwareBuilder: MMR-style selection balancing relevance and source diversity."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from libs.chunking.protocols import TokenCounter
from libs.context_builder.dedup import deduplicate
from libs.context_builder.models import (
    BuilderConfig,
    BuilderOutcome,
    BuilderResult,
    ExclusionRecord,
)
from libs.contracts.common import SelectionReason
from libs.contracts.context import ContextItem, ContextPack
from libs.contracts.retrieval import RankedCandidate


class DiversityAwareBuilder:
    """Selects chunks using MMR-style scoring that balances rerank score with source diversity.

    At each step, candidates are scored:
        mmr = (1 - diversity_weight) * normalized_relevance + diversity_weight * novelty
    where novelty = 1.0 if the source is not yet represented, else 0.0.
    """

    def __init__(
        self,
        token_counter: TokenCounter,
        config: BuilderConfig | None = None,
    ) -> None:
        self._token_counter = token_counter
        self._config = config or BuilderConfig()

    def build(
        self,
        candidates: list[RankedCandidate],
        query: str,
        token_budget: int,
    ) -> BuilderResult:
        start = time.monotonic()
        exclusions: list[ExclusionRecord] = []
        debug: dict[str, Any] = {
            "builder_type": "diversity_aware",
            "diversity_weight": self._config.diversity_weight,
            "input_count": len(candidates),
            "token_budget": token_budget,
        }

        if not candidates:
            return self._empty_result(
                query, token_budget,
                BuilderOutcome.EMPTY_CANDIDATES, start, debug,
            )

        # Dedup
        dedup_removed = 0
        if self._config.deduplicate:
            kept, dedup_excl = deduplicate(candidates)
            dedup_removed = len(dedup_excl)
            exclusions.extend(dedup_excl)
        else:
            kept = list(candidates)

        debug["post_dedup_count"] = len(kept)

        if not kept:
            return self._empty_result(
                query, token_budget,
                BuilderOutcome.EMPTY_CANDIDATES, start, debug,
                exclusions=exclusions,
                dedup_removed=dedup_removed,
                input_count=len(candidates),
            )

        # Precompute max rerank_score for normalization
        max_score = max(rc.rerank_score for rc in kept)

        # MMR-style selection
        evidence: list[ContextItem] = []
        selected_sources: set[str] = set()
        remaining = list(kept)
        running_total = 0
        w = self._config.diversity_weight

        max_ch = self._config.max_chunks
        while remaining and (max_ch == 0 or len(evidence) < max_ch):
            best_idx = -1
            best_mmr = -1.0
            best_is_diversity_pick = False

            # Also track pure-relevance best for selection_reason attribution
            best_relevance_idx = -1
            best_relevance_score = -1.0

            for i, rc in enumerate(remaining):
                relevance = rc.rerank_score / max_score if max_score > 0 else 1.0
                novelty = 0.0 if rc.candidate.chunk.source_id in selected_sources else 1.0
                mmr = (1.0 - w) * relevance + w * novelty

                if relevance > best_relevance_score:
                    best_relevance_score = relevance
                    best_relevance_idx = i

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
                    best_is_diversity_pick = (i != best_relevance_idx)

            if best_idx < 0:
                break

            rc = remaining.pop(best_idx)
            # Recompute best_relevance_idx after pop (it might have shifted)
            # Actually, we determine diversity_pick BEFORE pop, so the flag is correct.

            chunk = rc.candidate.chunk
            tokens = self._token_counter.count(chunk.content)

            if tokens == 0:
                exclusions.append(ExclusionRecord(
                    chunk_id=chunk.chunk_id,
                    reason="zero_tokens",
                    token_count=0,
                ))
                continue

            if running_total + tokens > token_budget:
                exclusions.append(
                    ExclusionRecord(
                        chunk_id=chunk.chunk_id,
                        reason=(
                            f"budget_exceeded:needed={tokens},"
                            f"remaining={token_budget - running_total}"
                        ),
                        token_count=tokens,
                    )
                )
                continue

            if best_is_diversity_pick:
                reason = SelectionReason.DIVERSITY
            else:
                reason = SelectionReason.TOP_RANKED
            evidence.append(
                ContextItem(
                    chunk=chunk,
                    rank=len(evidence) + 1,
                    token_count=tokens,
                    selection_reason=reason,
                )
            )
            running_total += tokens
            selected_sources.add(chunk.source_id)

        if not evidence and kept:
            outcome = BuilderOutcome.BUDGET_EXHAUSTED
        elif evidence:
            outcome = BuilderOutcome.SUCCESS
        else:
            outcome = BuilderOutcome.EMPTY_CANDIDATES

        diversity_score = min(len(selected_sources) / len(evidence), 1.0) if evidence else 0.0

        debug.update({
            "selected_count": len(evidence),
            "excluded_count": len(exclusions),
            "tokens_used": running_total,
            "tokens_remaining": token_budget - running_total,
            "unique_sources": len(selected_sources),
            "selection_reasons": {
                "top_ranked": sum(
                    1 for e in evidence
                    if e.selection_reason == SelectionReason.TOP_RANKED
                ),
                "diversity": sum(
                    1 for e in evidence
                    if e.selection_reason == SelectionReason.DIVERSITY
                ),
            },
        })

        pack = ContextPack(
            query=query,
            evidence=evidence,
            total_tokens=running_total,
            token_budget=token_budget,
            diversity_score=diversity_score,
        )

        elapsed = (time.monotonic() - start) * 1000
        return BuilderResult(
            context_pack=pack,
            outcome=outcome,
            exclusions=exclusions,
            input_count=len(candidates),
            dedup_removed=dedup_removed,
            total_latency_ms=elapsed,
            completed_at=datetime.now(UTC),
            debug=debug,
        )

    def _empty_result(
        self,
        query: str,
        token_budget: int,
        outcome: BuilderOutcome,
        start: float,
        debug: dict[str, Any],
        exclusions: list[ExclusionRecord] | None = None,
        dedup_removed: int = 0,
        input_count: int = 0,
    ) -> BuilderResult:
        pack = ContextPack(
            query=query, evidence=[], total_tokens=0,
            token_budget=token_budget, diversity_score=0.0,
        )
        elapsed = (time.monotonic() - start) * 1000
        debug.update({
            "selected_count": 0,
            "excluded_count": 0,
            "tokens_used": 0,
            "tokens_remaining": token_budget,
            "unique_sources": 0,
        })
        return BuilderResult(
            context_pack=pack, outcome=outcome,
            exclusions=exclusions or [], input_count=input_count,
            dedup_removed=dedup_removed, total_latency_ms=elapsed,
            completed_at=datetime.now(UTC), debug=debug,
        )
