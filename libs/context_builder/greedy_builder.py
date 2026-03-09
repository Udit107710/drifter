"""GreedyContextBuilder: rank-order packing under token budget."""

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


class GreedyContextBuilder:
    """Packs chunks in rank order until the token budget is exhausted.

    Higher-ranked candidates are always preferred. Deduplication by
    content_hash is applied first if configured.
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
            "builder_type": "greedy",
            "input_count": len(candidates),
            "token_budget": token_budget,
        }

        if not candidates:
            return self._empty_result(
                query, token_budget, BuilderOutcome.EMPTY_CANDIDATES,
                start, exclusions, debug, dedup_removed=0,
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

        # Greedy packing
        evidence: list[ContextItem] = []
        running_total = 0
        source_ids: set[str] = set()
        source_counts: dict[str, int] = {}
        max_per_source = self._config.max_chunks_per_source

        for rc in kept:
            chunk = rc.candidate.chunk
            tokens = self._token_counter.count(chunk.content)

            if tokens == 0:
                exclusions.append(
                    ExclusionRecord(chunk_id=chunk.chunk_id, reason="zero_tokens", token_count=0)
                )
                continue

            # Source diversity cap
            if max_per_source > 0:
                count = source_counts.get(chunk.source_id, 0)
                if count >= max_per_source:
                    exclusions.append(
                        ExclusionRecord(
                            chunk_id=chunk.chunk_id,
                            reason=f"source_cap:source={chunk.source_id},limit={max_per_source}",
                            token_count=tokens,
                        )
                    )
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

            evidence.append(
                ContextItem(
                    chunk=chunk,
                    rank=len(evidence) + 1,
                    token_count=tokens,
                    selection_reason=SelectionReason.TOP_RANKED,
                )
            )
            running_total += tokens
            source_ids.add(chunk.source_id)
            source_counts[chunk.source_id] = source_counts.get(chunk.source_id, 0) + 1

            if 0 < self._config.max_chunks <= len(evidence):
                break

        if not evidence and kept:
            outcome = BuilderOutcome.BUDGET_EXHAUSTED
        elif evidence:
            outcome = BuilderOutcome.SUCCESS
        else:
            outcome = BuilderOutcome.EMPTY_CANDIDATES

        diversity_score = len(source_ids) / len(evidence) if evidence else 0.0
        # Clamp to [0, 1]
        diversity_score = min(diversity_score, 1.0)

        debug.update({
            "selected_count": len(evidence),
            "excluded_count": len(exclusions),
            "tokens_used": running_total,
            "tokens_remaining": token_budget - running_total,
            "unique_sources": len(source_ids),
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
        exclusions: list[ExclusionRecord],
        debug: dict[str, Any],
        dedup_removed: int,
    ) -> BuilderResult:
        pack = ContextPack(
            query=query,
            evidence=[],
            total_tokens=0,
            token_budget=token_budget,
            diversity_score=0.0,
        )
        elapsed = (time.monotonic() - start) * 1000
        debug.update({
            "selected_count": 0,
            "excluded_count": len(exclusions),
            "tokens_used": 0,
            "tokens_remaining": token_budget,
            "unique_sources": 0,
        })
        return BuilderResult(
            context_pack=pack,
            outcome=outcome,
            exclusions=exclusions,
            input_count=0,
            dedup_removed=dedup_removed,
            total_latency_ms=elapsed,
            completed_at=datetime.now(UTC),
            debug=debug,
        )
