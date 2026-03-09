"""Async retrieval broker: parallel dense + lexical fanout via asyncio.gather."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

from libs.contracts.common import RetrievalMethod
from libs.contracts.retrieval import RetrievalCandidate, RetrievalQuery
from libs.resilience import RetryConfig, is_transient_error
from libs.retrieval.broker.async_protocols import AsyncQueryEmbedder
from libs.retrieval.broker.dedup import apply_source_caps
from libs.retrieval.broker.fusion import reciprocal_rank_fusion
from libs.retrieval.broker.models import (
    BrokerConfig,
    BrokerOutcome,
    BrokerResult,
    ErrorClassification,
    RetrievalMode,
    StoreResult,
)
from libs.retrieval.broker.protocols import (
    PassthroughNormalizer,
    QueryNormalizer,
)
from libs.retrieval.stores.async_protocols import (
    AsyncLexicalStore,
    AsyncVectorStore,
)

logger = logging.getLogger(__name__)


def _classify_error(exc: Exception) -> ErrorClassification:
    if is_transient_error(exc):
        return ErrorClassification.TRANSIENT
    return ErrorClassification.PERMANENT


class AsyncRetrievalBroker:
    """Async broker that uses asyncio.gather for parallel fanout."""

    def __init__(
        self,
        vector_store: AsyncVectorStore,
        lexical_store: AsyncLexicalStore,
        query_embedder: AsyncQueryEmbedder,
        config: BrokerConfig | None = None,
        normalizer: QueryNormalizer | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._lexical_store = lexical_store
        self._query_embedder = query_embedder
        self._config = config or BrokerConfig()
        self._normalizer = normalizer or PassthroughNormalizer()
        self._retry_config = retry_config

    async def run(self, query: RetrievalQuery) -> BrokerResult:
        """Execute async retrieval: normalize → parallel fanout → fuse."""
        start = time.monotonic()
        errors: list[str] = []
        debug: dict[str, Any] = {
            "mode": self._config.mode.value,
            "rrf_k": self._config.rrf_k,
            "async": True,
        }

        # Normalize
        normalized = self._normalizer.normalize(query.raw_query)
        effective_query = RetrievalQuery(
            raw_query=query.raw_query,
            normalized_query=normalized,
            trace_id=query.trace_id,
            top_k=query.top_k,
            filters=query.filters,
            metadata=query.metadata,
        )

        use_dense = self._config.mode in (
            RetrievalMode.DENSE, RetrievalMode.HYBRID,
        )
        use_lexical = self._config.mode in (
            RetrievalMode.LEXICAL, RetrievalMode.HYBRID,
        )

        # Parallel fanout
        tasks: list[asyncio.Task[Any]] = []
        task_labels: list[str] = []
        if use_dense:
            tasks.append(
                asyncio.create_task(
                    self._fanout_dense(effective_query),
                ),
            )
            task_labels.append("dense")
        if use_lexical:
            tasks.append(
                asyncio.create_task(
                    self._fanout_lexical(effective_query),
                ),
            )
            task_labels.append("lexical")

        results = await asyncio.gather(*tasks)

        store_results: list[StoreResult] = []
        ranked_lists: list[list[RetrievalCandidate]] = []
        weights: list[float] = []
        query_vector: list[float] | None = None

        for label, result in zip(task_labels, results, strict=True):
            if label == "dense":
                dense_sr, qv = result
                query_vector = qv
                store_results.append(dense_sr)
                if dense_sr.error:
                    errors.append(dense_sr.error)
                else:
                    ranked_lists.append(dense_sr.candidates)
                    weights.append(self._config.dense_weight)
            else:
                lex_sr = result
                store_results.append(lex_sr)
                if lex_sr.error:
                    errors.append(lex_sr.error)
                else:
                    ranked_lists.append(lex_sr.candidates)
                    weights.append(self._config.lexical_weight)

        # Populate debug
        for sr in store_results:
            if sr.retrieval_method == RetrievalMethod.DENSE and not sr.error:
                debug["pre_fusion_dense_count"] = sr.candidate_count
            elif sr.retrieval_method == RetrievalMethod.LEXICAL and not sr.error:
                debug["pre_fusion_lexical_count"] = sr.candidate_count
        debug.setdefault("pre_fusion_dense_count", 0)
        debug.setdefault("pre_fusion_lexical_count", 0)

        if query_vector is not None:
            debug["query_vector"] = query_vector[:8]
            debug["query_vector_dimensions"] = len(query_vector)

        # Check total failure
        active_count = int(use_dense) + int(use_lexical)
        failed_count = sum(
            1 for sr in store_results if sr.error is not None
        )

        if failed_count == active_count:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("async_broker: all stores failed")
            return BrokerResult(
                query=effective_query,
                mode=self._config.mode,
                candidates=[],
                candidate_count=0,
                store_results=store_results,
                outcome=BrokerOutcome.FAILED,
                total_latency_ms=elapsed,
                completed_at=datetime.now(UTC),
                errors=errors,
                debug=debug,
            )

        # RRF fusion
        fused = (
            reciprocal_rank_fusion(
                ranked_lists, weights, k=self._config.rrf_k,
            )
            if ranked_lists
            else []
        )
        debug["post_fusion_count"] = len(fused)

        # Source caps
        if self._config.max_candidates_per_source > 0:
            pre_cap = len(fused)
            fused = apply_source_caps(
                fused, self._config.max_candidates_per_source,
            )
            debug["post_source_cap_count"] = len(fused)
            debug["source_cap_removals"] = pre_cap - len(fused)
        else:
            debug["post_source_cap_count"] = len(fused)
            debug["source_cap_removals"] = 0

        # Truncate
        fused = fused[: effective_query.top_k]

        # Outcome
        if not fused:
            outcome = BrokerOutcome.NO_RESULTS
        elif failed_count > 0:
            outcome = BrokerOutcome.PARTIAL
        else:
            outcome = BrokerOutcome.SUCCESS

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "async_broker: outcome=%s candidates=%d latency=%.1fms",
            outcome.value, len(fused), elapsed,
        )

        return BrokerResult(
            query=effective_query,
            mode=self._config.mode,
            candidates=fused,
            candidate_count=len(fused),
            store_results=store_results,
            outcome=outcome,
            total_latency_ms=elapsed,
            completed_at=datetime.now(UTC),
            errors=errors,
            debug=debug,
        )

    async def _fanout_dense(
        self, query: RetrievalQuery,
    ) -> tuple[StoreResult, list[float] | None]:
        t0 = time.monotonic()
        try:
            qv = await self._query_embedder.async_embed_query(
                query.normalized_query,
            )
            candidates = await self._vector_store.async_search(query, qv)
            elapsed = (time.monotonic() - t0) * 1000
            return StoreResult(
                store_id=self._vector_store.store_id,
                retrieval_method=RetrievalMethod.DENSE,
                candidates=candidates,
                candidate_count=len(candidates),
                latency_ms=elapsed,
            ), qv
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            classification = _classify_error(exc)
            logger.error(
                "async_broker: dense exception: %s", exc,
            )
            return StoreResult(
                store_id=self._vector_store.store_id,
                retrieval_method=RetrievalMethod.DENSE,
                candidates=[],
                candidate_count=0,
                latency_ms=elapsed,
                error=(
                    f"store={self._vector_store.store_id}"
                    f" trace_id={query.trace_id}: {exc}"
                ),
                error_classification=classification,
            ), None

    async def _fanout_lexical(
        self, query: RetrievalQuery,
    ) -> StoreResult:
        t0 = time.monotonic()
        try:
            candidates = await self._lexical_store.async_search(query)
            elapsed = (time.monotonic() - t0) * 1000
            return StoreResult(
                store_id=self._lexical_store.store_id,
                retrieval_method=RetrievalMethod.LEXICAL,
                candidates=candidates,
                candidate_count=len(candidates),
                latency_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            classification = _classify_error(exc)
            logger.error(
                "async_broker: lexical exception: %s", exc,
            )
            return StoreResult(
                store_id=self._lexical_store.store_id,
                retrieval_method=RetrievalMethod.LEXICAL,
                candidates=[],
                candidate_count=0,
                latency_ms=elapsed,
                error=(
                    f"store={self._lexical_store.store_id}"
                    f" trace_id={query.trace_id}: {exc}"
                ),
                error_classification=classification,
            )
