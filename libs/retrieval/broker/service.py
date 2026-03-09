"""Retrieval broker: orchestrates dense, lexical, and hybrid retrieval."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any

from libs.contracts.common import RetrievalMethod
from libs.contracts.retrieval import RetrievalCandidate, RetrievalQuery
from libs.resilience import is_transient_error
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
    QueryEmbedder,
    QueryNormalizer,
)
from libs.retrieval.stores.protocols import LexicalStore, VectorStore

logger = logging.getLogger(__name__)


def _classify_error(exc: Exception) -> ErrorClassification:
    """Classify an exception as transient (retryable) or permanent."""
    if is_transient_error(exc):
        return ErrorClassification.TRANSIENT
    return ErrorClassification.PERMANENT


class RetrievalBroker:
    """Orchestrates retrieval fanout, RRF fusion, and candidate pool assembly."""

    def __init__(
        self,
        vector_store: VectorStore,
        lexical_store: LexicalStore,
        query_embedder: QueryEmbedder,
        config: BrokerConfig | None = None,
        normalizer: QueryNormalizer | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._lexical_store = lexical_store
        self._query_embedder = query_embedder
        self._config = config or BrokerConfig()
        self._normalizer = normalizer or PassthroughNormalizer()

    def run(self, query: RetrievalQuery) -> BrokerResult:
        """Execute a retrieval run: normalize → fanout → fuse → cap → return."""
        start = time.monotonic()
        errors: list[str] = []
        store_results: list[StoreResult] = []
        debug: dict[str, Any] = {"mode": self._config.mode.value, "rrf_k": self._config.rrf_k}

        # Step 1: Normalize query
        logger.debug("broker: mode=%s query=%r", self._config.mode.value, query.raw_query[:100])
        normalized = self._normalizer.normalize(query.raw_query)
        effective_query = RetrievalQuery(
            raw_query=query.raw_query,
            normalized_query=normalized,
            trace_id=query.trace_id,
            top_k=query.top_k,
            filters=query.filters,
            metadata=query.metadata,
        )

        # Step 2: Determine active stores
        use_dense = self._config.mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID)
        use_lexical = self._config.mode in (RetrievalMode.LEXICAL, RetrievalMode.HYBRID)

        ranked_lists: list[list[RetrievalCandidate]] = []
        weights: list[float] = []

        # Step 3: Dense fanout
        query_vector: list[float] | None = None
        if use_dense:
            dense_result, query_vector = self._fanout_dense(effective_query)
            store_results.append(dense_result)
            if dense_result.error:
                errors.append(dense_result.error)
            else:
                ranked_lists.append(dense_result.candidates)
                weights.append(self._config.dense_weight)

        # Step 4: Lexical fanout
        if use_lexical:
            lexical_result = self._fanout_lexical(effective_query)
            store_results.append(lexical_result)
            if lexical_result.error:
                errors.append(lexical_result.error)
            else:
                ranked_lists.append(lexical_result.candidates)
                weights.append(self._config.lexical_weight)

        # Populate debug
        dense_count = 0
        lexical_count = 0
        for sr in store_results:
            if sr.retrieval_method == RetrievalMethod.DENSE and sr.error is None:
                dense_count = sr.candidate_count
            elif sr.retrieval_method == RetrievalMethod.LEXICAL and sr.error is None:
                lexical_count = sr.candidate_count
        debug["pre_fusion_dense_count"] = dense_count
        debug["pre_fusion_lexical_count"] = lexical_count

        if query_vector is not None:
            debug["query_vector"] = query_vector[:8]
            debug["query_vector_dimensions"] = len(query_vector)

        # Log per-store results
        for sr in store_results:
            if sr.error is None:
                logger.debug(
                    "broker: store=%s method=%s count=%d latency=%.1fms",
                    sr.store_id, sr.retrieval_method.value, sr.candidate_count, sr.latency_ms,
                )
            else:
                logger.warning(
                    "broker: store=%s method=%s failed: %s",
                    sr.store_id, sr.retrieval_method.value, sr.error,
                )

        # Check if all active stores failed
        active_count = int(use_dense) + int(use_lexical)
        failed_count = sum(1 for sr in store_results if sr.error is not None)

        if failed_count == active_count:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("broker: all stores failed, errors=%s", errors)
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

        # Step 5: RRF fusion
        if ranked_lists:
            fused = reciprocal_rank_fusion(ranked_lists, weights, k=self._config.rrf_k)
        else:
            fused = []

        debug["post_fusion_count"] = len(fused)

        # Step 6: Source caps
        if self._config.max_candidates_per_source > 0:
            pre_cap = len(fused)
            fused = apply_source_caps(fused, self._config.max_candidates_per_source)
            debug["post_source_cap_count"] = len(fused)
            debug["source_cap_removals"] = pre_cap - len(fused)
        else:
            debug["post_source_cap_count"] = len(fused)
            debug["source_cap_removals"] = 0

        # Step 7: Truncate to top_k
        fused = fused[: effective_query.top_k]

        # Step 8: Determine outcome
        if not fused:
            outcome = BrokerOutcome.NO_RESULTS
        elif failed_count > 0:
            outcome = BrokerOutcome.PARTIAL
        else:
            outcome = BrokerOutcome.SUCCESS

        elapsed = (time.monotonic() - start) * 1000

        logger.info(
            "broker: outcome=%s candidates=%d latency=%.1fms",
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

    def _fanout_dense(
        self, query: RetrievalQuery
    ) -> tuple[StoreResult, list[float] | None]:
        """Execute dense retrieval via the vector store."""
        t0 = time.monotonic()
        try:
            query_vector = self._query_embedder.embed_query(query.normalized_query)
            candidates = self._vector_store.search(query, query_vector)
            elapsed = (time.monotonic() - t0) * 1000
            if self._config.fanout_timeout_ms > 0 and elapsed > self._config.fanout_timeout_ms:
                return StoreResult(
                    store_id=self._vector_store.store_id,
                    retrieval_method=RetrievalMethod.DENSE,
                    candidates=candidates,
                    candidate_count=len(candidates),
                    latency_ms=elapsed,
                    error=(
                        f"dense fanout exceeded timeout"
                        f" ({elapsed:.0f}ms > {self._config.fanout_timeout_ms}ms)"
                    ),
                    error_classification=ErrorClassification.TRANSIENT,
                ), query_vector
            return StoreResult(
                store_id=self._vector_store.store_id,
                retrieval_method=RetrievalMethod.DENSE,
                candidates=candidates,
                candidate_count=len(candidates),
                latency_ms=elapsed,
            ), query_vector
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            classification = _classify_error(exc)
            logger.error(
                "broker: dense fanout exception store=%s classification=%s: %s",
                self._vector_store.store_id, classification.value, exc,
            )
            return StoreResult(
                store_id=self._vector_store.store_id,
                retrieval_method=RetrievalMethod.DENSE,
                candidates=[],
                candidate_count=0,
                latency_ms=elapsed,
                error=f"store={self._vector_store.store_id} trace_id={query.trace_id}: {exc}",
                error_classification=classification,
            ), None

    def _fanout_lexical(self, query: RetrievalQuery) -> StoreResult:
        """Execute lexical retrieval via the lexical store."""
        t0 = time.monotonic()
        try:
            candidates = self._lexical_store.search(query)
            elapsed = (time.monotonic() - t0) * 1000
            if self._config.fanout_timeout_ms > 0 and elapsed > self._config.fanout_timeout_ms:
                return StoreResult(
                    store_id=self._lexical_store.store_id,
                    retrieval_method=RetrievalMethod.LEXICAL,
                    candidates=candidates,
                    candidate_count=len(candidates),
                    latency_ms=elapsed,
                    error=(
                        f"lexical fanout exceeded timeout"
                        f" ({elapsed:.0f}ms > {self._config.fanout_timeout_ms}ms)"
                    ),
                    error_classification=ErrorClassification.TRANSIENT,
                )
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
                "broker: lexical fanout exception store=%s classification=%s: %s",
                self._lexical_store.store_id, classification.value, exc,
            )
            return StoreResult(
                store_id=self._lexical_store.store_id,
                retrieval_method=RetrievalMethod.LEXICAL,
                candidates=[],
                candidate_count=0,
                latency_ms=elapsed,
                error=f"store={self._lexical_store.store_id} trace_id={query.trace_id}: {exc}",
                error_classification=classification,
            )
