"""HuggingFace Inference API reranker adapter.

Implements the ``Reranker`` protocol using ``huggingface_hub.InferenceClient``
with ``text_classification()`` for cross-encoder reranking.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from libs.adapters.config import HuggingFaceConfig
from libs.contracts.retrieval import RankedCandidate, RetrievalCandidate, RetrievalQuery

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class HuggingFaceReranker:
    """Cross-encoder reranker backed by the HuggingFace Inference API.

    Satisfies the ``Reranker`` protocol.

    Uses ``InferenceClient.text_classification()`` to score (query, document)
    pairs via a cross-encoder model (e.g. ``BAAI/bge-reranker-v2-m3``).
    """

    def __init__(self, config: HuggingFaceConfig, model_name: str) -> None:
        self._config = config
        self._model_name = model_name
        self._reranker_id = f"hf-reranker:{model_name}"
        self._client: Any | None = None

    # -- Protocol property ---------------------------------------------------

    @property
    def reranker_id(self) -> str:
        return self._reranker_id

    # -- Lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Create the HuggingFace InferenceClient."""
        from huggingface_hub import InferenceClient

        self._client = InferenceClient(
            provider=self._config.provider,
            api_key=self._config.api_key,
            timeout=self._config.timeout_s,
        )
        logger.info(
            "HuggingFace reranker connected — model=%s, provider=%s",
            self._config.reranker_model,
            self._config.provider,
        )

    def close(self) -> None:
        """Release the client."""
        self._client = None

    def health_check(self) -> bool:
        """Return True if the client is initialised."""
        return self._client is not None

    # -- Reranker protocol ---------------------------------------------------

    def rerank(
        self,
        candidates: list[RetrievalCandidate],
        query: RetrievalQuery,
    ) -> list[RankedCandidate]:
        """Score all candidates against the query using HF text_classification."""
        if self._client is None:
            raise RuntimeError("HuggingFaceReranker is not connected. Call connect() first.")

        if not candidates:
            return []

        scored: list[tuple[float, RetrievalCandidate]] = []
        for candidate in candidates:
            text = f"{query.normalized_query} [SEP] {candidate.chunk.content}"
            result = self._client.text_classification(
                text, model=self._config.reranker_model
            )
            # Extract the LABEL_1 (relevant) score; fall back to first label
            score = 0.0
            for label_info in result:
                if label_info.label == "LABEL_1":
                    score = label_info.score
                    break
            else:
                # If no LABEL_1 found, use first result score
                if result:
                    score = result[0].score
            scored.append((score, candidate))

        scored.sort(key=lambda t: t[0], reverse=True)

        return [
            RankedCandidate(
                candidate=candidate,
                rank=rank + 1,
                rerank_score=score,
                reranker_id=self._reranker_id,
            )
            for rank, (score, candidate) in enumerate(scored)
        ]
