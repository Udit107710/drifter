"""Local cross-encoder reranker running on CPU via transformers.

Loads a cross-encoder model (e.g. ``BAAI/bge-reranker-v2-m3``) locally
and scores (query, document) pairs on CPU.  No external API calls.
"""

from __future__ import annotations

import logging
from typing import Any

from libs.contracts.retrieval import RankedCandidate, RetrievalCandidate, RetrievalQuery

logger = logging.getLogger(__name__)


class LocalCrossEncoderReranker:
    """Cross-encoder reranker running locally on CPU.

    Satisfies the ``Reranker`` protocol.  Lazily loads the model on
    first ``rerank()`` call or explicitly via ``connect()``.
    """

    def __init__(self, model_name: str, timeout_s: float = 30.0) -> None:
        self._model_name = model_name
        self._timeout_s = timeout_s
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @property
    def reranker_id(self) -> str:
        return f"local-cross-encoder:{self._model_name}"

    def connect(self) -> None:
        """Load the cross-encoder model and tokenizer onto CPU."""
        self._load_model()

    def close(self) -> None:
        self._tokenizer = None
        self._model = None

    def health_check(self) -> bool:
        if self._model is not None:
            return True
        try:
            self._load_model()
            return True
        except Exception:
            return False

    def rerank(
        self,
        candidates: list[RetrievalCandidate],
        query: RetrievalQuery,
    ) -> list[RankedCandidate]:
        if not candidates:
            return []

        if self._model is None:
            self._load_model()

        import torch

        pairs = [
            [query.normalized_query, candidate.chunk.content]
            for candidate in candidates
        ]

        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            # Cross-encoders output logits; single-column = relevance score
            scores = outputs.logits.squeeze(-1).float().tolist()

        # Handle single-candidate case (tolist returns a scalar)
        if isinstance(scores, float):
            scores = [scores]

        scored = sorted(
            zip(scores, candidates, strict=True),
            key=lambda t: t[0],
            reverse=True,
        )

        return [
            RankedCandidate(
                candidate=candidate,
                rank=rank + 1,
                rerank_score=score,
                reranker_id=self.reranker_id,
            )
            for rank, (score, candidate) in enumerate(scored)
        ]

    def _load_model(self) -> None:
        """Load model onto CPU."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading cross-encoder model %s on CPU...", self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name,
            torch_dtype=torch.float32,
        )
        self._model.eval()
        logger.info("Cross-encoder model %s loaded on CPU", self._model_name)
