"""CrossEncoderReranker: placeholder for future cross-encoder model integration.

This module is kept for backward compatibility.  When a TEI config is
available, prefer :class:`libs.adapters.tei.TeiCrossEncoderReranker`.
"""

from __future__ import annotations

from libs.contracts.retrieval import RankedCandidate, RetrievalCandidate, RetrievalQuery


class CrossEncoderReranker:
    """Placeholder reranker that will use a cross-encoder model for scoring.

    Currently raises NotImplementedError — to be implemented when a
    cross-encoder inference backend (e.g. TEI) is available.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    @property
    def reranker_id(self) -> str:
        return f"cross-encoder:{self._model_name}"

    def rerank(
        self,
        candidates: list[RetrievalCandidate],
        query: RetrievalQuery,
    ) -> list[RankedCandidate]:
        raise NotImplementedError(
            f"CrossEncoderReranker({self._model_name}) is not yet implemented. "
            "Provide a cross-encoder inference backend to enable this reranker."
        )
