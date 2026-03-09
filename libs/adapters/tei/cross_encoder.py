"""TEI cross-encoder reranker adapter stub.

Satisfies the ``Reranker`` protocol without importing any
external HTTP or TEI client libraries at module level.
"""

from __future__ import annotations

from libs.adapters.config import TeiConfig
from libs.contracts import RankedCandidate, RetrievalCandidate, RetrievalQuery

_NOT_IMPLEMENTED_MSG = (
    "Implement TeiCrossEncoderReranker to use TEI for cross-encoder reranking"
)


class TeiCrossEncoderReranker:
    """Stub adapter for TEI cross-encoder reranking.

    Satisfies the ``Reranker`` protocol.  All data methods raise
    ``NotImplementedError`` until a real implementation is provided.
    """

    def __init__(self, config: TeiConfig, model_name: str) -> None:
        self._config = config
        self._model_name = model_name
        self._reranker_id = f"tei-cross-encoder:{model_name}"

    # -- Protocol property ---------------------------------------------------

    @property
    def reranker_id(self) -> str:
        return self._reranker_id

    # -- Lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Connect to the TEI server.  TODO: implement."""

    def close(self) -> None:
        """Close the TEI connection.  TODO: implement."""

    def health_check(self) -> bool:
        """Return *False* — not connected."""
        return False

    # -- Reranker protocol ---------------------------------------------------

    def rerank(
        self,
        candidates: list[RetrievalCandidate],
        query: RetrievalQuery,
    ) -> list[RankedCandidate]:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
