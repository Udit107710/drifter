"""TEI query embedder adapter stub.

Satisfies the ``QueryEmbedder`` protocol without importing any
external HTTP or TEI client libraries at module level.
"""

from __future__ import annotations

from libs.adapters.config import TeiConfig

_NOT_IMPLEMENTED_MSG = (
    "Implement TeiQueryEmbedder to use TEI for query embedding"
)


class TeiQueryEmbedder:
    """Stub adapter for Text Embeddings Inference query embedding.

    Satisfies the ``QueryEmbedder`` protocol.  All data methods raise
    ``NotImplementedError`` until a real implementation is provided.
    """

    def __init__(self, config: TeiConfig) -> None:
        self._config = config

    # -- Lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Connect to the TEI server.  TODO: implement."""

    def close(self) -> None:
        """Close the TEI connection.  TODO: implement."""

    def health_check(self) -> bool:
        """Return *False* — not connected."""
        return False

    # -- QueryEmbedder protocol ----------------------------------------------

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
