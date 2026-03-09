"""TEI embedding provider adapter stub.

Satisfies the ``EmbeddingProvider`` protocol without importing any
external HTTP or TEI client libraries at module level.
"""

from __future__ import annotations

from libs.adapters.config import TeiConfig
from libs.contracts import Chunk, ChunkEmbedding
from libs.embeddings.models import EmbeddingModelInfo

_NOT_IMPLEMENTED_MSG = (
    "Implement TeiEmbeddingProvider to use TEI for embedding generation"
)


class TeiEmbeddingProvider:
    """Stub adapter for Text Embeddings Inference embedding generation.

    Satisfies the ``EmbeddingProvider`` protocol.  All data methods raise
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

    # -- EmbeddingProvider protocol ------------------------------------------

    def model_info(self) -> EmbeddingModelInfo:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    def embed_chunks(self, chunks: list[Chunk]) -> list[ChunkEmbedding]:
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
