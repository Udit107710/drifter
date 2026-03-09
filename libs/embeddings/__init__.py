"""Embeddings subsystem.

Responsibilities:
- Embed chunks via provider-agnostic EmbeddingProvider protocol
- Batch embedding requests for throughput
- Track embedding model version per vector

Boundary: consumes list[Chunk], produces list[ChunkEmbedding].
Must not depend on a specific embedding provider.
"""

from libs.embeddings.mock_provider import DeterministicEmbeddingProvider
from libs.embeddings.models import EmbeddingModelInfo
from libs.embeddings.protocols import EmbeddingProvider
from libs.embeddings.query_embedder import DeterministicQueryEmbedder

__all__ = [
    "DeterministicEmbeddingProvider",
    "DeterministicQueryEmbedder",
    "EmbeddingModelInfo",
    "EmbeddingProvider",
]
