"""Embedding contracts: vector representations of chunks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from libs.contracts.common import ChunkId, EmbeddingId


@dataclass(frozen=True)
class ChunkEmbedding:
    """A dense vector representation of a chunk.

    Records the model identity and version so embeddings from different
    models are never silently mixed in the same index.
    """

    embedding_id: EmbeddingId
    chunk_id: ChunkId
    vector: list[float]
    model_id: str
    model_version: str
    dimensions: int
    created_at: datetime
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.embedding_id:
            raise ValueError("embedding_id must not be empty")
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        if not self.vector:
            raise ValueError("vector must not be empty")
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if not self.model_version:
            raise ValueError("model_version must not be empty")
        if self.dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        if len(self.vector) != self.dimensions:
            raise ValueError(
                f"vector length ({len(self.vector)}) must match dimensions ({self.dimensions})"
            )
