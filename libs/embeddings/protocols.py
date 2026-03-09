"""Embedding provider protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from libs.contracts.chunks import Chunk
from libs.contracts.embeddings import ChunkEmbedding
from libs.embeddings.models import EmbeddingModelInfo


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provider-agnostic interface for producing chunk embeddings."""

    def model_info(self) -> EmbeddingModelInfo: ...

    def embed_chunks(self, chunks: list[Chunk]) -> list[ChunkEmbedding]: ...
