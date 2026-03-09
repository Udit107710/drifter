"""Indexing subsystem protocols: storage and index writer interfaces."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from libs.contracts.chunks import Chunk
from libs.contracts.common import ChunkId, DocumentId
from libs.contracts.embeddings import ChunkEmbedding


@runtime_checkable
class ChunkRepository(Protocol):
    """Persistence for chunk records."""

    def store(self, chunk: Chunk) -> None: ...
    def store_batch(self, chunks: list[Chunk]) -> None: ...
    def get(self, chunk_id: ChunkId) -> Chunk | None: ...
    def get_by_document(self, document_id: DocumentId) -> list[Chunk]: ...
    def delete_by_document(self, document_id: DocumentId) -> int: ...


@runtime_checkable
class EmbeddingRepository(Protocol):
    """Persistence for chunk embeddings."""

    def store(self, embedding: ChunkEmbedding) -> None: ...
    def store_batch(self, embeddings: list[ChunkEmbedding]) -> None: ...
    def get_by_chunk(self, chunk_id: ChunkId) -> list[ChunkEmbedding]: ...
    def get_by_chunk_and_model(
        self, chunk_id: ChunkId, model_id: str, model_version: str
    ) -> ChunkEmbedding | None: ...
    def list_by_model(self, model_id: str, model_version: str) -> list[ChunkEmbedding]: ...
    def delete_by_chunk(self, chunk_id: ChunkId) -> int: ...


@runtime_checkable
class VectorIndexWriter(Protocol):
    """Writes embeddings and metadata to a vector store."""

    def write_batch(self, embeddings: list[ChunkEmbedding], chunks: list[Chunk]) -> int: ...
    def delete_by_chunk_ids(self, chunk_ids: list[ChunkId]) -> int: ...


@runtime_checkable
class LexicalIndexWriter(Protocol):
    """Writes chunk text to a lexical (full-text) index."""

    def write_batch(self, chunks: list[Chunk]) -> int: ...
    def delete_by_chunk_ids(self, chunk_ids: list[ChunkId]) -> int: ...
