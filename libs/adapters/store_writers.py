"""Index writer adapters that delegate to VectorStore / LexicalStore.

These bridge the indexing protocol (VectorIndexWriter, LexicalIndexWriter)
to the retrieval store protocol (VectorStore, LexicalStore), enabling the
IndexingService to write directly to real stores like Qdrant and OpenSearch.
"""

from __future__ import annotations

from libs.contracts.chunks import Chunk
from libs.contracts.common import ChunkId
from libs.contracts.embeddings import ChunkEmbedding
from libs.retrieval.stores.protocols import LexicalStore, VectorStore


class VectorStoreWriter:
    """Adapts a VectorStore to satisfy the VectorIndexWriter protocol."""

    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def write_batch(self, embeddings: list[ChunkEmbedding], chunks: list[Chunk]) -> int:
        chunk_map = {c.chunk_id: c for c in chunks}
        pairs = [
            (emb, chunk_map[emb.chunk_id])
            for emb in embeddings
            if emb.chunk_id in chunk_map
        ]
        if not pairs:
            return 0
        embs, chs = zip(*pairs, strict=True)
        self._store.add_batch(list(embs), list(chs))
        return len(pairs)

    def delete_by_chunk_ids(self, chunk_ids: list[ChunkId]) -> int:
        return self._store.delete(chunk_ids)


class LexicalStoreWriter:
    """Adapts a LexicalStore to satisfy the LexicalIndexWriter protocol."""

    def __init__(self, store: LexicalStore) -> None:
        self._store = store

    def write_batch(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        self._store.add_batch(chunks)
        return len(chunks)

    def delete_by_chunk_ids(self, chunk_ids: list[ChunkId]) -> int:
        return self._store.delete(chunk_ids)
