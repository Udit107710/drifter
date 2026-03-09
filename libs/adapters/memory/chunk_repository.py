"""In-memory chunk repository for testing."""

from __future__ import annotations

from collections import defaultdict

from libs.contracts.chunks import Chunk
from libs.contracts.common import ChunkId, DocumentId


class MemoryChunkRepository:
    """Dict-backed chunk store with secondary index by document_id."""

    def __init__(self) -> None:
        self._store: dict[ChunkId, Chunk] = {}
        self._by_document: dict[DocumentId, list[ChunkId]] = defaultdict(list)

    def store(self, chunk: Chunk) -> None:
        is_new = chunk.chunk_id not in self._store
        self._store[chunk.chunk_id] = chunk
        if is_new:
            self._by_document[chunk.document_id].append(chunk.chunk_id)

    def store_batch(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.store(chunk)

    def get(self, chunk_id: ChunkId) -> Chunk | None:
        return self._store.get(chunk_id)

    def get_by_document(self, document_id: DocumentId) -> list[Chunk]:
        chunk_ids = self._by_document.get(document_id, [])
        return [self._store[cid] for cid in chunk_ids if cid in self._store]

    def delete_by_document(self, document_id: DocumentId) -> int:
        chunk_ids = self._by_document.pop(document_id, [])
        count = 0
        for cid in chunk_ids:
            if cid in self._store:
                del self._store[cid]
                count += 1
        return count
