"""In-memory lexical index writer for testing."""

from __future__ import annotations

from libs.contracts.chunks import Chunk
from libs.contracts.common import ChunkId


class MemoryLexicalIndexWriter:
    """Dict-backed lexical index with upsert semantics."""

    def __init__(self) -> None:
        self._store: dict[ChunkId, Chunk] = {}

    def write_batch(self, chunks: list[Chunk]) -> int:
        for chunk in chunks:
            self._store[chunk.chunk_id] = chunk
        return len(chunks)

    def delete_by_chunk_ids(self, chunk_ids: list[ChunkId]) -> int:
        count = 0
        for cid in chunk_ids:
            if cid in self._store:
                del self._store[cid]
                count += 1
        return count
