"""Indexing subsystem.

Responsibilities:
- Write ChunkEmbeddings to vector stores
- Write chunk text and metadata to lexical stores
- Maintain consistency between stores
- Handle index lifecycle (create, version, alias swap, GC)

Boundary: consumes list[Chunk] + list[ChunkEmbedding], writes to indexes.
"""

from libs.indexing.lifecycle import (
    IndexFreshness,
    IndexRegistry,
    IndexVersion,
    MemoryIndexRegistry,
)
from libs.indexing.models import ChunkError, ErrorClassification, IndexingOutcome, IndexingResult
from libs.indexing.protocols import (
    ChunkRepository,
    EmbeddingRepository,
    LexicalIndexWriter,
    VectorIndexWriter,
)
from libs.indexing.service import IndexingService

__all__ = [
    "ChunkError",
    "ChunkRepository",
    "EmbeddingRepository",
    "ErrorClassification",
    "IndexFreshness",
    "IndexRegistry",
    "IndexVersion",
    "IndexingOutcome",
    "IndexingResult",
    "IndexingService",
    "LexicalIndexWriter",
    "MemoryIndexRegistry",
    "VectorIndexWriter",
]
