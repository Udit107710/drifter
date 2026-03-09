"""In-memory adapter implementations for deterministic local testing.

These adapters require no external services and produce repeatable results.
"""

from libs.adapters.memory.chunk_repository import MemoryChunkRepository
from libs.adapters.memory.crawl_state_repository import MemoryCrawlStateRepository
from libs.adapters.memory.embedding_repository import MemoryEmbeddingRepository
from libs.adapters.memory.lexical_index_writer import MemoryLexicalIndexWriter
from libs.adapters.memory.source_repository import MemorySourceRepository
from libs.adapters.memory.vector_index_writer import MemoryVectorIndexWriter

__all__ = [
    "MemoryChunkRepository",
    "MemoryCrawlStateRepository",
    "MemoryEmbeddingRepository",
    "MemoryLexicalIndexWriter",
    "MemorySourceRepository",
    "MemoryVectorIndexWriter",
]
