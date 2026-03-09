"""Chunking subsystem.

Splits CanonicalDocuments into Chunks using configurable strategies.
Consumes CanonicalDocument (ordered list[Block]), produces list[Chunk]
with full lineage, token counts, and deterministic IDs.
"""

from libs.chunking.config import FixedWindowConfig, ParentChildConfig, RecursiveConfig
from libs.chunking.protocols import ChunkingStrategy, TokenCounter
from libs.chunking.section_tracker import SectionTracker
from libs.chunking.strategies.fixed_window import FixedWindowChunker
from libs.chunking.strategies.parent_child import ParentChildChunker
from libs.chunking.strategies.recursive import RecursiveStructureChunker
from libs.chunking.token_counter import WhitespaceTokenCounter

__all__ = [
    "ChunkingStrategy",
    "FixedWindowChunker",
    "FixedWindowConfig",
    "ParentChildChunker",
    "ParentChildConfig",
    "RecursiveConfig",
    "RecursiveStructureChunker",
    "SectionTracker",
    "TokenCounter",
    "WhitespaceTokenCounter",
]
