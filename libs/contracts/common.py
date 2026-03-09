"""Shared enums, type aliases, and constants for the Drifter contract layer."""

from __future__ import annotations

from enum import Enum

# ── ID type aliases ──────────────────────────────────────────────────
# Using str for maximum portability (UUID-compatible, DB-friendly).
SourceId = str
DocumentId = str
BlockId = str
ChunkId = str
EmbeddingId = str
TraceId = str
RunId = str


# ── Enums ────────────────────────────────────────────────────────────


class BlockType(Enum):
    """Structural element types within a parsed document."""

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    CODE = "code"
    LIST = "list"
    IMAGE_CAPTION = "image_caption"
    QUOTE = "quote"
    METADATA = "metadata"


class RetrievalMethod(Enum):
    """How a candidate was retrieved from the index."""

    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


class SelectionReason(Enum):
    """Why a chunk was selected for the context pack."""

    TOP_RANKED = "top_ranked"
    DIVERSITY = "diversity"
    RECENCY = "recency"
    AUTHORITY = "authority"
