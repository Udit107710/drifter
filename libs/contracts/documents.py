"""Document-level contracts: source references, raw bytes, canonical documents, blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from libs.contracts.common import (
    BlockId,
    BlockType,
    DocumentId,
    SourceId,
)


@dataclass(frozen=True)
class SourceDocumentRef:
    """A pointer to an external document source.

    Created by ingestion when a document is fetched. Immutable once recorded.
    """

    source_id: SourceId
    uri: str
    content_hash: str
    fetched_at: datetime
    version: int
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    # Placeholders for authority/freshness scoring
    authority_score: float | None = None
    freshness_hint: datetime | None = None

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.uri:
            raise ValueError("uri must not be empty")
        if not self.content_hash:
            raise ValueError("content_hash must not be empty")
        if self.version < 1:
            raise ValueError("version must be >= 1")


@dataclass(frozen=True)
class RawDocument:
    """Raw bytes fetched from a source, before parsing.

    Bridges ingestion → parsing. Carries the original payload plus the
    source reference for lineage.
    """

    source_ref: SourceDocumentRef
    raw_bytes: bytes
    mime_type: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.raw_bytes:
            raise ValueError("raw_bytes must not be empty")
        if not self.mime_type:
            raise ValueError("mime_type must not be empty")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be >= 0")


@dataclass(frozen=True)
class Block:
    """A structural element within a parsed document (heading, paragraph, table, etc.)."""

    block_id: BlockId
    block_type: BlockType
    content: str
    position: int
    level: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("block_id must not be empty")
        if self.position < 0:
            raise ValueError("position must be >= 0")
        if self.level is not None and self.level < 1:
            raise ValueError("level must be >= 1 when set")


@dataclass(frozen=True)
class CanonicalDocument:
    """A parsed, structured representation of a source document.

    Produced by the parsing subsystem. Preserves document structure as a
    list of typed blocks — never flattened to plain text.
    """

    document_id: DocumentId
    source_ref: SourceDocumentRef
    blocks: list[Block]
    parser_version: str
    parsed_at: datetime
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    # ACL placeholder for future multi-tenant access control
    acl: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id must not be empty")
        if not self.blocks:
            raise ValueError("blocks must not be empty")
        if not self.parser_version:
            raise ValueError("parser_version must not be empty")
