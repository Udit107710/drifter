"""Chunk contracts: chunk records with lineage back to source documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from libs.contracts.common import BlockId, ChunkId, DocumentId, SourceId


@dataclass(frozen=True)
class ChunkLineage:
    """Provenance record linking a chunk back to its source document and blocks."""

    source_id: SourceId
    document_id: DocumentId
    block_ids: list[BlockId]
    chunk_strategy: str
    parser_version: str
    created_at: datetime
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.document_id:
            raise ValueError("document_id must not be empty")
        if not self.block_ids:
            raise ValueError("block_ids must not be empty")
        if not self.chunk_strategy:
            raise ValueError("chunk_strategy must not be empty")
        if not self.parser_version:
            raise ValueError("parser_version must not be empty")


@dataclass(frozen=True)
class Chunk:
    """A retrieval-ready text unit derived from one or more document blocks.

    Each chunk carries full lineage back to the source document, the blocks
    it spans, and the strategy that produced it. Token count is always set
    so downstream stages (context builder, generation) can budget accurately.
    """

    chunk_id: ChunkId
    document_id: DocumentId
    source_id: SourceId
    block_ids: list[BlockId]
    content: str
    content_hash: str
    token_count: int
    strategy: str
    byte_offset_start: int
    byte_offset_end: int
    lineage: ChunkLineage
    schema_version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    # ACL placeholder — propagated from CanonicalDocument
    acl: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        if not self.document_id:
            raise ValueError("document_id must not be empty")
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not self.block_ids:
            raise ValueError("block_ids must not be empty")
        if not self.content:
            raise ValueError("content must not be empty")
        if not self.content_hash:
            raise ValueError("content_hash must not be empty")
        if self.token_count < 1:
            raise ValueError("token_count must be >= 1")
        if self.byte_offset_start < 0:
            raise ValueError("byte_offset_start must be >= 0")
        if self.byte_offset_end <= self.byte_offset_start:
            raise ValueError("byte_offset_end must be > byte_offset_start")
