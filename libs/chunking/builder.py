"""Chunk builder — assembles Chunk instances with deterministic IDs and lineage."""

from __future__ import annotations

from typing import Any

from libs.chunking.chunk_id import content_hash, generate_chunk_id
from libs.chunking.protocols import TokenCounter
from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.documents import Block, CanonicalDocument


def compute_block_byte_offsets(blocks: list[Block]) -> dict[str, tuple[int, int]]:
    """Compute cumulative UTF-8 byte offsets for each block, joined with newline separators."""
    offsets: dict[str, tuple[int, int]] = {}
    pos = 0
    for i, block in enumerate(blocks):
        byte_len = len(block.content.encode("utf-8"))
        offsets[block.block_id] = (pos, pos + byte_len)
        pos += byte_len
        if i < len(blocks) - 1:
            pos += 1  # newline separator
    return offsets


def build_chunk(
    content: str,
    block_ids: list[str],
    sequence_index: int,
    strategy_name: str,
    doc: CanonicalDocument,
    token_counter: TokenCounter,
    block_byte_offsets: dict[str, tuple[int, int]],
    metadata: dict[str, Any] | None = None,
    acl: list[str] | None = None,
) -> Chunk:
    """Build a Chunk with deterministic ID, content hash, lineage, and byte offsets."""
    chunk_id = generate_chunk_id(doc.document_id, strategy_name, content, sequence_index)
    c_hash = content_hash(content)
    token_count = token_counter.count(content)

    # Byte offsets: from first block start to last block end
    starts = [block_byte_offsets[bid][0] for bid in block_ids if bid in block_byte_offsets]
    ends = [block_byte_offsets[bid][1] for bid in block_ids if bid in block_byte_offsets]
    byte_start = min(starts) if starts else 0
    byte_end = max(ends) if ends else len(content.encode("utf-8"))

    lineage = ChunkLineage(
        source_id=doc.source_ref.source_id,
        document_id=doc.document_id,
        block_ids=block_ids,
        chunk_strategy=strategy_name,
        parser_version=doc.parser_version,
        created_at=doc.parsed_at,
    )

    return Chunk(
        chunk_id=chunk_id,
        document_id=doc.document_id,
        source_id=doc.source_ref.source_id,
        block_ids=block_ids,
        content=content,
        content_hash=c_hash,
        token_count=token_count,
        strategy=strategy_name,
        byte_offset_start=byte_start,
        byte_offset_end=byte_end,
        lineage=lineage,
        metadata=metadata or {},
        acl=acl if acl is not None else list(doc.acl),
    )
