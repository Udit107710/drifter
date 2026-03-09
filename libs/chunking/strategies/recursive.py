"""Recursive structure-aware chunking strategy.

Walks document blocks respecting structural boundaries (headings, block-type
transitions) while enforcing token-budget constraints.  Produces chunks with
full lineage back to the source blocks.
"""

from __future__ import annotations

from libs.chunking.builder import build_chunk, compute_block_byte_offsets
from libs.chunking.config import RecursiveConfig
from libs.chunking.protocols import TokenCounter
from libs.chunking.section_tracker import SectionTracker
from libs.chunking.token_counter import WhitespaceTokenCounter
from libs.contracts.chunks import Chunk
from libs.contracts.common import BlockType
from libs.contracts.documents import Block, CanonicalDocument


class RecursiveStructureChunker:
    """Splits a document into chunks using recursive, structure-aware logic.

    The algorithm respects heading boundaries, block-type transitions (when
    ``prefer_structural`` is enabled), and token-budget limits while
    preserving block lineage.
    """

    def __init__(
        self,
        config: RecursiveConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._config = config or RecursiveConfig()
        self._token_counter = token_counter or WhitespaceTokenCounter()

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------

    def chunk(self, doc: CanonicalDocument) -> list[Chunk]:
        """Split *doc* into chunks with full lineage."""
        if not doc.blocks:
            return []

        block_byte_offsets = compute_block_byte_offsets(doc.blocks)
        tracker = SectionTracker()

        # Buffer: list of (Block, section_path) tuples
        buffer: list[tuple[Block, list[str]]] = []
        buffer_tokens: int = 0

        raw_chunks: list[_RawChunk] = []
        seq = 0

        def _flush() -> None:
            nonlocal buffer, buffer_tokens, seq
            if not buffer:
                return
            content = "\n".join(b.content for b, _ in buffer)
            block_ids = _unique_block_ids(buffer)
            section_path = buffer[0][1]
            raw_chunks.append(
                _RawChunk(
                    content=content,
                    block_ids=block_ids,
                    section_path=section_path,
                    sequence_index=seq,
                )
            )
            seq += 1
            buffer = []
            buffer_tokens = 0

        for block in doc.blocks:
            tracker.update(block)
            section_path = tracker.current_path()
            block_tokens = self._token_counter.count(block.content)

            # (a) Heading → always flush first, then start new buffer.
            if block.block_type is BlockType.HEADING:
                _flush()
                buffer.append((block, list(section_path)))
                buffer_tokens = block_tokens
                continue

            # (b) Adding this block would exceed max_chunk_size.
            if buffer_tokens + block_tokens > self._config.max_chunk_size:
                _flush()

                if block_tokens > self._config.max_chunk_size:
                    # Split oversized block into sub-chunks.
                    sub_chunks = self._split_oversized_block(
                        block, section_path, seq
                    )
                    raw_chunks.extend(sub_chunks)
                    seq += len(sub_chunks)
                    continue

                # Start new buffer with this block.
                buffer.append((block, list(section_path)))
                buffer_tokens = block_tokens
                continue

            # (c) Soft boundary: block type differs and buffer already
            #     meets min_chunk_size (only when prefer_structural is on).
            if (
                self._config.prefer_structural
                and buffer
                and buffer[-1][0].block_type is not block.block_type
                and buffer_tokens >= self._config.min_chunk_size
            ):
                _flush()
                buffer.append((block, list(section_path)))
                buffer_tokens = block_tokens
                continue

            # (d) Accumulate into buffer.
            buffer.append((block, list(section_path)))
            buffer_tokens += block_tokens

        _flush()

        # ------------------------------------------------------------------
        # Merge runts
        # ------------------------------------------------------------------
        merged = self._merge_runts(raw_chunks)

        # ------------------------------------------------------------------
        # Build final Chunk objects
        # ------------------------------------------------------------------
        chunks: list[Chunk] = []
        for idx, rc in enumerate(merged):
            chunks.append(
                build_chunk(
                    content=rc.content,
                    block_ids=rc.block_ids,
                    sequence_index=idx,
                    strategy_name=self.strategy_name(),
                    doc=doc,
                    token_counter=self._token_counter,
                    block_byte_offsets=block_byte_offsets,
                    metadata={"section_path": rc.section_path},
                )
            )
        return chunks

    def strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        return "recursive"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_oversized_block(
        self,
        block: Block,
        section_path: list[str],
        seq_start: int,
    ) -> list[_RawChunk]:
        """Split a single oversized block into token-limited sub-chunks."""
        words = block.content.split()
        max_size = self._config.max_chunk_size
        sub_chunks: list[_RawChunk] = []
        seq = seq_start

        for i in range(0, len(words), max_size):
            segment = " ".join(words[i : i + max_size])
            sub_chunks.append(
                _RawChunk(
                    content=segment,
                    block_ids=[block.block_id],
                    section_path=list(section_path),
                    sequence_index=seq,
                )
            )
            seq += 1

        return sub_chunks

    def _merge_runts(self, raw_chunks: list[_RawChunk]) -> list[_RawChunk]:
        """Merge chunks smaller than *min_chunk_size* into neighbours.

        Forward-merge first: if a chunk is a runt, merge it into the next
        chunk.  If the *last* chunk is a runt after forward merging, merge
        it backward into the previous chunk.
        """
        if not raw_chunks:
            return []

        min_size = self._config.min_chunk_size

        # Forward-merge: walk left to right; if a chunk is a runt, merge
        # it into the next chunk.
        merged: list[_RawChunk] = []
        pending: _RawChunk | None = None

        for rc in raw_chunks:
            if pending is not None:
                # Merge pending runt into this chunk.
                rc = _merge_two(pending, rc)
                pending = None

            tokens = self._token_counter.count(rc.content)
            if tokens < min_size:
                # This is a runt — try to merge forward.
                pending = rc
            else:
                merged.append(rc)

        # If the last chunk is still a pending runt, merge backward.
        if pending is not None:
            if merged:
                merged[-1] = _merge_two(merged[-1], pending)
            else:
                # Only chunk in the document — keep it as-is.
                merged.append(pending)

        return merged


# ------------------------------------------------------------------
# Internal data carrier
# ------------------------------------------------------------------

class _RawChunk:
    """Lightweight intermediate representation before building a Chunk."""

    __slots__ = ("block_ids", "content", "section_path", "sequence_index")

    def __init__(
        self,
        content: str,
        block_ids: list[str],
        section_path: list[str],
        sequence_index: int,
    ) -> None:
        self.content = content
        self.block_ids = block_ids
        self.section_path = section_path
        self.sequence_index = sequence_index


def _unique_block_ids(buffer: list[tuple[Block, list[str]]]) -> list[str]:
    """Return deduplicated block IDs preserving insertion order."""
    seen: set[str] = set()
    ids: list[str] = []
    for block, _ in buffer:
        if block.block_id not in seen:
            seen.add(block.block_id)
            ids.append(block.block_id)
    return ids


def _merge_two(first: _RawChunk, second: _RawChunk) -> _RawChunk:
    """Merge two raw chunks, concatenating content and block IDs."""
    combined_content = first.content + "\n" + second.content
    # Deduplicate block IDs preserving order.
    seen: set[str] = set()
    combined_ids: list[str] = []
    for bid in first.block_ids + second.block_ids:
        if bid not in seen:
            seen.add(bid)
            combined_ids.append(bid)
    return _RawChunk(
        content=combined_content,
        block_ids=combined_ids,
        section_path=first.section_path,
        sequence_index=first.sequence_index,
    )
