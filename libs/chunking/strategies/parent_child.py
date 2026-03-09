"""Parent-child chunking strategy.

Creates two-level chunk hierarchies: large *parent* chunks for broad
context and smaller overlapping *child* chunks for precise retrieval.
Parents are produced by grouping document blocks up to a token budget
(structural splitting), then each parent is sliced into fixed-window
children with configurable overlap.

Output order: all parent chunks first, then all child chunks.
"""

from __future__ import annotations

from typing import Any

from libs.chunking.builder import build_chunk, compute_block_byte_offsets
from libs.chunking.chunk_id import content_hash, generate_chunk_id
from libs.chunking.config import ParentChildConfig
from libs.chunking.protocols import TokenCounter
from libs.chunking.section_tracker import SectionTracker
from libs.chunking.token_counter import WhitespaceTokenCounter
from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.documents import Block, CanonicalDocument


class ParentChildChunker:
    """Two-level chunker producing parent and child chunks.

    Satisfies the ``ChunkingStrategy`` protocol.

    **Algorithm**

    1. Walk the document blocks and greedily group them into *parent*
       chunks of up to ``parent_chunk_size`` tokens.  A new parent is
       started whenever adding the next block would exceed the budget.
    2. For each parent, slide a fixed window of ``child_chunk_size``
       tokens with ``child_overlap`` overlap across the parent content
       to produce *child* chunks.  A trailing child smaller than
       ``min_child_size`` tokens is discarded.
    3. Parent metadata is set with ``is_parent=True`` and
       ``child_chunk_ids`` listing the IDs of its children.  Each child
       carries ``parent_chunk_id`` and ``child_index`` in its metadata.
    """

    def __init__(
        self,
        config: ParentChildConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._config = config or ParentChildConfig()
        self._token_counter = token_counter or WhitespaceTokenCounter()

    # ------------------------------------------------------------------
    # ChunkingStrategy protocol
    # ------------------------------------------------------------------

    def chunk(self, doc: CanonicalDocument) -> list[Chunk]:
        """Split *doc* into parent and child chunks with full lineage."""
        blocks = doc.blocks
        if not blocks:
            return []

        block_byte_offsets = compute_block_byte_offsets(blocks)

        # Build section tracker for metadata.
        tracker = SectionTracker()
        block_section: dict[str, list[str]] = {}
        for block in blocks:
            tracker.update(block)
            block_section[block.block_id] = tracker.current_path()

        # --- Phase 1: create parent groups (block_ids + content) ------
        parent_groups = self._create_parent_groups(blocks)

        # --- Phase 2: for each parent group, create children ----------
        all_parents: list[Chunk] = []
        all_children: list[Chunk] = []

        for parent_seq, group in enumerate(parent_groups):
            group_block_ids: list[str] = group["block_ids"]
            parent_content: str = group["content"]

            # Generate children from the parent content.
            child_chunks = self._split_children(
                parent_content=parent_content,
                parent_block_ids=group_block_ids,
                parent_seq=parent_seq,
                doc=doc,
                block_byte_offsets=block_byte_offsets,
            )

            child_chunk_ids = [c.chunk_id for c in child_chunks]

            # Section path from the first block of this parent group.
            section_path = (
                block_section.get(group_block_ids[0], [])
                if group_block_ids
                else []
            )

            # Build parent chunk with child_chunk_ids already present.
            parent_metadata: dict[str, Any] = {
                "is_parent": True,
                "child_chunk_ids": child_chunk_ids,
                "section_path": list(section_path),
            }

            parent_chunk = build_chunk(
                content=parent_content,
                block_ids=group_block_ids,
                sequence_index=parent_seq,
                strategy_name=self.strategy_name(),
                doc=doc,
                token_counter=self._token_counter,
                block_byte_offsets=block_byte_offsets,
                metadata=parent_metadata,
                acl=list(doc.acl),
            )

            all_parents.append(parent_chunk)
            all_children.extend(child_chunks)

        return all_parents + all_children

    def strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        return "parent_child"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_parent_groups(
        self, blocks: list[Block]
    ) -> list[dict[str, Any]]:
        """Greedily group blocks into parent-sized chunks.

        Each group is a dict with ``block_ids`` (ordered list) and
        ``content`` (blocks joined by newlines).
        """
        groups: list[dict[str, Any]] = []
        current_ids: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for block in blocks:
            block_tokens = self._token_counter.count(block.content)

            # If a single block exceeds parent_chunk_size, it becomes
            # its own group to avoid infinite loops.
            if not current_ids and block_tokens >= self._config.parent_chunk_size:
                groups.append(
                    {
                        "block_ids": [block.block_id],
                        "content": block.content,
                    }
                )
                continue

            # Would adding this block exceed the budget?
            # Account for the newline separator between parts.
            separator_cost = 1 if current_parts else 0
            projected = current_tokens + separator_cost + block_tokens

            if projected > self._config.parent_chunk_size and current_ids:
                # Flush current group.
                groups.append(
                    {
                        "block_ids": list(current_ids),
                        "content": "\n".join(current_parts),
                    }
                )
                current_ids = []
                current_parts = []
                current_tokens = 0

            current_ids.append(block.block_id)
            current_parts.append(block.content)
            current_tokens += (1 if current_tokens > 0 else 0) + block_tokens

        # Flush remaining.
        if current_ids:
            groups.append(
                {
                    "block_ids": list(current_ids),
                    "content": "\n".join(current_parts),
                }
            )

        return groups

    def _split_children(
        self,
        parent_content: str,
        parent_block_ids: list[str],
        parent_seq: int,
        doc: CanonicalDocument,
        block_byte_offsets: dict[str, tuple[int, int]],
    ) -> list[Chunk]:
        """Slide a fixed window over *parent_content* to produce child chunks.

        Child ``sequence_index`` values use ``parent_seq * 1000 + child_index``
        to avoid collisions with parent indices and across parent groups.
        """
        tokens = parent_content.split()
        if not tokens:
            return []

        child_size = self._config.child_chunk_size
        overlap = self._config.child_overlap
        step = child_size - overlap

        # Precompute the parent's byte-offset range for child offset derivation.
        parent_byte_start = self._parent_byte_start(
            parent_block_ids, block_byte_offsets, parent_content
        )

        # Build token char-offset map within parent_content so we can
        # derive per-child byte offsets relative to the document.
        token_char_starts: list[int] = []
        search_pos = 0
        for tok in tokens:
            idx = parent_content.index(tok, search_pos)
            token_char_starts.append(idx)
            search_pos = idx + len(tok)

        # We will compute a *temporary* parent chunk_id so children can
        # reference it.  This mirrors the ID that build_chunk will
        # produce for the parent.
        parent_chunk_id = generate_chunk_id(
            doc.document_id,
            self.strategy_name(),
            parent_content,
            parent_seq,
        )

        children: list[Chunk] = []
        child_index = 0
        window_start = 0
        total_tokens = len(tokens)

        while window_start < total_tokens:
            window_end = min(window_start + child_size, total_tokens)
            window_tokens = tokens[window_start:window_end]
            content = " ".join(window_tokens)
            token_count = self._token_counter.count(content)

            # Discard trailing runt child (but keep the first).
            if token_count < self._config.min_child_size and child_index > 0:
                break

            # Compute byte offsets relative to the document.
            child_char_start = token_char_starts[window_start]
            last_tok_start = token_char_starts[window_end - 1]
            child_char_end = last_tok_start + len(tokens[window_end - 1])

            child_byte_start = parent_byte_start + len(
                parent_content[:child_char_start].encode("utf-8")
            )
            child_byte_end = parent_byte_start + len(
                parent_content[:child_char_end].encode("utf-8")
            )

            seq = parent_seq * 1000 + child_index

            child_chunk_id = generate_chunk_id(
                doc.document_id,
                self.strategy_name(),
                content,
                seq,
            )
            c_hash = content_hash(content)

            lineage = ChunkLineage(
                source_id=doc.source_ref.source_id,
                document_id=doc.document_id,
                block_ids=parent_block_ids,
                chunk_strategy=self.strategy_name(),
                parser_version=doc.parser_version,
                created_at=doc.parsed_at,
            )

            child_metadata: dict[str, Any] = {
                "parent_chunk_id": parent_chunk_id,
                "child_index": child_index,
            }

            child_chunk = Chunk(
                chunk_id=child_chunk_id,
                document_id=doc.document_id,
                source_id=doc.source_ref.source_id,
                block_ids=parent_block_ids,
                content=content,
                content_hash=c_hash,
                token_count=token_count,
                strategy=self.strategy_name(),
                byte_offset_start=child_byte_start,
                byte_offset_end=child_byte_end,
                lineage=lineage,
                metadata=child_metadata,
                acl=list(doc.acl),
            )

            children.append(child_chunk)
            child_index += 1
            window_start += step

        return children

    @staticmethod
    def _parent_byte_start(
        block_ids: list[str],
        block_byte_offsets: dict[str, tuple[int, int]],
        fallback_content: str,
    ) -> int:
        """Return the document-level byte offset where the parent starts."""
        starts = [
            block_byte_offsets[bid][0]
            for bid in block_ids
            if bid in block_byte_offsets
        ]
        return min(starts) if starts else 0
