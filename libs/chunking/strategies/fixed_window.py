"""Fixed-window token chunking strategy.

Splits a CanonicalDocument into overlapping chunks of a fixed token
count by sliding a window across the concatenated block contents.
"""

from __future__ import annotations

from libs.chunking.builder import build_chunk, compute_block_byte_offsets
from libs.chunking.config import FixedWindowConfig
from libs.chunking.protocols import TokenCounter
from libs.chunking.section_tracker import SectionTracker
from libs.chunking.token_counter import WhitespaceTokenCounter
from libs.contracts.chunks import Chunk
from libs.contracts.documents import CanonicalDocument


class FixedWindowChunker:
    """Sliding-window chunker that produces fixed-size token chunks with overlap.

    Satisfies the ``ChunkingStrategy`` protocol.
    """

    def __init__(
        self,
        config: FixedWindowConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self._config = config or FixedWindowConfig()
        self._token_counter = token_counter or WhitespaceTokenCounter()

    # ------------------------------------------------------------------
    # ChunkingStrategy protocol
    # ------------------------------------------------------------------

    def chunk(self, doc: CanonicalDocument) -> list[Chunk]:
        """Split *doc* into fixed-window chunks with full lineage."""
        blocks = doc.blocks
        if not blocks:
            return []

        # 1. Concatenate block contents with \n separators.
        concatenated = "\n".join(b.content for b in blocks)

        # 2. Build char-position → block_id mapping.
        #    Newline separators are assigned to the preceding block.
        char_to_block: list[str] = []
        for i, block in enumerate(blocks):
            char_to_block.extend([block.block_id] * len(block.content))
            if i < len(blocks) - 1:
                # Newline separator belongs to the preceding block.
                char_to_block.append(block.block_id)

        # 3. Build a per-block section path snapshot.
        #    We walk the blocks once and record the section_path *after*
        #    updating with each block so that heading blocks themselves
        #    are included in the path.
        tracker = SectionTracker()
        block_section: dict[str, list[str]] = {}
        for block in blocks:
            tracker.update(block)
            block_section[block.block_id] = tracker.current_path()

        # 4. Tokenise the concatenated text, tracking each token's
        #    start character position.
        tokens: list[str] = []
        token_starts: list[int] = []
        pos = 0
        for segment in concatenated.split():
            # Find the real start position (skip whitespace).
            idx = concatenated.index(segment, pos)
            tokens.append(segment)
            token_starts.append(idx)
            pos = idx + len(segment)

        if not tokens:
            return []

        # 5. Pre-compute block byte offsets for build_chunk.
        block_byte_offsets = compute_block_byte_offsets(blocks)

        # 6. Slide the window.
        chunk_size = self._config.chunk_size
        overlap = self._config.overlap
        step = chunk_size - overlap

        chunks: list[Chunk] = []
        sequence_index = 0
        total_tokens = len(tokens)

        window_start = 0
        while window_start < total_tokens:
            window_end = min(window_start + chunk_size, total_tokens)
            window_tokens = tokens[window_start:window_end]
            window_token_starts = token_starts[window_start:window_end]

            # Count tokens via the counter (may differ from len for
            # non-whitespace counters).
            content = " ".join(window_tokens)
            token_count = self._token_counter.count(content)

            # Discard trailing chunk if below minimum size.
            if token_count < self._config.min_chunk_size and sequence_index > 0:
                break

            # Determine which block_ids this window spans.
            seen_block_ids: list[str] = []
            seen_set: set[str] = set()
            for tok_start in window_token_starts:
                if tok_start < len(char_to_block):
                    bid = char_to_block[tok_start]
                    if bid not in seen_set:
                        seen_set.add(bid)
                        seen_block_ids.append(bid)

            # Section path: use the section path at the first block
            # of this window.
            section_path = block_section.get(seen_block_ids[0], []) if seen_block_ids else []

            chunk = build_chunk(
                content=content,
                block_ids=seen_block_ids,
                sequence_index=sequence_index,
                strategy_name=self.strategy_name(),
                doc=doc,
                token_counter=self._token_counter,
                block_byte_offsets=block_byte_offsets,
                metadata={"section_path": list(section_path)},
                acl=list(doc.acl),
            )
            chunks.append(chunk)
            sequence_index += 1

            # Advance window.
            window_start += step

        return chunks

    def strategy_name(self) -> str:
        """Return the name of this chunking strategy."""
        return "fixed_window"
