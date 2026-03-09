"""Section tracker for maintaining heading hierarchy during chunking."""

from __future__ import annotations

from libs.contracts.common import BlockType
from libs.contracts.documents import Block


class SectionTracker:
    """Tracks the current heading hierarchy while iterating over blocks.

    Maintains a stack of ``(level, heading_text)`` tuples. When a new
    heading is encountered, all headings at the same or deeper level are
    popped before the new one is pushed, preserving a clean path from
    the document root to the current section.
    """

    def __init__(self) -> None:
        self._stack: list[tuple[int, str]] = []

    def update(self, block: Block) -> None:
        """Update the heading stack if *block* is a heading with a level."""
        if block.block_type is not BlockType.HEADING:
            return
        if block.level is None:
            return
        # Pop headings at the same or deeper level.
        while self._stack and self._stack[-1][0] >= block.level:
            self._stack.pop()
        self._stack.append((block.level, block.content))

    def current_path(self) -> list[str]:
        """Return the current section path as a list of heading texts."""
        return [name for _, name in self._stack]
