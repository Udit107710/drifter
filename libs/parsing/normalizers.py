"""Pure normalisation functions for block content and block lists.

These are composable hooks — call them individually or chain them together.
"""

from __future__ import annotations

import re
from dataclasses import replace

from libs.contracts.documents import Block


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs to a single space and strip edges."""
    return re.sub(r"[ \t]+", " ", text).strip()


def collapse_blank_lines(text: str) -> str:
    """Reduce 3+ consecutive newlines to exactly 2."""
    return re.sub(r"\n{3,}", "\n\n", text)


def strip_header_footer(
    blocks: list[Block],
    header_marker: str | None = None,
    footer_marker: str | None = None,
) -> list[Block]:
    """Remove blocks before *header_marker* and after *footer_marker*.

    If a marker is ``None`` the corresponding boundary is not applied.
    Matching is case-sensitive and checks whether the marker string appears
    anywhere in the block's content.
    """
    start = 0
    end = len(blocks)

    if header_marker is not None:
        for i, block in enumerate(blocks):
            if header_marker in block.content:
                start = i + 1
                break

    if footer_marker is not None:
        for i in range(len(blocks) - 1, -1, -1):
            if footer_marker in blocks[i].content:
                end = i
                break

    return blocks[start:end]


def reindex_positions(blocks: list[Block]) -> list[Block]:
    """Renumber block positions 0..N-1."""
    return [replace(block, position=i) for i, block in enumerate(blocks)]
