"""Parsing subsystem.

Responsibilities:
- Convert raw document bytes into CanonicalDocument with structured Blocks
- Preserve document structure (headings, paragraphs, tables, code, lists)
- Never flatten documents to plain text

Boundary: consumes SourceDocumentRef, produces CanonicalDocument.
"""

from libs.parsing.normalizers import (
    collapse_blank_lines,
    normalize_whitespace,
    reindex_positions,
    strip_header_footer,
)
from libs.parsing.parsers.markdown import MarkdownParser
from libs.parsing.parsers.plain_text import PlainTextParser
from libs.parsing.protocols import DocumentParser

__all__ = [
    "DocumentParser",
    "MarkdownParser",
    "PlainTextParser",
    "collapse_blank_lines",
    "normalize_whitespace",
    "reindex_positions",
    "strip_header_footer",
]
