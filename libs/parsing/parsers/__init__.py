"""Concrete parser implementations."""

from libs.parsing.parsers.markdown import MarkdownParser
from libs.parsing.parsers.plain_text import PlainTextParser

__all__ = [
    "MarkdownParser",
    "PlainTextParser",
]
