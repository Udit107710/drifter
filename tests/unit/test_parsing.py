"""Tests for the parsing subsystem: parsers and normalizers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from libs.contracts.common import BlockType
from libs.contracts.documents import Block, RawDocument, SourceDocumentRef
from libs.parsing.normalizers import (
    collapse_blank_lines,
    normalize_whitespace,
    reindex_positions,
    strip_header_footer,
)
from libs.parsing.parsers.markdown import MarkdownParser
from libs.parsing.parsers.plain_text import PlainTextParser
from libs.parsing.protocols import DocumentParser

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _make_source_ref(**overrides: object) -> SourceDocumentRef:
    defaults: dict[str, object] = {
        "source_id": "src-001",
        "uri": "https://example.com/doc.txt",
        "content_hash": "sha256:abc123",
        "fetched_at": datetime(2025, 1, 1, tzinfo=UTC),
        "version": 1,
    }
    defaults.update(overrides)
    return SourceDocumentRef(**defaults)  # type: ignore[arg-type]


def _make_raw(content: bytes, mime_type: str = "text/plain") -> RawDocument:
    return RawDocument(
        source_ref=_make_source_ref(),
        raw_bytes=content,
        mime_type=mime_type,
        size_bytes=len(content),
    )


# ── Protocol conformance ────────────────────────────────────────────


class TestProtocol:
    def test_plain_text_parser_satisfies_protocol(self) -> None:
        assert isinstance(PlainTextParser(), DocumentParser)

    def test_markdown_parser_satisfies_protocol(self) -> None:
        assert isinstance(MarkdownParser(), DocumentParser)


# ── PlainTextParser ─────────────────────────────────────────────────


class TestPlainTextParser:
    def test_parse_simple_text(self) -> None:
        parser = PlainTextParser()
        raw = _make_raw(b"Hello world.")
        doc = parser.parse(raw)
        assert len(doc.blocks) == 1
        assert doc.blocks[0].block_type == BlockType.PARAGRAPH
        assert doc.blocks[0].content == "Hello world."

    def test_multiple_paragraphs(self) -> None:
        parser = PlainTextParser()
        raw = _make_raw(b"First para.\n\nSecond para.\n\nThird para.")
        doc = parser.parse(raw)
        assert len(doc.blocks) == 3
        assert doc.blocks[0].position == 0
        assert doc.blocks[1].position == 1
        assert doc.blocks[2].position == 2

    def test_whitespace_normalized(self) -> None:
        parser = PlainTextParser()
        raw = _make_raw(b"  lots   of   spaces  ")
        doc = parser.parse(raw)
        assert doc.blocks[0].content == "lots of spaces"

    def test_block_ids_contain_source_id(self) -> None:
        parser = PlainTextParser()
        raw = _make_raw(b"Some text.")
        doc = parser.parse(raw)
        assert "src-001" in doc.blocks[0].block_id

    def test_parser_version(self) -> None:
        parser = PlainTextParser()
        raw = _make_raw(b"Text.")
        doc = parser.parse(raw)
        assert doc.parser_version == "plain_text:1.0.0"

    def test_document_id_format(self) -> None:
        parser = PlainTextParser()
        raw = _make_raw(b"Text.")
        doc = parser.parse(raw)
        assert doc.document_id == "doc:src-001:1"

    def test_multiline_paragraph_merged(self) -> None:
        parser = PlainTextParser()
        raw = _make_raw(b"Line one\nline two")
        doc = parser.parse(raw)
        assert len(doc.blocks) == 1
        assert doc.blocks[0].content == "Line one line two"

    def test_fixture_file(self) -> None:
        parser = PlainTextParser()
        content = (FIXTURES / "sample.txt").read_bytes()
        raw = _make_raw(content)
        doc = parser.parse(raw)
        assert len(doc.blocks) == 3
        assert doc.blocks[0].content == "Title paragraph here."
        assert "Second paragraph" in doc.blocks[1].content

    def test_latin1_fallback(self) -> None:
        parser = PlainTextParser()
        # \xe9 is 'é' in latin-1 but invalid standalone UTF-8
        raw = _make_raw(b"caf\xe9")
        doc = parser.parse(raw)
        assert doc.blocks[0].content == "café"

    def test_plain_text_utf8_fallback_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """failure_mode_analysis: decode fallback should be logged."""
        import logging

        parser = PlainTextParser()
        raw = _make_raw(b"\xff\xfeHello")
        with caplog.at_level(logging.WARNING):
            parser.parse(raw)
        assert "UTF-8 decode failed" in caplog.text


# ── MarkdownParser ──────────────────────────────────────────────────


class TestMarkdownParser:
    def test_headings(self) -> None:
        parser = MarkdownParser()
        raw = _make_raw(b"# Title\n\n## Subtitle", "text/markdown")
        doc = parser.parse(raw)
        headings = [b for b in doc.blocks if b.block_type == BlockType.HEADING]
        assert len(headings) == 2
        assert headings[0].level == 1
        assert headings[0].content == "Title"
        assert headings[1].level == 2

    def test_paragraphs(self) -> None:
        parser = MarkdownParser()
        raw = _make_raw(b"# H1\n\nSome paragraph.", "text/markdown")
        doc = parser.parse(raw)
        paras = [b for b in doc.blocks if b.block_type == BlockType.PARAGRAPH]
        assert len(paras) == 1
        assert paras[0].content == "Some paragraph."

    def test_fenced_code_with_language(self) -> None:
        parser = MarkdownParser()
        raw = _make_raw(b"```python\nprint('hi')\n```", "text/markdown")
        doc = parser.parse(raw)
        code = [b for b in doc.blocks if b.block_type == BlockType.CODE]
        assert len(code) == 1
        assert code[0].content == "print('hi')"
        assert code[0].metadata == {"language": "python"}

    def test_blockquote_merged(self) -> None:
        parser = MarkdownParser()
        raw = _make_raw(b"> line one\n> line two", "text/markdown")
        doc = parser.parse(raw)
        quotes = [b for b in doc.blocks if b.block_type == BlockType.QUOTE]
        assert len(quotes) == 1
        assert "line one" in quotes[0].content
        assert "line two" in quotes[0].content

    def test_unordered_list(self) -> None:
        parser = MarkdownParser()
        raw = _make_raw(b"- item one\n- item two", "text/markdown")
        doc = parser.parse(raw)
        lists = [b for b in doc.blocks if b.block_type == BlockType.LIST]
        assert len(lists) == 1
        assert "item one" in lists[0].content
        assert "item two" in lists[0].content

    def test_ordered_list(self) -> None:
        parser = MarkdownParser()
        raw = _make_raw(b"1. first\n2. second", "text/markdown")
        doc = parser.parse(raw)
        lists = [b for b in doc.blocks if b.block_type == BlockType.LIST]
        assert len(lists) == 1

    def test_table(self) -> None:
        parser = MarkdownParser()
        raw = _make_raw(b"| A | B |\n|---|---|\n| 1 | 2 |", "text/markdown")
        doc = parser.parse(raw)
        tables = [b for b in doc.blocks if b.block_type == BlockType.TABLE]
        assert len(tables) == 1

    def test_mixed_document_ordering(self) -> None:
        parser = MarkdownParser()
        content = (FIXTURES / "sample.md").read_bytes()
        raw = _make_raw(content, "text/markdown")
        doc = parser.parse(raw)

        types = [b.block_type for b in doc.blocks]
        assert types[0] == BlockType.HEADING  # # Main Title
        assert BlockType.PARAGRAPH in types
        assert BlockType.CODE in types
        assert BlockType.LIST in types
        assert BlockType.QUOTE in types
        assert BlockType.TABLE in types

        # Positions are sequential
        for i, block in enumerate(doc.blocks):
            assert block.position == i

    def test_empty_lines_no_empty_blocks(self) -> None:
        parser = MarkdownParser()
        raw = _make_raw(b"# H1\n\n\n\nParagraph.", "text/markdown")
        doc = parser.parse(raw)
        for block in doc.blocks:
            assert block.content.strip() != ""

    def test_parser_version(self) -> None:
        parser = MarkdownParser()
        raw = _make_raw(b"# Title", "text/markdown")
        doc = parser.parse(raw)
        assert doc.parser_version == "markdown:1.0.0"

    def test_supported_mime_types(self) -> None:
        parser = MarkdownParser()
        assert "text/markdown" in parser.supported_mime_types()
        assert "text/x-markdown" in parser.supported_mime_types()


# ── Normalizers ─────────────────────────────────────────────────────


class TestNormalizers:
    def test_normalize_whitespace(self) -> None:
        assert normalize_whitespace("  hello   world  ") == "hello world"
        assert normalize_whitespace("\thello\t\tworld\t") == "hello world"

    def test_collapse_blank_lines(self) -> None:
        assert collapse_blank_lines("a\n\n\n\nb") == "a\n\nb"
        assert collapse_blank_lines("a\n\nb") == "a\n\nb"  # no change

    def test_strip_header_footer(self) -> None:
        blocks = [
            Block(block_id="b0", block_type=BlockType.PARAGRAPH, content="HEADER", position=0),
            Block(block_id="b1", block_type=BlockType.PARAGRAPH, content="Body text", position=1),
            Block(block_id="b2", block_type=BlockType.PARAGRAPH, content="FOOTER", position=2),
        ]
        result = strip_header_footer(blocks, header_marker="HEADER", footer_marker="FOOTER")
        assert len(result) == 1
        assert result[0].content == "Body text"

    def test_strip_header_only(self) -> None:
        blocks = [
            Block(block_id="b0", block_type=BlockType.PARAGRAPH, content="HEADER", position=0),
            Block(block_id="b1", block_type=BlockType.PARAGRAPH, content="Body", position=1),
        ]
        result = strip_header_footer(blocks, header_marker="HEADER")
        assert len(result) == 1
        assert result[0].content == "Body"

    def test_reindex_positions(self) -> None:
        blocks = [
            Block(block_id="b0", block_type=BlockType.PARAGRAPH, content="A", position=5),
            Block(block_id="b1", block_type=BlockType.PARAGRAPH, content="B", position=10),
        ]
        result = reindex_positions(blocks)
        assert result[0].position == 0
        assert result[1].position == 1
        # Original block_ids preserved
        assert result[0].block_id == "b0"
