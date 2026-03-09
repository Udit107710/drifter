"""Markdown parser — regex-based, no external dependencies.

Uses a line-by-line state machine to emit structured blocks for headings,
fenced code, blockquotes, lists, tables, and paragraphs.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from enum import Enum, auto

from libs.contracts.common import BlockType
from libs.contracts.documents import (
    Block,
    CanonicalDocument,
    RawDocument,
)
from libs.parsing.normalizers import normalize_whitespace

_log = logging.getLogger(__name__)

_PARSER_VERSION = "markdown:1.0.0"

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
_FENCE_RE = re.compile(r"^```(\w*)")
_QUOTE_RE = re.compile(r"^>\s?(.*)")
_UL_RE = re.compile(r"^[-*+]\s+(.*)")
_OL_RE = re.compile(r"^\d+\.\s+(.*)")
_TABLE_RE = re.compile(r"^\|")


class _State(Enum):
    IDLE = auto()
    CODE = auto()
    QUOTE = auto()
    LIST = auto()
    TABLE = auto()
    PARAGRAPH = auto()


class MarkdownParser:
    """Converts ``text/markdown`` documents into typed blocks."""

    def supported_mime_types(self) -> list[str]:
        return ["text/markdown", "text/x-markdown"]

    def parse(self, raw: RawDocument) -> CanonicalDocument:
        source_id = raw.source_ref.source_id
        version = raw.source_ref.version

        try:
            text = raw.raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.raw_bytes.decode("latin-1")
            _log.warning(
                "UTF-8 decode failed for source=%s, falling back to latin-1",
                raw.source_ref.source_id,
            )

        blocks: list[Block] = []
        lines = text.split("\n")

        state = _State.IDLE
        buf: list[str] = []
        code_lang: str = ""
        pos = 0
        skipped = 0

        def _emit(block_type: BlockType, content: str, **kwargs: object) -> None:
            nonlocal pos, skipped
            content = content.strip()
            if not content:
                skipped += 1
                return
            blocks.append(
                Block(
                    block_id=f"{source_id}:blk:{pos}",
                    block_type=block_type,
                    content=content,
                    position=pos,
                    **kwargs,  # type: ignore[arg-type]
                )
            )
            pos += 1

        def _flush() -> None:
            nonlocal state, buf, code_lang
            if not buf:
                state = _State.IDLE
                return
            joined = "\n".join(buf)
            if state == _State.CODE:
                meta = {"language": code_lang} if code_lang else {}
                _emit(BlockType.CODE, joined, metadata=meta)
            elif state == _State.QUOTE:
                _emit(BlockType.QUOTE, normalize_whitespace(joined.replace("\n", " ")))
            elif state == _State.LIST:
                _emit(BlockType.LIST, joined)
            elif state == _State.TABLE:
                _emit(BlockType.TABLE, joined)
            elif state == _State.PARAGRAPH:
                _emit(BlockType.PARAGRAPH, normalize_whitespace(joined.replace("\n", " ")))
            buf = []
            code_lang = ""
            state = _State.IDLE

        for line in lines:
            # ── fenced code toggle ──
            fence_m = _FENCE_RE.match(line)
            if fence_m and state == _State.CODE:
                _flush()
                continue
            if fence_m and line.startswith("```"):
                _flush()
                state = _State.CODE
                code_lang = fence_m.group(1)
                continue
            if state == _State.CODE:
                buf.append(line)
                continue

            # ── heading (single-line) ──
            heading_m = _HEADING_RE.match(line)
            if heading_m:
                _flush()
                level = len(heading_m.group(1))
                _emit(BlockType.HEADING, heading_m.group(2), level=level)
                continue

            # ── blockquote ──
            quote_m = _QUOTE_RE.match(line)
            if quote_m:
                if state != _State.QUOTE:
                    _flush()
                    state = _State.QUOTE
                buf.append(quote_m.group(1))
                continue

            # ── table ──
            if _TABLE_RE.match(line):
                if state != _State.TABLE:
                    _flush()
                    state = _State.TABLE
                buf.append(line)
                continue

            # ── unordered / ordered list ──
            ul_m = _UL_RE.match(line)
            ol_m = _OL_RE.match(line) if not ul_m else None
            if ul_m or ol_m:
                if state != _State.LIST:
                    _flush()
                    state = _State.LIST
                buf.append(line)
                continue

            # ── blank line ──
            if not line.strip():
                if state in (_State.QUOTE, _State.LIST, _State.TABLE) or state == _State.PARAGRAPH:
                    _flush()
                continue

            # ── paragraph text ──
            if state != _State.PARAGRAPH:
                _flush()
                state = _State.PARAGRAPH
            buf.append(line)

        _flush()

        if skipped:
            _log.debug(
                "Skipped %d empty blocks for source=%s",
                skipped, source_id,
            )

        return CanonicalDocument(
            document_id=f"doc:{source_id}:{version}",
            source_ref=raw.source_ref,
            blocks=blocks,
            parser_version=_PARSER_VERSION,
            parsed_at=datetime.now(UTC),
        )
