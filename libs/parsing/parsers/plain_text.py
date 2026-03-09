"""Plain-text parser — splits on blank lines into PARAGRAPH blocks."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from libs.contracts.common import BlockType
from libs.contracts.documents import (
    Block,
    CanonicalDocument,
    RawDocument,
)
from libs.parsing.normalizers import normalize_whitespace

_log = logging.getLogger(__name__)

_PARSER_VERSION = "plain_text:1.0.0"


class PlainTextParser:
    """Converts ``text/plain`` documents into paragraph blocks."""

    def supported_mime_types(self) -> list[str]:
        return ["text/plain"]

    def parse(self, raw: RawDocument) -> CanonicalDocument:
        source_id = raw.source_ref.source_id
        version = raw.source_ref.version

        # Decode with UTF-8, fallback to latin-1
        try:
            text = raw.raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.raw_bytes.decode("latin-1")
            _log.warning(
                "UTF-8 decode failed for source=%s, falling back to latin-1",
                raw.source_ref.source_id,
            )

        # Split on blank lines
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        blocks: list[Block] = []
        for i, para in enumerate(paragraphs):
            content = normalize_whitespace(para.replace("\n", " "))
            blocks.append(
                Block(
                    block_id=f"{source_id}:blk:{i}",
                    block_type=BlockType.PARAGRAPH,
                    content=content,
                    position=i,
                )
            )

        return CanonicalDocument(
            document_id=f"doc:{source_id}:{version}",
            source_ref=raw.source_ref,
            blocks=blocks,
            parser_version=_PARSER_VERSION,
            parsed_at=datetime.now(UTC),
        )
