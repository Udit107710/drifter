"""Abstract PDF parser base — concrete implementations plug in external libs."""

from __future__ import annotations

import abc
from dataclasses import replace
from datetime import UTC, datetime

from libs.contracts.documents import (
    Block,
    CanonicalDocument,
    RawDocument,
)
from libs.parsing.normalizers import normalize_whitespace, reindex_positions

_PARSER_VERSION = "pdf_base:1.0.0"


class PdfParserBase(abc.ABC):
    """Base class for PDF parsers.

    Subclasses implement ``_extract_blocks`` using a concrete library
    (e.g. Unstructured, Apache Tika).  The ``parse`` method handles
    normalisation and ``CanonicalDocument`` construction.
    """

    @abc.abstractmethod
    def _extract_blocks(self, raw_bytes: bytes) -> list[Block]:
        """Extract structural blocks from raw PDF bytes."""
        ...

    def supported_mime_types(self) -> list[str]:
        return ["application/pdf"]

    def parse(self, raw: RawDocument) -> CanonicalDocument:
        source_id = raw.source_ref.source_id
        version = raw.source_ref.version

        blocks = self._extract_blocks(raw.raw_bytes)

        # Normalise whitespace in each block's content
        blocks = [
            replace(b, content=normalize_whitespace(b.content)) for b in blocks
        ]
        blocks = reindex_positions(blocks)

        return CanonicalDocument(
            document_id=f"doc:{source_id}:{version}",
            source_ref=raw.source_ref,
            blocks=blocks,
            parser_version=_PARSER_VERSION,
            parsed_at=datetime.now(UTC),
        )
