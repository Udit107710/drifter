"""Unstructured PDF parser — stub adapter."""

from __future__ import annotations

from libs.adapters.config import UnstructuredConfig
from libs.contracts.documents import Block
from libs.parsing.parsers.pdf import PdfParserBase


class UnstructuredPdfParser(PdfParserBase):
    """PDF parser backed by Unstructured.

    This is a stub adapter.  Install the ``unstructured`` package and
    implement the extraction logic to use this parser in production.
    """

    def __init__(self, config: UnstructuredConfig) -> None:
        self._config = config

    # -- lifecycle -------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the Unstructured service."""

    def close(self) -> None:
        """Release resources."""

    def health_check(self) -> bool:
        """Return ``False`` — stub is not connected to a real service."""
        return False

    # -- parsing ---------------------------------------------------------------

    def _extract_blocks(self, raw_bytes: bytes) -> list[Block]:
        raise NotImplementedError(
            "Install unstructured and implement UnstructuredPdfParser "
            "to use Unstructured"
        )
