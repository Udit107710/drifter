"""Apache Tika PDF parser — stub adapter."""

from __future__ import annotations

from libs.adapters.config import TikaConfig
from libs.contracts.documents import Block
from libs.parsing.parsers.pdf import PdfParserBase


class TikaPdfParser(PdfParserBase):
    """PDF parser backed by Apache Tika.

    This is a stub adapter.  Install ``tika-python`` and implement the
    extraction logic to use this parser in production.
    """

    def __init__(self, config: TikaConfig) -> None:
        self._config = config

    # -- lifecycle -------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the Tika service."""

    def close(self) -> None:
        """Release resources."""

    def health_check(self) -> bool:
        """Return ``False`` — stub is not connected to a real service."""
        return False

    # -- parsing ---------------------------------------------------------------

    def _extract_blocks(self, raw_bytes: bytes) -> list[Block]:
        raise NotImplementedError(
            "Install tika-python and implement TikaPdfParser "
            "to use Apache Tika"
        )
