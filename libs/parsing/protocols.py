"""Parser protocol — the contract every document parser must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from libs.contracts.documents import CanonicalDocument, RawDocument


@runtime_checkable
class DocumentParser(Protocol):
    """Converts a RawDocument into a CanonicalDocument with preserved structure."""

    def parse(self, raw: RawDocument) -> CanonicalDocument:
        """Parse raw bytes into a structured canonical document."""
        ...

    def supported_mime_types(self) -> list[str]:
        """Return the MIME types this parser can handle."""
        ...
