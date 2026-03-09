"""Local filesystem source connector."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from libs.ingestion.models import FetchResult, SourceConfig


class LocalFilesystemConnector:
    """Reads files from local filesystem paths."""

    def fetch(self, config: SourceConfig) -> FetchResult | None:
        """Read a file and return its contents. Returns None if the path doesn't exist."""
        path = Path(config.uri)
        if not path.exists():
            return None

        raw_bytes = path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(path))
        return FetchResult(
            raw_bytes=raw_bytes,
            mime_type=mime_type or "application/octet-stream",
            size_bytes=len(raw_bytes),
        )

    def list_source_ids(self, config: SourceConfig) -> list[str]:
        """List files at the URI. If directory, list recursively; if file, return [uri]."""
        path = Path(config.uri)
        if not path.exists():
            return []
        if path.is_file():
            return [config.uri]
        return [str(p) for p in sorted(path.rglob("*")) if p.is_file()]
