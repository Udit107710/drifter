"""Deterministic chunk ID and content hash generation."""

from __future__ import annotations

import hashlib

from libs.contracts.common import ChunkId


def generate_chunk_id(
    document_id: str,
    strategy: str,
    content: str,
    sequence_index: int,
) -> ChunkId:
    """Generate a deterministic chunk ID.

    The ID is a SHA-256 digest of the document ID, strategy name,
    sequence index, and content, truncated to 24 hex characters and
    prefixed with ``chk:``.
    """
    payload = f"{document_id}|{strategy}|{sequence_index}|{content}"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return f"chk:{digest}"


def content_hash(content: str) -> str:
    """Return a prefixed SHA-256 hash of *content*."""
    return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
