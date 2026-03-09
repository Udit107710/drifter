"""Deterministic query embedder for testing."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from libs.contracts.chunks import Chunk, ChunkLineage
from libs.embeddings.mock_provider import DeterministicEmbeddingProvider


class DeterministicQueryEmbedder:
    """Wraps DeterministicEmbeddingProvider to embed query strings.

    Creates a synthetic Chunk from the query text, embeds it using the
    existing mock provider, and returns the vector. For testing only.
    """

    def __init__(
        self,
        model_id: str = "deterministic-test",
        model_version: str = "1.0",
        dimensions: int = 64,
    ) -> None:
        self._provider = DeterministicEmbeddingProvider(
            model_id=model_id,
            model_version=model_version,
            dimensions=dimensions,
        )

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string into a dense vector."""
        chunk = self._make_synthetic_chunk(text)
        embeddings = self._provider.embed_chunks([chunk])
        return embeddings[0].vector

    @staticmethod
    def _make_synthetic_chunk(text: str) -> Chunk:
        """Build a minimal Chunk from query text for embedding."""
        content_hash = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
        now = datetime.now(UTC)
        return Chunk(
            chunk_id="query-synthetic",
            document_id="query-doc",
            source_id="query-source",
            block_ids=["query-block"],
            content=text,
            content_hash=content_hash,
            token_count=max(1, len(text.split())),
            strategy="query",
            byte_offset_start=0,
            byte_offset_end=max(1, len(text.encode())),
            lineage=ChunkLineage(
                source_id="query-source",
                document_id="query-doc",
                block_ids=["query-block"],
                chunk_strategy="query",
                parser_version="query:1.0",
                created_at=now,
            ),
        )
