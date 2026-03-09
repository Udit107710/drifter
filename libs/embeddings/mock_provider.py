"""Deterministic embedding provider for testing."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from libs.contracts.chunks import Chunk
from libs.contracts.embeddings import ChunkEmbedding
from libs.embeddings.models import EmbeddingModelInfo


class DeterministicEmbeddingProvider:
    """Produces repeatable embeddings by hashing chunk content.

    Same input always produces the same vector, making tests fully deterministic.
    """

    def __init__(
        self,
        model_id: str = "deterministic-test",
        model_version: str = "1.0",
        dimensions: int = 64,
    ) -> None:
        self._model_id = model_id
        self._model_version = model_version
        self._dimensions = dimensions

    def model_info(self) -> EmbeddingModelInfo:
        return EmbeddingModelInfo(
            model_id=self._model_id,
            model_version=self._model_version,
            dimensions=self._dimensions,
            max_tokens=8192,
        )

    def embed_chunks(self, chunks: list[Chunk]) -> list[ChunkEmbedding]:
        return [self._embed_single(chunk) for chunk in chunks]

    def _embed_single(self, chunk: Chunk) -> ChunkEmbedding:
        vector = self._make_vector(chunk.content_hash)
        embedding_id = self._make_embedding_id(chunk.chunk_id)
        return ChunkEmbedding(
            embedding_id=embedding_id,
            chunk_id=chunk.chunk_id,
            vector=vector,
            model_id=self._model_id,
            model_version=self._model_version,
            dimensions=self._dimensions,
            created_at=datetime.now(UTC),
        )

    def _make_vector(self, content_hash: str) -> list[float]:
        """Hash content_hash + model identity and unpack bytes into floats in [0, 1]."""
        seed = f"{content_hash}:{self._model_id}:{self._model_version}"
        # Generate enough bytes: each float needs 1 byte from digest, but SHA256 gives 32 bytes.
        # To get self._dimensions floats, repeatedly hash to extend.
        floats: list[float] = []
        counter = 0
        while len(floats) < self._dimensions:
            h = hashlib.sha256(f"{seed}:{counter}".encode()).digest()
            for byte in h:
                if len(floats) >= self._dimensions:
                    break
                floats.append(byte / 255.0)
            counter += 1
        return floats

    def _make_embedding_id(self, chunk_id: str) -> str:
        raw = f"emb:{chunk_id}:{self._model_id}:{self._model_version}"
        short_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"emb-{short_hash}"
