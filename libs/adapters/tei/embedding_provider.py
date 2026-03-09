"""TEI embedding provider adapter.

Implements the ``EmbeddingProvider`` protocol using the Text Embeddings
Inference (TEI) HTTP API.  Calls ``POST /embed`` to generate dense vectors.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from libs.adapters.config import TeiConfig
from libs.contracts.chunks import Chunk
from libs.contracts.embeddings import ChunkEmbedding
from libs.embeddings.models import EmbeddingModelInfo

logger = logging.getLogger(__name__)


class TeiEmbeddingProvider:
    """Embedding provider backed by a TEI server.

    Satisfies the ``EmbeddingProvider`` protocol.
    """

    def __init__(self, config: TeiConfig) -> None:
        self._config = config
        self._client: httpx.Client | None = None
        self._model_info_cache: EmbeddingModelInfo | None = None

    # -- Lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Create the HTTP client. Model info is fetched lazily on first use."""
        self._client = httpx.Client(
            base_url=self._config.base_url,
            timeout=self._config.timeout_s,
        )
        # Try to fetch model info eagerly; log but don't fail if server is down.
        try:
            self._model_info_cache = self._fetch_model_info()
            logger.info(
                "TEI embedding provider connected — %s (model=%s, dim=%d)",
                self._config.base_url,
                self._model_info_cache.model_id,
                self._model_info_cache.dimensions,
            )
        except Exception:
            logger.warning(
                "TEI server not reachable at %s — model info will be fetched on first use",
                self._config.base_url,
            )

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def health_check(self) -> bool:
        """Return True if the server is reachable."""
        if self._client is None:
            return False
        try:
            resp = self._client.get("/health")
            return resp.status_code == 200
        except Exception:
            return False

    # -- EmbeddingProvider protocol ------------------------------------------

    def model_info(self) -> EmbeddingModelInfo:
        if self._model_info_cache is not None:
            return self._model_info_cache
        if self._client is None:
            raise RuntimeError("TeiEmbeddingProvider is not connected. Call connect() first.")
        self._model_info_cache = self._fetch_model_info()
        return self._model_info_cache

    def embed_chunks(self, chunks: list[Chunk]) -> list[ChunkEmbedding]:
        """Embed chunks in batches via TEI ``/embed`` endpoint."""
        if self._client is None:
            raise RuntimeError("TeiEmbeddingProvider is not connected. Call connect() first.")

        info = self.model_info()
        embeddings: list[ChunkEmbedding] = []

        for i in range(0, len(chunks), self._config.max_batch_size):
            batch = chunks[i : i + self._config.max_batch_size]
            texts = [chunk.content for chunk in batch]
            vectors = self._embed_texts(texts)

            for chunk, vector in zip(batch, vectors, strict=True):
                embedding_id = _make_embedding_id(chunk.chunk_id, info.model_id, info.model_version)
                embeddings.append(
                    ChunkEmbedding(
                        embedding_id=embedding_id,
                        chunk_id=chunk.chunk_id,
                        vector=vector,
                        model_id=info.model_id,
                        model_version=info.model_version,
                        dimensions=info.dimensions,
                        created_at=datetime.now(UTC),
                    )
                )

        return embeddings

    # -- Internal ------------------------------------------------------------

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Call TEI /embed endpoint."""
        assert self._client is not None
        resp = self._client.post("/embed", json={"inputs": texts})
        resp.raise_for_status()
        return resp.json()

    def _fetch_model_info(self) -> EmbeddingModelInfo:
        """Probe the TEI server for model metadata."""
        assert self._client is not None

        # TEI exposes model info at GET /info
        resp = self._client.get("/info")
        resp.raise_for_status()
        info: dict[str, Any] = resp.json()

        model_id = self._config.model_id or info.get("model_id", "unknown")
        model_version = self._config.model_version or info.get("model_sha", "unknown")

        # Probe dimensions by embedding a single token
        probe = self._embed_texts(["hello"])
        dimensions = len(probe[0])

        max_tokens = info.get("max_input_length", 8192)

        return EmbeddingModelInfo(
            model_id=model_id,
            model_version=model_version,
            dimensions=dimensions,
            max_tokens=max_tokens,
        )


def _make_embedding_id(chunk_id: str, model_id: str, model_version: str) -> str:
    raw = f"emb:{chunk_id}:{model_id}:{model_version}"
    short_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"emb-{short_hash}"
