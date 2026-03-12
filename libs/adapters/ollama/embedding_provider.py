"""Ollama embedding provider adapter.

Implements the ``EmbeddingProvider`` protocol using the native Ollama
``POST /api/embed`` endpoint.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from libs.adapters.config import OllamaConfig
from libs.contracts.chunks import Chunk
from libs.contracts.embeddings import ChunkEmbedding
from libs.embeddings.models import EmbeddingModelInfo

logger = logging.getLogger(__name__)


class OllamaEmbeddingProvider:
    """Embedding provider backed by an Ollama server.

    Satisfies the ``EmbeddingProvider`` protocol.  Uses the native
    ``POST /api/embed`` endpoint which accepts a list of input texts.
    """

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._client: httpx.Client | None = None
        self._model_info_cache: EmbeddingModelInfo | None = None

    def connect(self) -> None:
        """Create the HTTP client."""
        self._client = httpx.Client(
            base_url=self._config.base_url,
            timeout=self._config.timeout_s,
        )
        try:
            self._model_info_cache = self._fetch_model_info()
            logger.info(
                "Ollama embedding provider connected — %s (model=%s, dim=%d)",
                self._config.base_url,
                self._model_info_cache.model_id,
                self._model_info_cache.dimensions,
            )
        except Exception:
            logger.warning(
                "Ollama server not reachable at %s — model info will be fetched on first use",
                self._config.base_url,
            )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def health_check(self) -> bool:
        if self._client is None:
            return False
        try:
            resp = self._client.get("/api/tags")
            return resp.status_code == 200
        except Exception:
            return False

    def model_info(self) -> EmbeddingModelInfo:
        if self._model_info_cache is not None:
            return self._model_info_cache
        if self._client is None:
            raise RuntimeError(
                "OllamaEmbeddingProvider is not connected. Call connect() first."
            )
        self._model_info_cache = self._fetch_model_info()
        return self._model_info_cache

    def embed_chunks(self, chunks: list[Chunk]) -> list[ChunkEmbedding]:
        """Embed chunks via Ollama ``/api/embed`` endpoint."""
        if self._client is None:
            raise RuntimeError(
                "OllamaEmbeddingProvider is not connected. Call connect() first."
            )

        info = self.model_info()
        embeddings: list[ChunkEmbedding] = []
        texts = [chunk.content for chunk in chunks]

        # Ollama /api/embed supports batch input
        vectors = self._embed_texts(texts)

        for chunk, vector in zip(chunks, vectors, strict=True):
            embedding_id = _make_embedding_id(
                chunk.chunk_id, info.model_id, info.model_version,
            )
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

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama /api/embed endpoint."""
        assert self._client is not None
        resp = self._client.post(
            "/api/embed",
            json={"model": self._config.model_id, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"]

    def _fetch_model_info(self) -> EmbeddingModelInfo:
        """Probe the Ollama server for model metadata."""
        assert self._client is not None

        # Probe dimensions by embedding a single token
        probe = self._embed_texts(["hello"])
        dimensions = len(probe[0])

        # Get model version from /api/show
        model_version = "v1"
        try:
            resp = self._client.post(
                "/api/show",
                json={"model": self._config.model_id},
            )
            if resp.status_code == 200:
                show_data: dict[str, Any] = resp.json()
                model_version = show_data.get("modelinfo", {}).get(
                    "general.file_type", "v1",
                )
        except Exception:
            pass

        return EmbeddingModelInfo(
            model_id=self._config.model_id,
            model_version=model_version,
            dimensions=dimensions,
            max_tokens=self._config.num_ctx,
        )


def _make_embedding_id(chunk_id: str, model_id: str, model_version: str) -> str:
    raw = f"emb:{chunk_id}:{model_id}:{model_version}"
    short_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"emb-{short_hash}"
