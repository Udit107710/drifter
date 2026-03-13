"""vLLM embedding provider adapter.

Implements the ``EmbeddingProvider`` protocol using the vLLM
``/v1/embeddings`` endpoint (OpenAI-compatible).
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from libs.adapters.config import VllmConfig, VllmEmbeddingsConfig
from libs.contracts.chunks import Chunk
from libs.contracts.embeddings import ChunkEmbedding
from libs.embeddings.models import EmbeddingModelInfo

logger = logging.getLogger(__name__)


class VllmEmbeddingProvider:
    """Embedding provider backed by a vLLM server.

    Satisfies the ``EmbeddingProvider`` protocol.
    """

    def __init__(self, config: VllmConfig | VllmEmbeddingsConfig) -> None:
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
                "vLLM embedding provider connected — %s (model=%s, dim=%d)",
                self._config.base_url,
                self._model_info_cache.model_id,
                self._model_info_cache.dimensions,
            )
        except Exception:
            logger.warning(
                "vLLM server not reachable at %s — model info will be fetched on first use",
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
            resp = self._client.get("/health")
            return resp.status_code == 200
        except Exception:
            return False

    def model_info(self) -> EmbeddingModelInfo:
        if self._model_info_cache is not None:
            return self._model_info_cache
        if self._client is None:
            raise RuntimeError("VllmEmbeddingProvider is not connected. Call connect() first.")
        self._model_info_cache = self._fetch_model_info()
        return self._model_info_cache

    def embed_chunks(self, chunks: list[Chunk]) -> list[ChunkEmbedding]:
        """Embed chunks via vLLM ``/v1/embeddings`` endpoint."""
        if self._client is None:
            raise RuntimeError("VllmEmbeddingProvider is not connected. Call connect() first.")

        info = self.model_info()
        embeddings: list[ChunkEmbedding] = []

        # vLLM supports batch embedding natively
        texts = [chunk.content for chunk in chunks]
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
        """Call vLLM /v1/embeddings endpoint."""
        assert self._client is not None
        resp = self._client.post(
            "/v1/embeddings",
            json={"input": texts, "model": self._config.model_id},
        )
        resp.raise_for_status()
        data = resp.json()
        # Sort by index to ensure order matches input
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]

    def _fetch_model_info(self) -> EmbeddingModelInfo:
        """Probe the vLLM server for model metadata."""
        assert self._client is not None

        # Probe dimensions by embedding a single token
        probe = self._embed_texts(["hello"])
        dimensions = len(probe[0])

        # Try to get model info from /v1/models
        model_id = self._config.model_id
        model_version = "v1"
        try:
            resp = self._client.get("/v1/models")
            if resp.status_code == 200:
                models: dict[str, Any] = resp.json()
                for m in models.get("data", []):
                    if m.get("id") == model_id:
                        model_version = m.get("owned_by", "v1")
                        break
        except Exception:
            pass

        return EmbeddingModelInfo(
            model_id=model_id,
            model_version=model_version,
            dimensions=dimensions,
            max_tokens=8192,
        )


def _make_embedding_id(chunk_id: str, model_id: str, model_version: str) -> str:
    raw = f"emb:{chunk_id}:{model_id}:{model_version}"
    short_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"emb-{short_hash}"
