"""OpenRouter embedding provider adapter.

Implements the ``EmbeddingProvider`` protocol using the OpenAI-compatible
``POST /v1/embeddings`` endpoint on OpenRouter.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import UTC, datetime

import httpx

from libs.adapters.config import OpenRouterConfig
from libs.contracts.chunks import Chunk
from libs.contracts.embeddings import ChunkEmbedding
from libs.embeddings.models import EmbeddingModelInfo

logger = logging.getLogger(__name__)


class OpenRouterEmbeddingProvider:
    """Embedding provider backed by OpenRouter.

    Satisfies the ``EmbeddingProvider`` protocol.
    """

    def __init__(self, config: OpenRouterConfig) -> None:
        self._config = config
        self._client: httpx.Client | None = None
        self._model_info_cache: EmbeddingModelInfo | None = None

    # -- Lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Create the HTTP client."""
        self._client = httpx.Client(
            base_url=self._config.base_url,
            timeout=self._config.timeout_s,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
                "X-Title": self._config.app_name,
            },
        )
        try:
            self._model_info_cache = self._fetch_model_info()
            logger.info(
                "OpenRouter embedding provider connected — model=%s, dim=%d",
                self._model_info_cache.model_id,
                self._model_info_cache.dimensions,
            )
        except Exception:
            logger.warning(
                "OpenRouter not reachable — model info will be fetched on first use",
            )

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def health_check(self) -> bool:
        """Return True if the client is configured."""
        return self._client is not None

    # -- EmbeddingProvider protocol ------------------------------------------

    def model_info(self) -> EmbeddingModelInfo:
        if self._model_info_cache is not None:
            return self._model_info_cache
        if self._client is None:
            raise RuntimeError(
                "OpenRouterEmbeddingProvider is not connected. Call connect() first."
            )
        self._model_info_cache = self._fetch_model_info()
        return self._model_info_cache

    def embed_chunks(self, chunks: list[Chunk]) -> list[ChunkEmbedding]:
        """Embed chunks in batches via OpenRouter embeddings endpoint."""
        if self._client is None:
            raise RuntimeError(
                "OpenRouterEmbeddingProvider is not connected. Call connect() first."
            )

        info = self.model_info()
        embeddings: list[ChunkEmbedding] = []

        for i in range(0, len(chunks), self._config.max_batch_size):
            batch = chunks[i : i + self._config.max_batch_size]
            texts = [chunk.content for chunk in batch]
            vectors = self._embed_texts(texts)

            for chunk, vector in zip(batch, vectors, strict=True):
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

    # -- Internal ------------------------------------------------------------

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Call OpenRouter /v1/embeddings endpoint."""
        assert self._client is not None
        resp = self._client.post(
            "/v1/embeddings",
            json={
                "model": self._config.embedding_model,
                "input": texts,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    def _fetch_model_info(self) -> EmbeddingModelInfo:
        """Probe dimensions by embedding a single token."""
        probe = self._embed_texts(["hello"])
        dimensions = len(probe[0])

        return EmbeddingModelInfo(
            model_id=self._config.embedding_model,
            model_version="v1",
            dimensions=dimensions,
            max_tokens=8192,
        )


def _make_embedding_id(
    chunk_id: str, model_id: str, model_version: str,
) -> str:
    raw = f"emb:{chunk_id}:{model_id}:{model_version}"
    short_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"emb-{short_hash}"
