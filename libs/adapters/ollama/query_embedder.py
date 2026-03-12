"""Ollama query embedder adapter.

Implements the ``QueryEmbedder`` protocol using the native Ollama
``POST /api/embed`` endpoint.
"""

from __future__ import annotations

import logging

import httpx

from libs.adapters.config import OllamaConfig

logger = logging.getLogger(__name__)


class OllamaQueryEmbedder:
    """Query embedder backed by an Ollama server.

    Satisfies the ``QueryEmbedder`` protocol.
    """

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._client: httpx.Client | None = None

    def connect(self) -> None:
        self._client = httpx.Client(
            base_url=self._config.base_url,
            timeout=self._config.timeout_s,
        )
        if self.health_check():
            logger.info("Ollama query embedder connected — %s", self._config.base_url)
        else:
            logger.warning(
                "Ollama query embedder created but server not reachable at %s",
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

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string into a dense vector."""
        if self._client is None:
            raise RuntimeError(
                "OllamaQueryEmbedder is not connected. Call connect() first."
            )

        resp = self._client.post(
            "/api/embed",
            json={"model": self._config.model_id, "input": [text]},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["embeddings"][0]
