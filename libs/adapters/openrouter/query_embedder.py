"""OpenRouter query embedder adapter.

Implements the ``QueryEmbedder`` protocol using the OpenAI-compatible
``POST /v1/embeddings`` endpoint on OpenRouter.
"""

from __future__ import annotations

import logging

import httpx

from libs.adapters.config import OpenRouterConfig

logger = logging.getLogger(__name__)


class OpenRouterQueryEmbedder:
    """Query embedder backed by OpenRouter.

    Satisfies the ``QueryEmbedder`` protocol (``embed_query(text) -> list[float]``).
    """

    def __init__(self, config: OpenRouterConfig) -> None:
        self._config = config
        self._client: httpx.Client | None = None

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
        logger.info(
            "OpenRouter query embedder configured — model=%s",
            self._config.embedding_model,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def health_check(self) -> bool:
        """Return True if the client is configured."""
        return self._client is not None

    # -- QueryEmbedder protocol ----------------------------------------------

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string into a dense vector."""
        if self._client is None:
            raise RuntimeError(
                "OpenRouterQueryEmbedder is not connected. Call connect() first."
            )

        resp = self._client.post(
            "/v1/embeddings",
            json={
                "model": self._config.embedding_model,
                "input": text,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
