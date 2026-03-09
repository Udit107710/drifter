"""TEI query embedder adapter.

Implements the ``QueryEmbedder`` protocol using the Text Embeddings
Inference (TEI) HTTP API.  Calls ``POST /embed`` with a single query string.
"""

from __future__ import annotations

import logging

import httpx

from libs.adapters.config import TeiConfig

logger = logging.getLogger(__name__)


class TeiQueryEmbedder:
    """Query embedder backed by a TEI server.

    Satisfies the ``QueryEmbedder`` protocol (``embed_query(text) -> list[float]``).
    """

    def __init__(self, config: TeiConfig) -> None:
        self._config = config
        self._client: httpx.Client | None = None

    # -- Lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Create the HTTP client."""
        self._client = httpx.Client(
            base_url=self._config.base_url,
            timeout=self._config.timeout_s,
        )
        if self.health_check():
            logger.info("TEI query embedder connected — %s", self._config.base_url)
        else:
            logger.warning(
                "TEI query embedder created but server not reachable at %s",
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

    # -- QueryEmbedder protocol ----------------------------------------------

    def embed_query(self, text: str) -> list[float]:
        """Embed a query string into a dense vector."""
        if self._client is None:
            raise RuntimeError("TeiQueryEmbedder is not connected. Call connect() first.")

        resp = self._client.post("/embed", json={"inputs": [text]})
        resp.raise_for_status()
        vectors: list[list[float]] = resp.json()
        return vectors[0]
