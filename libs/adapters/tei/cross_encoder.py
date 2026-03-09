"""TEI cross-encoder reranker adapter.

Implements the ``Reranker`` protocol using the Text Embeddings Inference
(TEI) HTTP API.  Calls ``POST /rerank`` with query-document pairs.
"""

from __future__ import annotations

import logging

import httpx

from libs.adapters.config import TeiConfig
from libs.contracts.retrieval import RankedCandidate, RetrievalCandidate, RetrievalQuery

logger = logging.getLogger(__name__)


class TeiCrossEncoderReranker:
    """Cross-encoder reranker backed by a TEI server.

    Satisfies the ``Reranker`` protocol.

    TEI's ``/rerank`` endpoint scores (query, document) pairs using a
    cross-encoder model and returns relevance scores.
    """

    def __init__(self, config: TeiConfig, model_name: str) -> None:
        self._config = config
        self._model_name = model_name
        self._reranker_id = f"tei-cross-encoder:{model_name}"
        self._client: httpx.Client | None = None

    # -- Protocol property ---------------------------------------------------

    @property
    def reranker_id(self) -> str:
        return self._reranker_id

    # -- Lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Create the HTTP client."""
        self._client = httpx.Client(
            base_url=self._config.base_url,
            timeout=self._config.timeout_s,
        )
        if self.health_check():
            logger.info("TEI cross-encoder reranker connected — %s", self._config.base_url)
        else:
            logger.warning(
                "TEI cross-encoder created but server not reachable at %s",
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

    # -- Reranker protocol ---------------------------------------------------

    def rerank(
        self,
        candidates: list[RetrievalCandidate],
        query: RetrievalQuery,
    ) -> list[RankedCandidate]:
        """Score all candidates against the query using TEI cross-encoder."""
        if self._client is None:
            raise RuntimeError("TeiCrossEncoderReranker is not connected. Call connect() first.")

        if not candidates:
            return []

        # TEI /rerank expects: {"query": "...", "texts": ["...", ...]}
        texts = [c.chunk.content for c in candidates]
        resp = self._client.post(
            "/rerank",
            json={
                "query": query.normalized_query,
                "texts": texts,
                "return_text": False,
            },
        )
        resp.raise_for_status()

        # TEI returns: [{"index": 0, "score": 0.95}, ...] sorted by score desc
        results: list[dict] = resp.json()

        # Build index → score mapping
        score_by_index: dict[int, float] = {}
        for entry in results:
            score_by_index[entry["index"]] = float(entry["score"])

        # Pair candidates with scores, sort by score descending
        scored: list[tuple[float, RetrievalCandidate]] = []
        for i, candidate in enumerate(candidates):
            score = score_by_index.get(i, 0.0)
            scored.append((score, candidate))

        scored.sort(key=lambda t: t[0], reverse=True)

        return [
            RankedCandidate(
                candidate=candidate,
                rank=rank + 1,
                rerank_score=score,
                reranker_id=self._reranker_id,
            )
            for rank, (score, candidate) in enumerate(scored)
        ]
