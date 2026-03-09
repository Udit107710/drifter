"""Qdrant vector store adapter.

Implements the ``VectorStore`` protocol using ``qdrant_client``.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from libs.adapters.config import QdrantConfig
from libs.contracts import Chunk, ChunkEmbedding, ChunkId, RetrievalCandidate, RetrievalQuery
from libs.contracts.chunks import ChunkLineage
from libs.contracts.common import RetrievalMethod

logger = logging.getLogger(__name__)

_DEFAULT_VECTOR_SIZE = 384  # sensible default (e.g. MiniLM)


# ── Chunk serialization helpers ──────────────────────────────────────


def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    """Serialize a Chunk to a JSON-safe dict."""
    lineage = {
        "source_id": chunk.lineage.source_id,
        "document_id": chunk.lineage.document_id,
        "block_ids": list(chunk.lineage.block_ids),
        "chunk_strategy": chunk.lineage.chunk_strategy,
        "parser_version": chunk.lineage.parser_version,
        "created_at": chunk.lineage.created_at.isoformat(),
        "schema_version": chunk.lineage.schema_version,
    }
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "source_id": chunk.source_id,
        "block_ids": list(chunk.block_ids),
        "content": chunk.content,
        "content_hash": chunk.content_hash,
        "token_count": chunk.token_count,
        "strategy": chunk.strategy,
        "byte_offset_start": chunk.byte_offset_start,
        "byte_offset_end": chunk.byte_offset_end,
        "lineage": lineage,
        "schema_version": chunk.schema_version,
        "metadata": dict(chunk.metadata),
        "acl": list(chunk.acl),
    }


def _dict_to_chunk(d: dict[str, Any]) -> Chunk:
    """Deserialize a dict back into a Chunk."""
    lin = d["lineage"]
    lineage = ChunkLineage(
        source_id=lin["source_id"],
        document_id=lin["document_id"],
        block_ids=lin["block_ids"],
        chunk_strategy=lin["chunk_strategy"],
        parser_version=lin["parser_version"],
        created_at=datetime.fromisoformat(lin["created_at"]),
        schema_version=lin.get("schema_version", 1),
    )
    return Chunk(
        chunk_id=d["chunk_id"],
        document_id=d["document_id"],
        source_id=d["source_id"],
        block_ids=d["block_ids"],
        content=d["content"],
        content_hash=d["content_hash"],
        token_count=d["token_count"],
        strategy=d["strategy"],
        byte_offset_start=d["byte_offset_start"],
        byte_offset_end=d["byte_offset_end"],
        lineage=lineage,
        schema_version=d.get("schema_version", 1),
        metadata=d.get("metadata", {}),
        acl=d.get("acl", []),
    )


# ── Point ID helpers ─────────────────────────────────────────────────


def _to_point_id(chunk_id: str) -> str:
    """Convert a chunk_id to a Qdrant-compatible UUID string.

    If the chunk_id is already a valid UUID, use it directly.
    Otherwise, generate a deterministic UUID5 from the chunk_id.
    """
    try:
        uuid.UUID(chunk_id)
        return chunk_id
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


# ── Qdrant filter helpers ────────────────────────────────────────────


def _build_qdrant_filter(filters: dict[str, Any]) -> models.Filter | None:
    """Convert a dict of field conditions to a Qdrant Filter.

    Supports simple equality conditions: ``{"source_id": "abc"}``
    """
    if not filters:
        return None

    conditions: list[models.FieldCondition] = []
    for key, value in filters.items():
        if isinstance(value, list):
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=value),
                )
            )
        else:
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )

    return models.Filter(must=conditions)


# ── QdrantVectorStore ────────────────────────────────────────────────


class QdrantVectorStore:
    """Qdrant-backed vector store implementing the ``VectorStore`` protocol."""

    def __init__(self, config: QdrantConfig) -> None:
        self._config = config
        self._store_id = f"qdrant:{config.collection_name}"
        self._client: QdrantClient | None = None

    # -- Protocol property ---------------------------------------------------

    @property
    def store_id(self) -> str:
        return self._store_id

    # -- Lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Create the QdrantClient. Collection is created lazily on first add."""
        cfg = self._config
        self._client = QdrantClient(
            host=cfg.host,
            port=cfg.port,
            grpc_port=cfg.grpc_port,
            api_key=cfg.api_key,
            https=cfg.use_tls,
            timeout=cfg.timeout_s,
        )
        logger.info("Connected to Qdrant at %s:%s", cfg.host, cfg.port)

    def close(self) -> None:
        """Close the Qdrant client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def health_check(self) -> bool:
        """Check if Qdrant is reachable."""
        if self._client is None:
            return False
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    # -- Data methods --------------------------------------------------------

    def add(self, embedding: ChunkEmbedding, chunk: Chunk) -> None:
        """Upsert a single point."""
        client = self._require_client()
        self._ensure_collection(vector_size=embedding.dimensions)
        point_id = _to_point_id(chunk.chunk_id)
        payload = self._build_payload(chunk)

        client.upsert(
            collection_name=self._config.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding.vector,
                    payload=payload,
                ),
            ],
        )

    def add_batch(self, embeddings: list[ChunkEmbedding], chunks: list[Chunk]) -> None:
        """Upsert a batch of points."""
        if not embeddings:
            return
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")

        client = self._require_client()
        self._ensure_collection(vector_size=embeddings[0].dimensions)

        points = [
            models.PointStruct(
                id=_to_point_id(chunk.chunk_id),
                vector=emb.vector,
                payload=self._build_payload(chunk),
            )
            for emb, chunk in zip(embeddings, chunks, strict=True)
        ]

        client.upsert(
            collection_name=self._config.collection_name,
            points=points,
        )

    def delete(self, chunk_ids: list[ChunkId]) -> int:
        """Delete points by chunk IDs. Returns the count deleted."""
        if not chunk_ids:
            return 0

        client = self._require_client()
        point_ids = [_to_point_id(cid) for cid in chunk_ids]

        # Get current count to determine how many were actually deleted.
        try:
            before = client.count(
                collection_name=self._config.collection_name,
                exact=True,
            ).count
        except (UnexpectedResponse, Exception):
            return 0

        client.delete(
            collection_name=self._config.collection_name,
            points_selector=models.PointIdsList(points=point_ids),
        )

        after = client.count(
            collection_name=self._config.collection_name,
            exact=True,
        ).count

        return before - after

    def search(
        self, query: RetrievalQuery, query_vector: list[float]
    ) -> list[RetrievalCandidate]:
        """kNN search returning RetrievalCandidate objects."""
        client = self._require_client()

        try:
            response = client.query_points(
                collection_name=self._config.collection_name,
                query=query_vector,
                limit=query.top_k,
                query_filter=_build_qdrant_filter(query.filters),
                with_payload=True,
            )
        except (UnexpectedResponse, Exception) as exc:
            # Collection may not exist yet.
            logger.debug("Search failed (collection may not exist): %s", exc)
            return []

        candidates: list[RetrievalCandidate] = []
        for hit in response.points:
            payload = hit.payload or {}
            chunk_json = payload.get("chunk_json")
            if chunk_json is None:
                logger.warning("Point %s has no chunk_json payload, skipping", hit.id)
                continue

            chunk_dict = json.loads(chunk_json) if isinstance(chunk_json, str) else chunk_json
            chunk = _dict_to_chunk(chunk_dict)

            candidates.append(
                RetrievalCandidate(
                    chunk=chunk,
                    score=hit.score,
                    retrieval_method=RetrievalMethod.DENSE,
                    store_id=self._store_id,
                )
            )

        return candidates

    def count(self) -> int:
        """Return the number of points in the collection."""
        client = self._require_client()
        try:
            result = client.count(
                collection_name=self._config.collection_name,
                exact=True,
            )
            return result.count
        except (UnexpectedResponse, Exception):
            return 0

    # -- Private helpers -----------------------------------------------------

    def _require_client(self) -> QdrantClient:
        if self._client is None:
            raise RuntimeError("QdrantVectorStore is not connected. Call connect() first.")
        return self._client

    def _ensure_collection(self, vector_size: int = _DEFAULT_VECTOR_SIZE) -> None:
        """Create the collection if it doesn't already exist.

        If the collection exists but has a different vector size,
        log a warning (it may have been created with a different model).
        """
        client = self._require_client()
        try:
            info = client.get_collection(self._config.collection_name)
            # Verify dimension matches
            existing_size = getattr(
                info.config.params.vectors, "size", None
            )
            if existing_size is not None and existing_size != vector_size:
                logger.warning(
                    "Collection %r exists with vector size %d, expected %d",
                    self._config.collection_name,
                    existing_size,
                    vector_size,
                )
        except (UnexpectedResponse, Exception):
            logger.info(
                "Creating collection %r with vector size %d",
                self._config.collection_name,
                vector_size,
            )
            client.create_collection(
                collection_name=self._config.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    @staticmethod
    def _build_payload(chunk: Chunk) -> dict[str, Any]:
        """Build the Qdrant point payload from a Chunk."""
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "source_id": chunk.source_id,
            "content": chunk.content,
            "content_hash": chunk.content_hash,
            "token_count": chunk.token_count,
            "strategy": chunk.strategy,
            "metadata": dict(chunk.metadata),
            "chunk_json": json.dumps(_chunk_to_dict(chunk)),
        }
