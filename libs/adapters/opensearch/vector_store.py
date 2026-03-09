"""OpenSearch vector store adapter — kNN search via opensearch-py.

Implements the ``VectorStore`` protocol using OpenSearch's kNN plugin
with HNSW indexing and cosine similarity.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime
from typing import Any

from opensearchpy import NotFoundError, OpenSearch, helpers

from libs.adapters.config import OpenSearchConfig
from libs.contracts import Chunk, ChunkEmbedding, ChunkId, RetrievalCandidate, RetrievalQuery
from libs.contracts.chunks import ChunkLineage
from libs.contracts.common import RetrievalMethod

logger = logging.getLogger(__name__)


def _chunk_to_json(chunk: Chunk) -> str:
    """Serialize a frozen Chunk dataclass to JSON with datetime handling."""

    def _default(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(dataclasses.asdict(chunk), default=_default)


def _chunk_from_json(raw: str) -> Chunk:
    """Deserialize a Chunk from its JSON representation."""
    data = json.loads(raw)
    lineage_data = data.pop("lineage")
    lineage_data["created_at"] = datetime.fromisoformat(lineage_data["created_at"])
    lineage = ChunkLineage(**lineage_data)
    return Chunk(lineage=lineage, **data)


class OpenSearchVectorStore:
    """OpenSearch kNN vector retrieval adapter.

    Satisfies the ``VectorStore`` protocol.

    The vector index is created lazily on the first ``add`` or ``add_batch``
    call because the embedding dimension must be known at index creation time.
    """

    def __init__(self, config: OpenSearchConfig) -> None:
        self._config = config
        self._store_id = f"opensearch-vector:{config.index_prefix}"
        self._index_name = f"{config.index_prefix}-vectors"
        self._client: OpenSearch | None = None
        self._dimensions: int | None = None
        self._index_created: bool = False

    # -- Protocol property ---------------------------------------------------

    @property
    def store_id(self) -> str:
        return self._store_id

    # -- Lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Connect to the OpenSearch cluster."""
        self._client = OpenSearch(
            hosts=self._config.hosts,
            http_auth=(self._config.username, self._config.password),
            use_ssl=self._config.use_ssl,
            verify_certs=False,
            timeout=self._config.timeout_s,
        )
        # Check if index already exists (e.g. from a previous run).
        if self._client.indices.exists(index=self._index_name):
            self._index_created = True

    def close(self) -> None:
        """Close the OpenSearch connection."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def health_check(self) -> bool:
        """Return True if the cluster is reachable."""
        if self._client is None:
            return False
        try:
            info = self._client.cluster.health()
            return info.get("status") in ("green", "yellow")
        except Exception:
            return False

    # -- Data methods --------------------------------------------------------

    def add(self, embedding: ChunkEmbedding, chunk: Chunk) -> None:
        """Index a single chunk with its embedding vector."""
        client = self._require_client()
        self._ensure_index(embedding.dimensions)
        doc = self._build_document(embedding, chunk)
        client.index(
            index=self._index_name,
            id=chunk.chunk_id,
            body=doc,
            refresh="wait_for",
        )

    def add_batch(self, embeddings: list[ChunkEmbedding], chunks: list[Chunk]) -> None:
        """Bulk-index chunks with their embedding vectors."""
        if not embeddings:
            return
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"embeddings ({len(embeddings)}) and chunks ({len(chunks)}) "
                "must have the same length"
            )
        client = self._require_client()
        self._ensure_index(embeddings[0].dimensions)
        actions = [
            {
                "_index": self._index_name,
                "_id": chunk.chunk_id,
                "_source": self._build_document(emb, chunk),
            }
            for emb, chunk in zip(embeddings, chunks, strict=True)
        ]
        helpers.bulk(client, actions, refresh="wait_for")

    def delete(self, chunk_ids: list[ChunkId]) -> int:
        """Delete chunks by their IDs. Returns the number actually deleted."""
        if not chunk_ids:
            return 0
        client = self._require_client()
        deleted = 0
        for cid in chunk_ids:
            try:
                resp = client.delete(
                    index=self._index_name,
                    id=cid,
                    refresh="wait_for",
                )
                if resp.get("result") == "deleted":
                    deleted += 1
            except NotFoundError:
                pass
        return deleted

    def search(
        self, query: RetrievalQuery, query_vector: list[float]
    ) -> list[RetrievalCandidate]:
        """Execute a kNN vector search."""
        client = self._require_client()

        body: dict[str, Any] = {
            "size": query.top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": query.top_k,
                    }
                }
            },
        }

        # Apply metadata filters if present.
        if query.filters:
            filter_clauses = self._build_filters(query.filters)
            body["query"] = {
                "bool": {
                    "must": [body["query"]],
                    "filter": filter_clauses,
                }
            }

        try:
            resp = client.search(index=self._index_name, body=body)
        except NotFoundError:
            return []

        candidates: list[RetrievalCandidate] = []
        for hit in resp["hits"]["hits"]:
            source = hit["_source"]
            chunk = _chunk_from_json(json.dumps(source["chunk_json"]))
            candidates.append(
                RetrievalCandidate(
                    chunk=chunk,
                    score=float(hit["_score"]),
                    retrieval_method=RetrievalMethod.DENSE,
                    store_id=self._store_id,
                )
            )
        return candidates

    def count(self) -> int:
        """Return the number of documents in the index."""
        client = self._require_client()
        try:
            resp = client.count(index=self._index_name)
            return int(resp["count"])
        except NotFoundError:
            return 0

    # -- Internal helpers ----------------------------------------------------

    def _require_client(self) -> OpenSearch:
        if self._client is None:
            raise RuntimeError(
                "OpenSearchVectorStore is not connected. Call connect() first."
            )
        return self._client

    def _ensure_index(self, dimensions: int) -> None:
        """Lazily create the vector index once the dimension is known."""
        if self._index_created:
            return
        client = self._require_client()
        if client.indices.exists(index=self._index_name):
            self._index_created = True
            self._dimensions = dimensions
            return

        self._dimensions = dimensions
        mapping: dict[str, Any] = {
            "settings": {
                "index": {"knn": True},
            },
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dimensions,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "lucene",
                        },
                    },
                    "content": {"type": "text"},
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "source_id": {"type": "keyword"},
                    "chunk_json": {"type": "object", "enabled": False},
                }
            },
        }
        client.indices.create(index=self._index_name, body=mapping)
        self._index_created = True
        logger.info(
            "Created vector index %s (dimensions=%d)", self._index_name, dimensions
        )

    @staticmethod
    def _build_document(embedding: ChunkEmbedding, chunk: Chunk) -> dict[str, Any]:
        """Build the OpenSearch document body from an embedding and its chunk."""
        chunk_dict = json.loads(_chunk_to_json(chunk))
        return {
            "embedding": embedding.vector,
            "content": chunk.content,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "source_id": chunk.source_id,
            "chunk_json": chunk_dict,
        }

    @staticmethod
    def _build_filters(filters: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert query filters to OpenSearch bool-filter term clauses."""
        clauses: list[dict[str, Any]] = []
        for key, value in filters.items():
            clauses.append({"term": {key: value}})
        return clauses
