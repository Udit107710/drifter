"""OpenSearch lexical store adapter — BM25 search via opensearch-py.

Implements the ``LexicalStore`` protocol using OpenSearch's built-in
BM25 scoring on a ``text`` field.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime
from typing import Any

from opensearchpy import NotFoundError, OpenSearch, helpers

from libs.adapters.config import OpenSearchConfig
from libs.contracts import Chunk, ChunkId, RetrievalCandidate, RetrievalQuery
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


class OpenSearchLexicalStore:
    """OpenSearch BM25 lexical retrieval adapter.

    Satisfies the ``LexicalStore`` protocol.
    """

    def __init__(self, config: OpenSearchConfig) -> None:
        self._config = config
        self._store_id = f"opensearch-lexical:{config.index_prefix}"
        self._index_name = f"{config.index_prefix}-lexical"
        self._client: OpenSearch | None = None

    # -- Protocol property ---------------------------------------------------

    @property
    def store_id(self) -> str:
        return self._store_id

    # -- Lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Connect to the OpenSearch cluster and ensure the index exists."""
        self._client = OpenSearch(
            hosts=self._config.hosts,
            http_auth=(self._config.username, self._config.password),
            use_ssl=self._config.use_ssl,
            verify_certs=False,
            timeout=self._config.timeout_s,
        )
        self._ensure_index()

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

    def add(self, chunk: Chunk) -> None:
        """Index a single chunk."""
        client = self._require_client()
        doc = self._build_document(chunk)
        client.index(
            index=self._index_name,
            id=chunk.chunk_id,
            body=doc,
            refresh="wait_for",
        )

    def add_batch(self, chunks: list[Chunk]) -> None:
        """Bulk-index a list of chunks."""
        if not chunks:
            return
        client = self._require_client()
        actions = [
            {
                "_index": self._index_name,
                "_id": chunk.chunk_id,
                "_source": self._build_document(chunk),
            }
            for chunk in chunks
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

    def search(self, query: RetrievalQuery) -> list[RetrievalCandidate]:
        """Execute a BM25 search against the content field."""
        client = self._require_client()

        must_clause: dict[str, Any] = {
            "match": {"content": query.normalized_query},
        }

        filter_clauses = self._build_filters(query.filters)

        if filter_clauses:
            body: dict[str, Any] = {
                "size": query.top_k,
                "query": {
                    "bool": {
                        "must": [must_clause],
                        "filter": filter_clauses,
                    }
                },
            }
        else:
            body = {
                "size": query.top_k,
                "query": must_clause,
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
                    retrieval_method=RetrievalMethod.LEXICAL,
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
                "OpenSearchLexicalStore is not connected. Call connect() first."
            )
        return self._client

    def _ensure_index(self) -> None:
        """Create the lexical index if it does not already exist."""
        client = self._require_client()
        if client.indices.exists(index=self._index_name):
            return
        mapping: dict[str, Any] = {
            "mappings": {
                "properties": {
                    "content": {"type": "text", "analyzer": "standard"},
                    "chunk_id": {"type": "keyword"},
                    "document_id": {"type": "keyword"},
                    "source_id": {"type": "keyword"},
                    "chunk_json": {"type": "object", "enabled": False},
                }
            }
        }
        client.indices.create(index=self._index_name, body=mapping)
        logger.info("Created lexical index %s", self._index_name)

    @staticmethod
    def _build_document(chunk: Chunk) -> dict[str, Any]:
        """Build the OpenSearch document body from a Chunk."""
        chunk_dict = json.loads(_chunk_to_json(chunk))
        return {
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
