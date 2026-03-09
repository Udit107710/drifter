"""Index lifecycle: version tracking, freshness, and metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class IndexVersion:
    """Metadata for a single index version."""

    index_id: str
    version: int
    model_id: str
    model_version: str
    chunk_count: int
    created_at: datetime
    is_active: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.index_id:
            raise ValueError("index_id must not be empty")
        if self.version < 1:
            raise ValueError("version must be >= 1")
        if self.chunk_count < 0:
            raise ValueError("chunk_count must be >= 0")


@dataclass(frozen=True)
class IndexFreshness:
    """Freshness tracking for an index."""

    index_id: str
    last_updated_at: datetime
    document_count: int
    stale_document_count: int = 0

    @property
    def freshness_ratio(self) -> float:
        if self.document_count == 0:
            return 1.0
        return 1.0 - (self.stale_document_count / self.document_count)


@runtime_checkable
class IndexRegistry(Protocol):
    """Registry for managing index versions and lifecycle."""

    def create_version(
        self, index_id: str, model_id: str, model_version: str, chunk_count: int
    ) -> IndexVersion: ...
    def get_active(self, index_id: str) -> IndexVersion | None: ...
    def activate(self, index_id: str, version: int) -> None: ...
    def list_versions(self, index_id: str) -> list[IndexVersion]: ...
    def get_freshness(self, index_id: str) -> IndexFreshness | None: ...


class MemoryIndexRegistry:
    """In-memory implementation of IndexRegistry for testing."""

    def __init__(self) -> None:
        self._versions: dict[str, list[IndexVersion]] = {}
        self._freshness: dict[str, IndexFreshness] = {}

    def create_version(
        self, index_id: str, model_id: str, model_version: str, chunk_count: int
    ) -> IndexVersion:
        from datetime import UTC

        versions = self._versions.setdefault(index_id, [])
        version_num = len(versions) + 1
        iv = IndexVersion(
            index_id=index_id,
            version=version_num,
            model_id=model_id,
            model_version=model_version,
            chunk_count=chunk_count,
            created_at=datetime.now(UTC),
        )
        versions.append(iv)
        return iv

    def get_active(self, index_id: str) -> IndexVersion | None:
        for v in reversed(self._versions.get(index_id, [])):
            if v.is_active:
                return v
        return None

    def activate(self, index_id: str, version: int) -> None:
        from dataclasses import replace

        versions = self._versions.get(index_id, [])
        updated: list[IndexVersion] = []
        for v in versions:
            if v.version == version:
                updated.append(replace(v, is_active=True))
            elif v.is_active:
                updated.append(replace(v, is_active=False))
            else:
                updated.append(v)
        self._versions[index_id] = updated

    def list_versions(self, index_id: str) -> list[IndexVersion]:
        return list(self._versions.get(index_id, []))

    def get_freshness(self, index_id: str) -> IndexFreshness | None:
        return self._freshness.get(index_id)

    def update_freshness(self, freshness: IndexFreshness) -> None:
        self._freshness[freshness.index_id] = freshness
