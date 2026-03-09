"""Indexing subsystem models: result types for indexing operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from libs.contracts.common import RunId
from libs.embeddings.models import EmbeddingModelInfo


class IndexingOutcome(Enum):
    """Outcome of an indexing run."""

    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class ErrorClassification(Enum):
    """Classification of an indexing error."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"


@dataclass(frozen=True)
class ChunkError:
    """Per-chunk error tracking for batch operations."""

    chunk_id: str
    stage: str  # "embedding", "chunk_store", "vector_index", "lexical_index"
    error: str
    classification: ErrorClassification


@dataclass(frozen=True)
class IndexingResult:
    """Summary of a completed indexing run."""

    run_id: RunId
    chunks_received: int
    chunks_embedded: int
    chunks_indexed_vector: int
    chunks_indexed_lexical: int
    outcome: IndexingOutcome
    model_info: EmbeddingModelInfo
    completed_at: datetime
    errors: list[str] = field(default_factory=list)
    chunk_errors: list[ChunkError] = field(default_factory=list)
