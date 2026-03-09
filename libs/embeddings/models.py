"""Embedding model metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EmbeddingModelInfo:
    """Immutable descriptor for an embedding model's identity and capabilities."""

    model_id: str
    model_version: str
    dimensions: int
    max_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if not self.model_version:
            raise ValueError("model_version must not be empty")
        if self.dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
