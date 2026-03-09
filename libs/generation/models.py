"""Generation models: request, validation, outcome, and result envelope."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from libs.contracts.common import ChunkId, TraceId
from libs.contracts.generation import GeneratedAnswer


class GenerationOutcome(Enum):
    SUCCESS = "success"
    EMPTY_CONTEXT = "empty_context"
    GENERATION_FAILED = "generation_failed"
    VALIDATION_FAILED = "validation_failed"


@dataclass(frozen=True)
class GenerationRequest:
    """Rendered prompt ready for LLM consumption."""

    rendered_prompt: str
    system_prompt: str
    context_chunk_ids: list[ChunkId]
    query: str
    trace_id: TraceId
    token_budget: int

    def __post_init__(self) -> None:
        if not self.rendered_prompt:
            raise ValueError("rendered_prompt must not be empty")
        if not self.query:
            raise ValueError("query must not be empty")
        if not self.trace_id:
            raise ValueError("trace_id must not be empty")
        if self.token_budget < 1:
            raise ValueError("token_budget must be >= 1")


@dataclass(frozen=True)
class ValidationResult:
    """Result of citation validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    orphaned_citations: list[ChunkId] = field(default_factory=list)
    uncited_chunks: list[ChunkId] = field(default_factory=list)


@dataclass(frozen=True)
class GenerationResult:
    """Envelope wrapping the generation output with diagnostics."""

    answer: GeneratedAnswer | None
    outcome: GenerationOutcome
    generator_id: str
    total_latency_ms: float
    completed_at: datetime
    validation: ValidationResult | None = None
    errors: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
