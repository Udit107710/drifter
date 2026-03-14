"""Generation contracts: answers, citations, and token accounting."""

from __future__ import annotations

from dataclasses import dataclass

from libs.contracts.common import ChunkId, SourceId, TraceId


@dataclass(frozen=True)
class TokenUsage:
    """Token accounting for a single LLM call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    thinking_tokens: int = 0

    def __post_init__(self) -> None:
        if self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be >= 0")
        if self.completion_tokens < 0:
            raise ValueError("completion_tokens must be >= 0")
        if self.thinking_tokens < 0:
            raise ValueError("thinking_tokens must be >= 0")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("total_tokens must equal prompt_tokens + completion_tokens")


@dataclass(frozen=True)
class Citation:
    """A reference from a generated claim to the supporting chunk."""

    claim: str
    chunk_id: ChunkId
    chunk_content: str
    source_id: SourceId
    confidence: float
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.claim:
            raise ValueError("claim must not be empty")
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")
        if not self.chunk_content:
            raise ValueError("chunk_content must not be empty")
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class GeneratedAnswer:
    """The final output of the generation stage."""

    answer: str
    citations: list[Citation]
    model_id: str
    token_usage: TokenUsage
    trace_id: TraceId
    thinking: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.answer:
            raise ValueError("answer must not be empty")
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if not self.trace_id:
            raise ValueError("trace_id must not be empty")
