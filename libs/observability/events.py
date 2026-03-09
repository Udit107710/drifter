"""Structured pipeline events for observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PipelineEvent:
    """Base event emitted by a pipeline stage."""
    stage: str
    trace_id: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionEvent(PipelineEvent):
    """Event from the ingestion stage."""
    source_count: int = 0
    new_sources: int = 0
    skipped_sources: int = 0
    error_count: int = 0


@dataclass(frozen=True)
class ParsingEvent(PipelineEvent):
    """Event from the parsing stage."""
    document_id: str = ""
    block_count: int = 0
    parser_type: str = ""
    content_bytes: int = 0


@dataclass(frozen=True)
class ChunkingEvent(PipelineEvent):
    """Event from the chunking stage."""
    document_id: str = ""
    input_block_count: int = 0
    output_chunk_count: int = 0
    strategy: str = ""
    avg_chunk_tokens: float = 0.0


@dataclass(frozen=True)
class EmbeddingEvent(PipelineEvent):
    """Event from the embedding stage."""
    chunk_count: int = 0
    model_id: str = ""
    dimensions: int = 0
    batch_size: int = 0


@dataclass(frozen=True)
class IndexingEvent(PipelineEvent):
    """Event from the indexing stage."""
    chunk_count: int = 0
    success_count: int = 0
    error_count: int = 0
    outcome: str = ""


@dataclass(frozen=True)
class RetrievalEvent(PipelineEvent):
    """Event from the retrieval stage."""
    mode: str = ""
    dense_count: int = 0
    lexical_count: int = 0
    fused_count: int = 0
    outcome: str = ""


@dataclass(frozen=True)
class RerankingEvent(PipelineEvent):
    """Event from the reranking stage."""
    input_count: int = 0
    output_count: int = 0
    reranker_id: str = ""
    outcome: str = ""


@dataclass(frozen=True)
class ContextBuildEvent(PipelineEvent):
    """Event from the context building stage."""
    input_count: int = 0
    selected_count: int = 0
    excluded_count: int = 0
    tokens_used: int = 0
    token_budget: int = 0
    outcome: str = ""


@dataclass(frozen=True)
class GenerationEvent(PipelineEvent):
    """Event from the generation stage."""
    model_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    citation_count: int = 0
    outcome: str = ""
