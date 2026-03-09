"""Shared domain models for the Drifter RAG pipeline.

This package defines the typed contracts that flow between subsystems:

    SourceDocumentRef → RawDocument → CanonicalDocument → Block → Chunk
    → ChunkEmbedding → RetrievalCandidate → RankedCandidate → ContextPack
    → GeneratedAnswer

Every model carries metadata, lineage, and version information.
No subsystem logic lives here — only data definitions.
"""

from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.common import (
    BlockId,
    BlockType,
    ChunkId,
    DocumentId,
    EmbeddingId,
    RetrievalMethod,
    RunId,
    SelectionReason,
    SourceId,
    TraceId,
)
from libs.contracts.context import ContextItem, ContextPack
from libs.contracts.documents import (
    Block,
    CanonicalDocument,
    RawDocument,
    SourceDocumentRef,
)
from libs.contracts.embeddings import ChunkEmbedding
from libs.contracts.evaluation import EvaluationCase, EvaluationResult
from libs.contracts.generation import Citation, GeneratedAnswer, TokenUsage
from libs.contracts.retrieval import RankedCandidate, RetrievalCandidate, RetrievalQuery

__all__ = [
    "Block",
    "BlockId",
    "BlockType",
    "CanonicalDocument",
    "Chunk",
    "ChunkEmbedding",
    "ChunkId",
    "ChunkLineage",
    "Citation",
    "ContextItem",
    "ContextPack",
    "DocumentId",
    "EmbeddingId",
    "EvaluationCase",
    "EvaluationResult",
    "GeneratedAnswer",
    "RankedCandidate",
    "RawDocument",
    "RetrievalCandidate",
    "RetrievalMethod",
    "RetrievalQuery",
    "RunId",
    "SelectionReason",
    "SourceDocumentRef",
    "SourceId",
    "TokenUsage",
    "TraceId",
]
