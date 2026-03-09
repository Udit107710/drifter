"""Evaluation dataset management: seed datasets and loaders."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from libs.contracts.evaluation import EvaluationCase


def create_seed_dataset() -> list[EvaluationCase]:
    """Create a small deterministic seed dataset for local testing.

    These cases cover common retrieval scenarios:
    - Single relevant document
    - Multiple relevant documents
    - Different topic areas
    """
    return [
        EvaluationCase(
            case_id="seed-001",
            query="What is machine learning?",
            expected_answer=(
                "Machine learning is a subset of AI that enables"
                " systems to learn from data."
            ),
            relevant_chunk_ids=["chunk-ml-001", "chunk-ml-002"],
            metadata={"topic": "ml", "difficulty": "easy"},
        ),
        EvaluationCase(
            case_id="seed-002",
            query="How does backpropagation work?",
            expected_answer=(
                "Backpropagation computes gradients by applying"
                " the chain rule backwards through the network."
            ),
            relevant_chunk_ids=["chunk-bp-001", "chunk-bp-002", "chunk-bp-003"],
            metadata={"topic": "ml", "difficulty": "medium"},
        ),
        EvaluationCase(
            case_id="seed-003",
            query="What is a vector database?",
            expected_answer=(
                "A vector database stores and indexes"
                " high-dimensional vectors for similarity search."
            ),
            relevant_chunk_ids=["chunk-vdb-001"],
            metadata={"topic": "databases", "difficulty": "easy"},
        ),
        EvaluationCase(
            case_id="seed-004",
            query="Explain the transformer architecture",
            expected_answer=(
                "Transformers use self-attention to process"
                " sequences in parallel, replacing recurrence."
            ),
            relevant_chunk_ids=["chunk-tf-001", "chunk-tf-002"],
            metadata={"topic": "ml", "difficulty": "hard"},
        ),
        EvaluationCase(
            case_id="seed-005",
            query="What is retrieval augmented generation?",
            expected_answer=(
                "RAG combines retrieval of relevant documents"
                " with LLM generation for grounded answers."
            ),
            relevant_chunk_ids=["chunk-rag-001", "chunk-rag-002", "chunk-rag-003"],
            metadata={"topic": "rag", "difficulty": "medium"},
        ),
    ]


def save_dataset(cases: list[EvaluationCase], path: Path) -> None:
    """Save evaluation cases to a JSON file."""
    data: list[dict[str, Any]] = [asdict(c) for c in cases]
    path.write_text(json.dumps(data, indent=2, default=str))


def load_dataset(path: Path) -> list[EvaluationCase]:
    """Load evaluation cases from a JSON file."""
    data: list[dict[str, Any]] = json.loads(path.read_text())
    return [EvaluationCase(**item) for item in data]
