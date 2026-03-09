"""Sample experiment configurations demonstrating one-variable-at-a-time design."""

from __future__ import annotations

from libs.evaluation.models import EvaluationConfig
from libs.experiments.models import ExperimentConfig


def chunk_size_experiment(
    dataset_path: str,
    artifact_dir: str,
    chunk_size: int,
) -> ExperimentConfig:
    """Create an experiment varying chunk size while holding other variables fixed."""
    return ExperimentConfig(
        name=f"chunk-size-{chunk_size}",
        hypothesis=(
            f"Chunk size {chunk_size} tokens improves recall by providing "
            "better semantic boundaries for retrieval."
        ),
        eval_config=EvaluationConfig(
            retrieval_mode="dense",
            embedding_model="default",
            reranker_id="none",
            chunking_strategy=f"fixed-{chunk_size}",
        ),
        dataset_path=dataset_path,
        artifact_dir=artifact_dir,
        tags=["chunk-size", "ablation"],
        extra={"chunk_size": chunk_size},
    )


def retrieval_mode_experiment(
    dataset_path: str,
    artifact_dir: str,
    mode: str,
) -> ExperimentConfig:
    """Create an experiment varying retrieval mode (dense, lexical, hybrid)."""
    return ExperimentConfig(
        name=f"retrieval-mode-{mode}",
        hypothesis=(
            f"{mode.title()} retrieval provides better precision-recall "
            "trade-off for this dataset."
        ),
        eval_config=EvaluationConfig(
            retrieval_mode=mode,
            embedding_model="default",
            reranker_id="none",
            chunking_strategy="fixed-512",
        ),
        dataset_path=dataset_path,
        artifact_dir=artifact_dir,
        tags=["retrieval-mode", "ablation"],
        extra={"retrieval_mode": mode},
    )


def reranker_experiment(
    dataset_path: str,
    artifact_dir: str,
    reranker_id: str,
) -> ExperimentConfig:
    """Create an experiment varying reranker while holding retrieval mode fixed."""
    return ExperimentConfig(
        name=f"reranker-{reranker_id}",
        hypothesis=(
            f"Reranker '{reranker_id}' improves NDCG over the baseline "
            "retrieval ordering."
        ),
        eval_config=EvaluationConfig(
            retrieval_mode="dense",
            embedding_model="default",
            reranker_id=reranker_id,
            chunking_strategy="fixed-512",
        ),
        dataset_path=dataset_path,
        artifact_dir=artifact_dir,
        tags=["reranker", "ablation"],
        extra={"reranker_id": reranker_id},
    )
