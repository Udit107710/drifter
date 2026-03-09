"""Evaluation subsystem.

Responsibilities:
- Calculate retrieval metrics (Recall@k, MRR, NDCG)
- Calculate answer quality metrics (faithfulness, citation accuracy)
- Manage evaluation datasets
- Generate evaluation reports

Evaluation is a first-class feature, not an afterthought.
"""

from libs.evaluation.answer_metrics import (
    CitationAccuracyResult,
    FaithfulnessResult,
    detect_unsupported_claims,
    evaluate_citation_accuracy,
    evaluate_faithfulness,
)
from libs.evaluation.dataset import create_seed_dataset, load_dataset, save_dataset
from libs.evaluation.evaluator import RetrievalEvaluator, Retriever
from libs.evaluation.models import (
    EvaluationConfig,
    EvaluationReport,
    QueryResult,
    StageMetrics,
)
from libs.evaluation.report import (
    generate_markdown_summary,
    report_to_dict,
    save_json_report,
    save_markdown_report,
)
from libs.evaluation.retrieval_metrics import (
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "CitationAccuracyResult",
    "EvaluationConfig",
    "EvaluationReport",
    "FaithfulnessResult",
    "QueryResult",
    "RetrievalEvaluator",
    "Retriever",
    "StageMetrics",
    "create_seed_dataset",
    "detect_unsupported_claims",
    "evaluate_citation_accuracy",
    "evaluate_faithfulness",
    "generate_markdown_summary",
    "load_dataset",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "report_to_dict",
    "save_dataset",
    "save_json_report",
    "save_markdown_report",
]
