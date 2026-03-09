# libs/evaluation/

First-class evaluation subsystem measuring retrieval and answer quality.

## Boundary

- **Consumes:** EvaluationCase (ground truth) + Retriever (implementation under test)
- **Produces:** EvaluationReport (per-query and aggregate metrics)
- **Rule:** Evaluation is not an afterthought. Changes affecting retrieval quality must include evaluation.

## Evaluator

```python
class RetrievalEvaluator:
    def __init__(self, config: EvaluationConfig | None = None, run_id: str = "") -> None: ...
    def evaluate(
        self,
        cases: list[EvaluationCase],
        retriever: Retriever,
        k_values: list[int] | None = None,
        relevance_grades: dict[str, dict[str, int]] | None = None,
    ) -> EvaluationReport: ...
```

### Retriever Protocol

```python
class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[str]: ...  # Returns chunk IDs
```

Any retrieval implementation can be evaluated by wrapping it in this protocol.

## Retrieval Metrics (`retrieval_metrics.py`)

| Metric | Function | Description |
|--------|----------|-------------|
| Recall@k | `recall_at_k()` | Fraction of relevant documents retrieved in top-k |
| Precision@k | `precision_at_k()` | Fraction of top-k that are relevant |
| MRR | `mrr()` | Mean Reciprocal Rank — position of first relevant result |
| NDCG@k | `ndcg_at_k()` | Normalized Discounted Cumulative Gain (supports graded relevance) |

## Answer Metrics (`answer_metrics.py`)

| Metric | Function | Description |
|--------|----------|-------------|
| Faithfulness | `evaluate_faithfulness()` | Are claims grounded in the provided evidence? |
| Citation accuracy | `evaluate_citation_accuracy()` | Do citations correctly reference supporting chunks? |
| Unsupported claims | `detect_unsupported_claims()` | Claims not backed by any evidence |

## Dataset Management (`dataset.py`)

```python
def load_dataset(path: Path) -> list[EvaluationCase]: ...
def save_dataset(cases: list[EvaluationCase], path: Path) -> None: ...
def create_seed_dataset() -> list[EvaluationCase]: ...
```

Datasets are JSON files containing `EvaluationCase` objects.

## Reporting (`report.py`)

```python
def save_json_report(report: EvaluationReport, path: Path) -> None: ...
def save_markdown_report(report: EvaluationReport, path: Path) -> None: ...
def report_to_dict(report: EvaluationReport) -> dict: ...
def generate_markdown_summary(report: EvaluationReport) -> str: ...
```

## Key Types

| Type | Purpose |
|------|---------|
| `EvaluationCase` | Query + expected answer + relevant chunk IDs |
| `EvaluationConfig` | retrieval_mode, embedding_model, reranker_id, chunking_strategy, k_values |
| `EvaluationReport` | run_id + config + query_results + stage_metrics + timestamp |
| `QueryResult` | case_id + query + retrieved_ids + relevant_ids + metrics |
| `StageMetrics` | stage + metric_means + metric_medians + query_count |

## Evaluation Strategy

Evaluate each retrieval mode separately:
1. Lexical only
2. Dense only
3. Hybrid (pre-rerank)
4. Hybrid + reranked

Recommended datasets: BEIR, HotpotQA, Natural Questions, plus custom local gold sets.

## Testing

Fully deterministic. Uses synthetic evaluation cases with known relevant chunk IDs.
