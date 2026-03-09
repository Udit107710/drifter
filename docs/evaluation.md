# Evaluation Subsystem

## Architecture Role

Evaluation is a **first-class subsystem**, not an afterthought. It measures retrieval quality and answer quality at each stage of the RAG pipeline independently, not just end-to-end.

```
EvaluationCase → Retriever → RetrievalEvaluator → EvaluationReport → Markdown / JSON
```

The evaluation loop:

1. A **seed dataset** of `EvaluationCase` objects defines queries, expected answers, and ground-truth relevant chunk IDs.
2. A **Retriever** (any object satisfying the `Retriever` protocol) returns chunk IDs for each query.
3. The **RetrievalEvaluator** computes per-query and aggregate retrieval metrics.
4. An **EvaluationReport** collects all results with configuration metadata for reproducibility.
5. **ReportWriter** serializes reports to JSON (machine-readable) and Markdown (human-readable).

This design keeps evaluation decoupled from the retrieval implementation. Any retriever -- dense, lexical, hybrid, reranked -- can be evaluated with the same dataset and evaluator.

## Retrieval Metrics

All retrieval metrics compare the ordered list of retrieved chunk IDs against a ground-truth set of relevant chunk IDs.

### Recall@k

```
Recall@k = |retrieved[:k] ∩ relevant| / |relevant|
```

Measures **coverage**: what fraction of the relevant documents did the retriever find in the top k? High recall means the system is not missing important information. Range: [0.0, 1.0]. Returns 0.0 if the relevant set is empty.

### Precision@k

```
Precision@k = |retrieved[:k] ∩ relevant| / k
```

Measures **noise**: what fraction of the top-k results are actually relevant? High precision means the retriever is not polluting the context with irrelevant chunks. Range: [0.0, 1.0]. Returns 0.0 if k <= 0.

### MRR (Mean Reciprocal Rank)

```
MRR = 1 / rank_of_first_relevant
```

Measures **user experience**: how quickly does the user encounter a relevant result? An MRR of 1.0 means the first result is always relevant. Range: (0.0, 1.0] or 0.0 if no relevant document is found.

### NDCG@k (Normalized Discounted Cumulative Gain)

```
DCG@k  = Σ_{i=0}^{k-1} (2^{rel_i} - 1) / log2(i + 2)
IDCG@k = DCG of the ideal ranking (sorted by relevance descending)
NDCG@k = DCG@k / IDCG@k
```

Measures **ranking quality** with support for graded relevance (not just binary). A chunk can be highly relevant (grade 3), somewhat relevant (grade 1), or irrelevant (grade 0). NDCG rewards putting the most relevant chunks at the top. Range: [0.0, 1.0]. Falls back to binary relevance (relevant=1, not relevant=0) when graded relevance is not provided.

## Answer Metrics

Answer metrics evaluate the quality of the generated response, independent of retrieval.

### Citation Accuracy

```
citation_accuracy = valid_citations / total_citations
```

A citation is **valid** if its `chunk_id` exists in the context that was provided to the generator. Invalid citations indicate the model fabricated a source reference. Returns 1.0 if there are no citations (no citations means no invalid citations).

### Faithfulness

```
faithfulness = supported_claims / total_claims
```

A claim is **supported** if the chunk it references is present in the context. This is a structural check -- the claim-to-chunk mapping must be provided. For semantic faithfulness (does the claim text actually follow from the chunk content?), integrate an LLM-as-judge in the future.

### Unsupported Claim Detection

Given a `GeneratedAnswer` and the set of context chunk IDs, returns the list of claims whose citations reference chunks that were not in context. This surfaces potential hallucinations for human review.

## Seed Dataset

The seed dataset provides five deterministic `EvaluationCase` objects for local testing. Each case includes:

| Field | Description |
|-------|-------------|
| `case_id` | Unique identifier (e.g., `seed-001`) |
| `query` | The evaluation query |
| `expected_answer` | Reference answer text |
| `relevant_chunk_ids` | Ground-truth chunk IDs the retriever should find |
| `metadata` | Topic, difficulty, and any extra fields |

**Seed cases:**

| ID | Query | Relevant Chunks | Topic |
|----|-------|-----------------|-------|
| seed-001 | What is machine learning? | chunk-ml-001, chunk-ml-002 | ml |
| seed-002 | How does backpropagation work? | chunk-bp-001, chunk-bp-002, chunk-bp-003 | ml |
| seed-003 | What is a vector database? | chunk-vdb-001 | databases |
| seed-004 | Explain the transformer architecture | chunk-tf-001, chunk-tf-002 | ml |
| seed-005 | What is retrieval augmented generation? | chunk-rag-001, chunk-rag-002, chunk-rag-003 | rag |

The dataset is extensible. For larger-scale evaluation, load external benchmarks (BEIR, HotpotQA, Natural Questions) using `load_dataset` with the same `EvaluationCase` schema. Save and load datasets as JSON with `save_dataset` / `load_dataset` for reproducible experiments.

## Evaluation Report

The `EvaluationReport` captures everything needed to reproduce and compare experiments.

### JSON format

Machine-readable, suitable for dashboards and automated comparison:

```json
{
  "run_id": "eval-20260309-143000",
  "config": {
    "retrieval_mode": "dense",
    "embedding_model": "all-MiniLM-L6-v2",
    "reranker_id": "",
    "chunking_strategy": "fixed_window",
    "k_values": [5, 10, 20]
  },
  "dataset_size": 5,
  "evaluated_at": "2026-03-09T14:30:00+00:00",
  "stage_metrics": [
    {
      "stage": "retrieval",
      "metric_means": {
        "recall@5": 0.6500,
        "precision@5": 0.3200,
        "mrr": 0.7333,
        "ndcg@5": 0.5800
      },
      "metric_medians": { ... },
      "query_count": 5
    }
  ],
  "query_results": [ ... ]
}
```

### Markdown format

Human-readable, suitable for experiment logs and pull request summaries. Generated by `generate_markdown_summary`. Includes configuration, aggregate metric tables with mean and median, and per-query breakdowns showing retrieved/relevant/hit counts.

## Usage

### Basic evaluation

```python
from libs.evaluation import (
    RetrievalEvaluator,
    EvaluationConfig,
    create_seed_dataset,
    generate_markdown_summary,
    save_json_report,
)
from pathlib import Path

# Load dataset
cases = create_seed_dataset()

# Configure evaluation
config = EvaluationConfig(
    retrieval_mode="dense",
    embedding_model="all-MiniLM-L6-v2",
    chunking_strategy="fixed_window",
    k_values=[5, 10],
)

# Run evaluation with any Retriever implementation
evaluator = RetrievalEvaluator(config=config)
report = evaluator.evaluate(cases, my_retriever, k_values=[5, 10])

# Output
print(generate_markdown_summary(report))
save_json_report(report, Path("eval_results.json"))
```

### Answer quality evaluation

```python
from libs.evaluation import evaluate_citation_accuracy, evaluate_faithfulness

# Citation accuracy: do citations reference chunks that were in context?
accuracy = evaluate_citation_accuracy(generated_answer, context_chunk_ids)
print(f"Citation accuracy: {accuracy.score:.2%}")

# Faithfulness: are claims supported by context?
faithfulness = evaluate_faithfulness(claims, context_chunk_ids, claim_to_chunk)
print(f"Faithfulness: {faithfulness.score:.2%}")
print(f"Unsupported: {faithfulness.unsupported_claims}")
```

### Dataset persistence

```python
from libs.evaluation import save_dataset, load_dataset, create_seed_dataset
from pathlib import Path

cases = create_seed_dataset()
save_dataset(cases, Path("my_dataset.json"))
loaded = load_dataset(Path("my_dataset.json"))
assert len(loaded) == len(cases)
```

## Components

| Component | Location | Responsibility |
|-----------|----------|----------------|
| `RetrievalEvaluator` | `libs/evaluation/evaluator.py` | Runs cases against a retriever, computes metrics, produces reports |
| `Retriever` protocol | `libs/evaluation/evaluator.py` | Interface any retriever must satisfy for evaluation |
| `recall_at_k`, `precision_at_k`, `mrr`, `ndcg_at_k` | `libs/evaluation/retrieval_metrics.py` | Individual metric functions |
| `evaluate_citation_accuracy`, `evaluate_faithfulness`, `detect_unsupported_claims` | `libs/evaluation/answer_metrics.py` | Answer quality metric functions |
| `EvaluationReport`, `QueryResult`, `StageMetrics`, `EvaluationConfig` | `libs/evaluation/models.py` | Typed data models for reports |
| `EvaluationCase` | `libs/contracts/evaluation.py` | Ground-truth query-answer pair |
| `create_seed_dataset`, `save_dataset`, `load_dataset` | `libs/evaluation/dataset.py` | Dataset creation and persistence |
| `report_to_dict`, `save_json_report`, `generate_markdown_summary`, `save_markdown_report` | `libs/evaluation/report.py` | Report serialization to JSON and Markdown |

## Skills Applied

- **evaluation_analysis**: Stage-wise metric computation with per-query and aggregate breakdowns.
- **benchmark_setup**: Seed dataset creation with extensibility to external benchmarks (BEIR, HotpotQA, NQ).
- **experiment_logging**: Full configuration capture in reports for reproducible comparisons.
- **metric_interpretation**: Each metric targets a specific quality dimension (coverage, noise, user experience, ranking quality, faithfulness, citation integrity).
