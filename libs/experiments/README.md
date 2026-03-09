# libs/experiments/

Reproducible experiment framework with configuration-driven definitions, execution, and comparison.

## Boundary

- **Consumes:** ExperimentConfig + Retriever
- **Produces:** ExperimentRun (with evaluation report, provenance, artifacts)
- **Rule:** Experiments must be reproducible. Record dataset, strategy, model, config, and metrics.

## Runner

```python
class ExperimentRunner:
    def __init__(self, store: ExperimentStore | None = None) -> None: ...
    def run(self, config: ExperimentConfig, retriever: Retriever, run_id: str = "") -> ExperimentRun: ...
```

Pipeline:
1. Load dataset from config path
2. Run `RetrievalEvaluator` with the given retriever
3. Save JSON and markdown artifacts
4. Persist run to store (if provided)
5. Return `ExperimentRun` with full provenance

## Key Types

| Type | Purpose |
|------|---------|
| `ExperimentConfig` | name, hypothesis, eval_config, dataset_path, artifact_dir, k_values, tags |
| `ExperimentRun` | run_id + config + report + status + timestamps + git_sha + duration |
| `ExperimentStatus` | COMPLETED, FAILED |
| `MetricDelta` | Metric change between baseline and candidate (absolute + relative) |
| `ExperimentComparison` | Side-by-side comparison of two runs with deltas and config diffs |

## Comparison (`comparison.py`)

```python
def compare_runs(baseline: ExperimentRun, candidate: ExperimentRun) -> ExperimentComparison: ...
def generate_comparison_markdown(comparison: ExperimentComparison) -> str: ...
```

Computes metric deltas and identifies config differences between runs.

## Persistence

| Store | Purpose |
|-------|---------|
| `ExperimentStore` | Protocol: save, get, list_all, list_by_tag, list_by_name |
| `InMemoryExperimentStore` | Dict-backed store for testing |

## Provenance

Every experiment run records:
- **Git SHA** (`git_info.py`) — exact code version
- **Config** — full experiment configuration
- **Timestamps** — start, end, duration
- **Artifacts** — JSON + markdown reports saved to disk

## Sample Configs (`sample_configs.py`)

Pre-built experiment configurations:
- `retrieval_mode_experiment()` — Compare dense vs. lexical vs. hybrid
- `chunk_size_experiment()` — Compare different chunk sizes
- `reranker_experiment()` — Compare reranking strategies

## Testing

Uses `InMemoryExperimentStore` and synthetic datasets. Fully deterministic.
