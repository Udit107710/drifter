# Experiment Runner and Reproducibility

## Overview

The experiments subsystem (`libs/experiments/`) provides structured experiment management on top of the evaluation harness. It supports configuration-driven experiment definitions, an orchestrating runner, baseline comparison, and a persistence layer.

## Key Concepts

### ExperimentConfig

A declarative experiment definition capturing everything needed for reproduction:

- **name**: Human-readable identifier
- **hypothesis**: What you expect to observe (required — forces intentional experimentation)
- **eval_config**: Evaluation parameters (retrieval mode, embedding model, reranker, chunking strategy)
- **dataset_path**: Path to a JSON evaluation dataset
- **artifact_dir**: Output directory for JSON/markdown reports
- **k_values**: Recall/precision cut-offs (default: [5, 10, 20])
- **tags**: For filtering and grouping runs
- **baseline_run_id**: Optional reference for automatic comparison

### ExperimentRun

The result of executing an experiment. Captures full provenance:

- Config snapshot, evaluation report with per-query breakdown
- Git SHA (auto-captured), timestamps, duration
- Status (PENDING, RUNNING, COMPLETED, FAILED) and error message if failed

### ExperimentStore

Protocol for persisting runs. The `InMemoryExperimentStore` is provided for testing. Production implementations can back this with Postgres, file-system JSON, etc.

## Usage

```python
from libs.evaluation.dataset import load_dataset, save_dataset, create_seed_dataset
from libs.experiments import ExperimentRunner, InMemoryExperimentStore, compare_runs
from libs.experiments.sample_configs import chunk_size_experiment

# 1. Define experiment
config = chunk_size_experiment(
    dataset_path="data/eval_dataset.json",
    artifact_dir="results/chunk-256",
    chunk_size=256,
)

# 2. Run
store = InMemoryExperimentStore()
runner = ExperimentRunner(store=store)
run = runner.run(config, my_retriever, run_id="exp-001")

# 3. Compare against baseline
baseline = store.get("exp-baseline")
if baseline:
    comparison = compare_runs(baseline, run)
```

## One-Variable-at-a-Time Design

The `sample_configs` module provides factory functions that vary exactly one parameter:

| Factory | Variable | Fixed |
|---------|----------|-------|
| `chunk_size_experiment()` | Chunk size | Dense retrieval, no reranker |
| `retrieval_mode_experiment()` | Retrieval mode | Fixed chunk size, no reranker |
| `reranker_experiment()` | Reranker ID | Dense retrieval, fixed chunk size |

## Comparison Reports

`compare_runs()` produces an `ExperimentComparison` with:

- **MetricDelta** per metric: absolute change, relative change, improved flag
- **Config diffs**: which parameters differ between baseline and candidate

`generate_comparison_markdown()` renders this as a human-readable table.

## Artifacts

Each run produces in the artifact directory:
- `{run_id}.json` — machine-readable evaluation report
- `{run_id}.md` — human-readable markdown summary
