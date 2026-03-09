# CLI Command Reference

## Usage

```
rag <command> [options]
```

## Global Flags

| Flag | Description |
|------|-------------|
| `--json` | JSON output to stdout |
| `--trace <id>` | Use specific trace ID |
| `--config KEY=VALUE` | Override config (repeatable) |
| `-v`, `--verbose` | Verbose output |

Global flags must appear before the subcommand.

## Subsystem Commands

### `rag retrieve <query>`

Run retrieval only.

| Arg | Default | Description |
|-----|---------|-------------|
| `--top-k` | 50 | Number of candidates |
| `--mode` | hybrid | `dense`, `lexical`, or `hybrid` |

### `rag rerank <query>`

Retrieve + rerank.

| Arg | Default | Description |
|-----|---------|-------------|
| `--top-k` | 50 | Retrieval candidates |
| `--top-n` | 0 | Reranker top-n cutoff (0 = all) |

### `rag build-context <query>`

Retrieve + rerank + build context pack.

| Arg | Default | Description |
|-----|---------|-------------|
| `--top-k` | 50 | Retrieval candidates |
| `--token-budget` | 3000 | Token budget for context |

### `rag generate <query>`

Full pipeline, show generation details.

| Arg | Default | Description |
|-----|---------|-------------|
| `--top-k` | 50 | Retrieval candidates |
| `--token-budget` | 3000 | Token budget |

### `rag ingest`

Run ingestion pipeline.

| Arg | Default | Description |
|-----|---------|-------------|
| `--run-id` | auto | Explicit run ID for replay detection |

### `rag index`

Index previously ingested chunks.

| Arg | Default | Description |
|-----|---------|-------------|
| `--run-id` | auto | Run ID to index chunks for |

## End-to-End Commands

### `rag ask <query>`

Full pipeline, display answer with citations.

| Arg | Default | Description |
|-----|---------|-------------|
| `--top-k` | 50 | Retrieval candidates |
| `--token-budget` | 3000 | Token budget |
| `--mode` | hybrid | Retrieval mode |

### `rag debug-query <query>`

Full pipeline with all debug information. Always outputs JSON.

| Arg | Default | Description |
|-----|---------|-------------|
| `--top-k` | 50 | Retrieval candidates |
| `--token-budget` | 3000 | Token budget |

## Evaluation Commands

### `rag evaluate`

Run retrieval evaluation against a dataset.

| Arg | Default | Description |
|-----|---------|-------------|
| `--dataset` | required | Path to evaluation dataset JSON |
| `--k` | 5,10,20 | Comma-separated k values |

### `rag experiment run`

Run an experiment.

| Arg | Default | Description |
|-----|---------|-------------|
| `--config` | required | Experiment config JSON path |
| `--hypothesis` | "" | Experiment hypothesis |

### `rag experiment compare`

Compare two experiment runs. (Requires persistent store.)

| Arg | Default | Description |
|-----|---------|-------------|
| `--baseline` | required | Baseline run ID |
| `--candidate` | required | Candidate run ID |

## Output Modes

**Human mode** (default): metadata (trace ID, latency, outcome) goes to stderr; primary content goes to stdout. Useful for piping.

**JSON mode** (`--json`): single JSON object to stdout:
```json
{
  "trace_id": "...",
  "outcome": "success",
  "latency_ms": 42.5,
  "data": { ... },
  "errors": []
}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Partial (degraded results) |
| 2 | Failed |
| 3 | Config error |
| 4 | Input error |
