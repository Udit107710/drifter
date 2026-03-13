# CLI Command Reference

## Usage

```
drifter <command> [options]
```

## Global Flags

| Flag | Description |
|------|-------------|
| `--json` | JSON output to stdout |
| `--trace <id>` | Use specific trace ID |
| `--config KEY=VALUE` | Override config (repeatable) |
| `-v`, `--verbose` | Verbose output |
| `--config-file CONFIG_FILE` | Override path to config YAML |
| `--env-file ENV_FILE` | Override path to .env file |

Global flags must appear before the subcommand.

## Subsystem Commands

### `drifter retrieve <query>`

Run retrieval only.

| Arg | Default | Description |
|-----|---------|-------------|
| `--top-k` | 50 | Number of candidates |
| `--mode` | hybrid | `dense`, `lexical`, or `hybrid` |

### `drifter rerank <query>`

Retrieve + rerank.

| Arg | Default | Description |
|-----|---------|-------------|
| `--top-k` | 50 | Retrieval candidates |
| `--top-n` | 0 | Reranker top-n cutoff (0 = all) |

### `drifter build-context <query>`

Retrieve + rerank + build context pack.

| Arg | Default | Description |
|-----|---------|-------------|
| `--top-k` | 50 | Retrieval candidates |
| `--token-budget` | 3000 | Token budget for context |

### `drifter generate <query>`

Full pipeline, show generation details.

| Arg | Default | Description |
|-----|---------|-------------|
| `--top-k` | 50 | Retrieval candidates |
| `--token-budget` | 3000 | Token budget |

### `drifter ingest`

Ingest documents from a local file or directory. Runs the full pipeline: read → parse → chunk → embed → index into vector and lexical stores.

| Arg | Default | Description |
|-----|---------|-------------|
| `--path` | required | File or directory to ingest |
| `--run-id` | auto | Explicit run ID for replay detection |

Supported file types: Markdown (`.md`), plain text (`.txt`).

**Examples:**

```bash
# Ingest a directory of markdown files
uv run drifter ingest --path data/novel_chapters/

# Ingest a single file
uv run drifter ingest --path data/notes.md

# With explicit run ID for idempotent replay
uv run drifter ingest --path data/novel_chapters/ --run-id my-run-001
```

**Output:**
```
trace: 28de3d36...
run_id: ingest-20260309-134436
latency: 1100.3ms

Documents ingested: 24
Chunks produced:    96
Chunks indexed:     192
```

### `drifter index`

Index previously ingested chunks.

| Arg | Default | Description |
|-----|---------|-------------|
| `--run-id` | auto | Run ID to index chunks for |

## End-to-End Commands

### `drifter ask <query>`

Full pipeline, display answer with citations.

| Arg | Default | Description |
|-----|---------|-------------|
| `--top-k` | 50 | Retrieval candidates |
| `--token-budget` | 3000 | Token budget |
| `--mode` | hybrid | Retrieval mode |
| `--stream` | off | Enable streaming generation with thinking model output |

### `drifter debug-query <query>`

Full pipeline with all debug information. Always outputs JSON.

| Arg | Default | Description |
|-----|---------|-------------|
| `--top-k` | 50 | Retrieval candidates |
| `--token-budget` | 3000 | Token budget |

## Evaluation Commands

### `drifter evaluate`

Run retrieval evaluation against a dataset.

| Arg | Default | Description |
|-----|---------|-------------|
| `--dataset` | required | Path to evaluation dataset JSON |
| `--k` | 5,10,20 | Comma-separated k values |

### `drifter experiment run`

Run an experiment.

| Arg | Default | Description |
|-----|---------|-------------|
| `--config` | required | Experiment config JSON path |
| `--hypothesis` | "" | Experiment hypothesis |

### `drifter experiment compare`

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
