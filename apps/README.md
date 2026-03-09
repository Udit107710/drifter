# apps/

Application entry points. Each app is a thin presentation layer over the orchestrators.

## Applications

### `cli/` — Command-Line Interface (implemented)

The `rag` CLI provides access to all pipeline stages, evaluation, and experiments.

```bash
uv run rag ask "What is machine learning?"
uv run rag --json retrieve "query" --mode dense --top-k 10
uv run rag debug-query "query"
uv run rag evaluate --dataset data.json
```

Structure:
- `main.py` — argparse setup, global flags (`--json`, `--trace`, `--config`, `-v`), dispatch
- `output.py` — `OutputRenderer` with JSON and human-readable modes
- `errors.py` — Exit codes (0=success, 1=partial, 2=failed, 3=config, 4=input)
- `commands/` — Individual command handlers (one file per command)

Commands:
| Command | Purpose |
|---------|---------|
| `retrieve` | Retrieval only |
| `rerank` | Retrieve + rerank |
| `build-context` | Retrieve + rerank + context pack |
| `generate` | Full pipeline, show generation details |
| `ask` | Full pipeline, display answer + citations |
| `debug-query` | Full pipeline, always JSON with all debug info |
| `ingest` | Run ingestion pipeline |
| `index` | Index previously ingested chunks |
| `evaluate` | Run retrieval evaluation |
| `experiment` | Run/compare experiments |

See [docs/cli_commands.md](../docs/cli_commands.md) for the full reference.

### `api/` — Query-Serving HTTP API (planned)

Online, latency-sensitive, read-heavy. Will expose `QueryOrchestrator` over HTTP. No ingestion or indexing logic.

### `worker/` — Background Ingestion Worker (planned)

Offline, batch-oriented, write-heavy. Will drive `IngestionOrchestrator` for batch document processing.

### `debugger_ui/` — Pipeline Debugger UI (planned)

Browser-based interface for inspecting pipeline internals: chunk lineage, retrieval candidates, reranking scores, trace visualization.

See [docs/debugger_ui_plan.md](../docs/debugger_ui_plan.md) for the design plan.

## Design Principle

Apps are thin. They:
1. Parse user input (CLI args, HTTP requests)
2. Create a `ServiceRegistry` via `create_registry()`
3. Call an orchestrator
4. Render the result

Business logic stays in `libs/`. Pipeline composition stays in `orchestrators/`.
