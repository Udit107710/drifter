# orchestrators/

Thin composition layers that wire library services into coherent pipelines.

## Architecture

Orchestrators sit between the application layer (`apps/`) and the core libraries (`libs/`). They:
- Import only protocols and contracts from `libs/` — never concrete adapters
- Own degraded-mode logic (fallbacks when stages fail)
- Propagate trace IDs through all pipeline stages
- Wrap each stage in `pipeline_span()` for observability

## Modules

### `bootstrap.py` — ServiceRegistry

The composition root. Creates all services from configuration.

```python
@dataclass
class ServiceRegistry:
    tracer: Tracer
    retrieval_broker: RetrievalBroker
    reranker_service: RerankerService
    context_builder_service: ContextBuilderService
    generation_service: GenerationService
    indexing_service: IndexingService | None
    evaluator: RetrievalEvaluator
    experiment_runner: ExperimentRunner
    token_budget: int

def create_registry(overrides: dict[str, Any] | None = None) -> ServiceRegistry: ...
```

Steps:
1. Load configs from `DRIFTER_*` env vars
2. Validate overrides (reject secret fields)
3. Call adapter factories to create concrete implementations
4. Construct library services
5. Return `ServiceRegistry`

When no env vars are set, all services use in-memory/mock implementations.

### `query.py` — QueryOrchestrator

Composes the 4-stage query pipeline:

```
retrieval -> reranking -> context building -> generation
```

```python
class QueryOrchestrator:
    def run(self, query: str, trace_id: str | None = None, top_k: int = 50, token_budget: int | None = None) -> QueryResult: ...
    def run_retrieve_only(self, query: str, ...) -> QueryResult: ...
    def run_through_rerank(self, query: str, ...) -> QueryResult: ...
    def run_through_context(self, query: str, ...) -> QueryResult: ...
```

**Degraded mode:** If the broker returns FAILED, short-circuits immediately. If reranking fails, falls back to retrieval-order candidates (outcome becomes "partial").

**QueryResult** contains all intermediate outputs:
- `broker_result` — Retrieval candidates
- `reranker_result` — Reranked candidates
- `builder_result` — Context pack
- `generation_result` — Generated answer

### `ingestion.py` — IngestionOrchestrator

Composes the ingestion pipeline:

```
ingestion -> parsing -> chunking -> indexing
```

```python
class IngestionOrchestrator:
    def run(self, run_id: str | None = None, trace_id: str | None = None) -> IngestionPipelineResult: ...
```

Handles missing parsers and indexing errors gracefully, collecting errors without aborting.

## Testing

All orchestrators tested with in-memory stores and mock services. No external dependencies.
