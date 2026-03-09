# 00 — System Design

Reference architecture for the Drifter RAG platform.

---

## System Goals

1. **Deep understanding** — every pipeline stage is explicit, inspectable, and independently testable
2. **Modularity** — subsystems communicate through typed contracts; swapping an implementation never requires touching another subsystem
3. **Observability** — every stage emits structured traces (OpenTelemetry); latency, throughput, and error rates are measurable per-stage
4. **Evaluation-first** — retrieval quality and answer quality are measured continuously, not bolted on later
5. **Scalability path** — the architecture supports single-process local development and multi-service deployment without structural changes
6. **Framework independence** — external tools (Qdrant, OpenSearch, vLLM, etc.) live behind adapter interfaces; core logic has zero framework imports

---

## Architectural Overview

RAG is a **multi-stage information retrieval system**. The four stages must remain independent modules:

```
candidate generation → precision ranking → context optimization → LLM reasoning
      (retrieval)         (reranking)        (context builder)      (generation)
```

The system is organized into three planes that separate concerns:

```
┌─────────────────────────────────────────────────────────┐
│                     CONTROL PLANE                       │
│  source registry · experiment config · index lifecycle  │
│  evaluation datasets · provider config · crawl state    │
├─────────────────────────────────────────────────────────┤
│                      DATA PLANE                         │
│  raw docs · canonical docs · chunks · embeddings        │
│  vector index · lexical index · metadata/lineage store  │
│  evaluation reports                                     │
├─────────────────────────────────────────────────────────┤
│                     QUERY PLANE                         │
│  query normalization → retrieval broker → reranking     │
│  → context builder → generation → citation/verification │
│  tracing / diagnostics                                  │
└─────────────────────────────────────────────────────────┘
```

---

## Planes and Responsibilities

### Control Plane

Owns **configuration, orchestration, and lifecycle** — never touches document content directly.

| Component | Responsibility |
|---|---|
| Source registry | Declares data sources (URI, type, auth, schedule) |
| Crawl / index coordinator | Tracks crawl state, triggers ingestion runs, manages tombstones |
| Experiment registry | Defines experiment parameters (chunking strategy, embedding model, retrieval mode, reranker) |
| Evaluation dataset registry | Manages gold-standard query–answer pairs and relevance judgments |
| Provider configuration | Stores connection details and model identifiers for external services |
| Index lifecycle manager | Handles index creation, versioning, alias swaps, and garbage collection |

### Data Plane

Owns **persisted knowledge artifacts and indexes** — written by ingestion, read by query plane.

| Store | Contents | Production backing | Local-mode backing |
|---|---|---|---|
| Raw document store | Original bytes as fetched from sources | MinIO (S3-compatible) | In-memory / filesystem |
| Canonical document store | Parsed, structured representations (`CanonicalDocument` with `Block` list) | Postgres | In-memory |
| Chunk store | `Chunk` records with full lineage back to source | Postgres | In-memory |
| Embedding records | `ChunkEmbedding` with model version and vector | Postgres | In-memory |
| Crawl state store | Crawl progress, work items, tombstone tracking | Postgres | In-memory |
| Vector index | Dense retrieval index (adapter: Qdrant) | Qdrant | In-memory |
| Lexical index | Full-text search index (adapter: OpenSearch) | OpenSearch | In-memory |
| Metadata / lineage store | Cross-references between sources, documents, chunks, and embeddings | Postgres | In-memory |
| Experiment config + evaluation results | Experiment parameters and stored metric snapshots | Postgres | In-memory |

### Query Plane

Owns **online request handling** — reads from data plane, never writes to it.

| Stage | Input → Output |
|---|---|
| Query normalization | Raw query → normalized query (cleaned, expanded) |
| Retrieval broker | Normalized query → `list[RetrievalCandidate]` (orchestrates dense + lexical + hybrid) |
| Reranking | `list[RetrievalCandidate]` → `list[RankedCandidate]` (cross-encoder or feature-based scoring) |
| Context builder | `list[RankedCandidate]` → `ContextPack` (token budgeting, dedup, diversity) |
| Generation | `ContextPack` + query → `GeneratedAnswer` with `list[Citation]` |
| Verification | `GeneratedAnswer` → verified answer (citation check, unsupported claim detection) |

---

## Subsystems

14 subsystems, each a library under `libs/`:

| # | Subsystem | Library path | Plane |
|---|---|---|---|
| 1 | Ingestion | `libs/ingestion` | Control + Data |
| 2 | Parsing | `libs/parsing` | Data |
| 3 | Chunking | `libs/chunking` | Data |
| 4 | Embeddings | `libs/embeddings` | Data |
| 5 | Indexing | `libs/indexing` | Data |
| 6 | Retrieval stores | `libs/retrieval/stores` | Data |
| 7 | Retrieval broker | `libs/retrieval/broker` | Query |
| 8 | Reranking | `libs/reranking` | Query |
| 9 | Context builder | `libs/context_builder` | Query |
| 10 | Generation | `libs/generation` | Query |
| 11 | Observability | `libs/observability` | Cross-cutting |
| 12 | Evaluation | `libs/evaluation` | Control |
| 13 | Experiments | `libs/experiments` | Control |
| 14 | Integration adapters | `libs/adapters` | Cross-cutting |

### Dependency rules

```
ingestion → parsing → chunking → embeddings → indexing
                                                  ↓
query normalization → retrieval broker → reranking → context builder → generation
```

- Arrows indicate data flow, not import dependencies.
- Each subsystem depends on **shared contracts** (the typed domain models), not on other subsystem internals.
- Adapters are injected; subsystems depend on protocols/interfaces, not concrete implementations.

---

## Request Lifecycle (Query Path)

```
1. Client sends query
2. Query normalization
   - whitespace/encoding cleanup
   - optional query expansion or rewriting
   - attach trace_id (OpenTelemetry span starts here)
3. Retrieval broker
   - fan out to configured retrieval strategies:
     a. dense: embed query → vector search (Qdrant adapter)
     b. lexical: tokenize query → full-text search (OpenSearch adapter)
     c. hybrid: merge and deduplicate results from (a) and (b)
   - return list[RetrievalCandidate] with scores and metadata
4. Reranking
   - score candidates with cross-encoder or feature-based model
   - return list[RankedCandidate] ordered by relevance
   - truncate to top-k
5. Context builder
   - allocate token budget (model context window minus system prompt minus generation reserve)
   - select evidence from ranked candidates
   - remove redundant passages
   - optimize for source diversity
   - output ContextPack with selected chunks, token counts, and lineage
6. Generation
   - construct prompt: system instructions + ContextPack + user query
   - call LLM (OpenRouter, OpenAI, Gemini, or vLLM adapter)
   - parse response into GeneratedAnswer with Citations
   - each Citation references a specific chunk in the ContextPack
7. Verification (optional)
   - check each claim against cited evidence
   - flag unsupported claims
8. Return response with answer, citations, and trace_id
```

**Latency budget (p95 target: ~2s total for query path):**

| Stage | Budget |
|---|---|
| Query normalization | 10ms |
| Dense retrieval (vector search) | 50ms |
| Lexical retrieval | 50ms |
| Retrieval merge/dedup | 10ms |
| Reranking (cross-encoder, top-50→top-10) | 200ms |
| Context building | 20ms |
| LLM generation | ~1500ms |
| Verification | 50ms |
| Overhead (serialization, network) | 110ms |

The hot path is: **retrieval broker → reranking → context builder → generation**. These stages must:
- Be non-blocking / async where possible
- Support timeout and circuit-breaker per external call
- Emit per-stage latency spans
- Degrade gracefully (e.g., skip reranking if cross-encoder is unavailable; skip lexical if OpenSearch is down)

---

## Indexing Lifecycle (Ingestion Path)

```
1. Source registry provides source definition (URI, type, schedule, auth)
2. Crawl coordinator checks crawl state
   - determines which documents are new, modified, or deleted
   - produces work items: INGEST, UPDATE, or TOMBSTONE
3. Ingestion
   - fetches raw bytes → raw document store
   - records SourceDocumentRef with content hash, fetch timestamp, version
4. Parsing
   - raw bytes → CanonicalDocument (structured blocks: headings, paragraphs, tables, code, lists)
   - preserves document structure — never flatten to plain text
   - records parser version in metadata
5. Chunking
   - CanonicalDocument → list[Chunk]
   - respects token boundaries (no mid-word/mid-sentence splits)
   - supports strategies: fixed-size, semantic, parent-child
   - each Chunk carries lineage: source_id, document_id, block_ids, byte offsets
   - propagates metadata (title, section headings, source URL)
6. Embedding
   - list[Chunk] → list[ChunkEmbedding]
   - batched for throughput (TEI adapter)
   - each embedding records: model_id, model_version, dimensions
7. Indexing
   - writes ChunkEmbeddings to vector index
   - writes chunk text + metadata to lexical index
   - writes lineage records to metadata store
   - all writes for a single source document are atomic (succeed or fail together)
8. Tombstone handling
   - deleted source documents → mark chunks as deleted in all indexes
   - garbage collection removes tombstoned records after retention period
```

**Idempotency:** Re-ingesting the same document version (same content hash) is a no-op. The crawl coordinator skips unchanged documents.

**Replay safety:** Every ingestion run is identified by a run_id. Restarting a failed run replays only incomplete work items.

**Versioning:** When a document is updated, new chunks are indexed alongside old ones. The old version is tombstoned only after the new version is fully indexed (blue-green pattern for index consistency).

**Consistency:** The metadata/lineage store is the source of truth. If a crash occurs between writing to the vector index and lexical index, a consistency reconciler detects and repairs the mismatch.

---

## Key Internal Data Contracts

```
SourceDocumentRef
  source_id: str
  uri: str
  content_hash: str
  fetched_at: datetime
  version: int
  metadata: dict[str, Any]

CanonicalDocument
  document_id: str
  source_ref: SourceDocumentRef
  blocks: list[Block]
  parser_version: str
  parsed_at: datetime

Block
  block_id: str
  block_type: enum (heading, paragraph, table, code, list, image_caption)
  content: str
  level: int | None          # heading level
  position: int              # order within document
  metadata: dict[str, Any]

Chunk
  chunk_id: str
  document_id: str
  source_id: str
  block_ids: list[str]       # which blocks this chunk spans
  content: str
  token_count: int
  strategy: str              # chunking strategy used
  byte_offset_start: int
  byte_offset_end: int
  metadata: dict[str, Any]   # propagated: title, section, source_url, etc.
  lineage: ChunkLineage

ChunkLineage
  source_id: str
  document_id: str
  block_ids: list[str]
  chunk_strategy: str
  parser_version: str
  created_at: datetime

ChunkEmbedding
  chunk_id: str
  vector: list[float]
  model_id: str
  model_version: str
  dimensions: int
  created_at: datetime

RetrievalCandidate
  chunk: Chunk
  score: float
  retrieval_method: str      # "dense", "lexical", "hybrid"
  store_id: str

RankedCandidate
  candidate: RetrievalCandidate
  rank: int
  rerank_score: float
  reranker_id: str

ContextPack
  query: str
  evidence: list[ContextItem]
  total_tokens: int
  token_budget: int
  diversity_score: float

ContextItem
  chunk: Chunk
  rank: int
  token_count: int
  selected_reason: str       # "top_ranked", "diversity", etc.

GeneratedAnswer
  answer: str
  citations: list[Citation]
  model_id: str
  token_usage: TokenUsage
  trace_id: str

Citation
  claim: str
  chunk_id: str
  chunk_content: str         # exact text cited
  source_id: str
  confidence: float

TokenUsage
  prompt_tokens: int
  completion_tokens: int
  total_tokens: int
```

---

## Failure Modes

### Ingestion path

| Failure | Impact | Mitigation |
|---|---|---|
| Source unreachable | No new documents | Retry with backoff; alert after N failures; stale-data metric |
| Parser crash on malformed document | Single document skipped | Isolate per-document; dead-letter queue; log structured error |
| Embedding service down | Chunks created but not embedded | Retry queue; ingestion run marked incomplete; reconciler picks up |
| Vector index write failure | Embeddings exist but not searchable | Atomic write with rollback; consistency reconciler |
| Lexical index write failure | Same — partial indexing | Same as above |
| Tombstone not propagated | Deleted docs still returned in search | Consistency reconciler; audit log |

### Query path

| Failure | Impact | Mitigation |
|---|---|---|
| Vector search timeout | No dense candidates | Circuit breaker; fall back to lexical-only |
| Lexical search timeout | No lexical candidates | Fall back to dense-only |
| Both retrieval stores down | No candidates | Return error with trace_id; do not hallucinate |
| Reranker timeout | Candidates unranked | Skip reranking; use retrieval scores as ranking |
| LLM timeout | No answer | Return partial response or error; do not retry synchronously |
| LLM hallucination | Unfaithful answer | Verification stage flags unsupported claims |
| Prompt injection via document | System prompt override | Treat all retrieved content as untrusted; delimiter-based prompt structure |

### System-level

| Failure | Impact | Mitigation |
|---|---|---|
| Index version mismatch | Stale or mixed results | Blue-green indexing; alias swap only after full reindex |
| Metadata store inconsistency | Lineage broken | Reconciler job; schema-level foreign key constraints |
| Resource exhaustion (embedding batches) | Backpressure on ingestion | Rate limiting; bounded work queues |

---

## Orchestration — How Libraries Compose

Libraries never import or call each other. They are wired together by **orchestrators** — thin composition layers that:

1. Instantiate libraries with their adapter dependencies (constructor injection)
2. Call one library, take its output, pass it as input to the next
3. Handle cross-cutting concerns (tracing spans, error propagation, degraded-mode fallbacks)

### The two orchestrators

**IngestionOrchestrator** — drives the ingestion pipeline:

```python
class IngestionOrchestrator:
    def __init__(self, source_registry, parser, chunker, embedder, index_writer, tracer):
        # each argument is an already-configured library instance
        ...

    def ingest(self, source_id: str) -> IngestionResult:
        source_ref = self.source_registry.fetch(source_id)          # → SourceDocumentRef
        canonical  = self.parser.parse(source_ref)                  # → CanonicalDocument
        chunks     = self.chunker.chunk(canonical)                  # → list[Chunk]
        embeddings = self.embedder.embed(chunks)                    # → list[ChunkEmbedding]
        self.index_writer.write(chunks, embeddings)                 # → indexes updated
        return IngestionResult(...)
```

**QueryOrchestrator** — drives the query pipeline:

```python
class QueryOrchestrator:
    def __init__(self, normalizer, retrieval_broker, reranker, context_builder, generator, tracer):
        ...

    def answer(self, raw_query: str) -> GeneratedAnswer:
        query      = self.normalizer.normalize(raw_query)           # → normalized query
        candidates = self.retrieval_broker.retrieve(query)          # → list[RetrievalCandidate]
        ranked     = self.reranker.rerank(query, candidates)        # → list[RankedCandidate]
        context    = self.context_builder.build(query, ranked)      # → ContextPack
        answer     = self.generator.generate(query, context)        # → GeneratedAnswer
        return answer
```

### Key design points

**Libraries are unaware of each other.** The parser doesn't know chunking exists. The retrieval broker doesn't know generation exists. They only know about `libs/contracts` — the shared domain types.

**Orchestrators own the pipeline shape.** If you want to skip reranking, the orchestrator decides that — not the reranker. If you want to add a verification step after generation, you add it in the orchestrator.

**Orchestrators are where degraded-mode logic lives.** If the reranker times out, the orchestrator falls back to using retrieval scores. If lexical search is down, the orchestrator runs dense-only. Individual libraries don't handle these cross-library failure modes.

**Dependency injection happens at construction time.** A factory or configuration module creates the concrete adapters (Qdrant, OpenSearch, or in-memory mocks) and passes them to the libraries, then passes the libraries to the orchestrator. This is the single place where the entire system is wired:

```python
# orchestrators/bootstrap.py — the composition root (only file that knows concrete implementations)
from libs.adapters.factory import create_vector_store, create_generator, create_reranker
from libs.adapters.env import load_qdrant_config, load_openrouter_config

def create_registry(overrides=None) -> ServiceRegistry:
    qdrant_cfg = load_qdrant_config()
    vector_store = create_vector_store(qdrant_cfg)   # MemoryVectorStore if unset
    generator = create_generator(load_openrouter_config())  # MockGenerator if unset
    # ... constructs all services, returns ServiceRegistry
```

Factory functions in `libs/adapters/factory.py` create adapter instances with in-memory/mock fallbacks when configs are `None`. All factories return properly typed Protocol instances (`VectorStore`, `EmbeddingProvider`, `QueryEmbedder`, `Reranker`, `Generator`, `SpanCollector`), enabling static type checking at call sites.

Adapter lifecycle is managed through two runtime-checkable protocols in `libs/adapters/protocols.py`: `Connectable` (for adapters that need `connect()`) and `HealthCheckable` (for adapters that support `health_check()`). The bootstrap uses `isinstance` checks against these protocols instead of `hasattr` duck-typing.

Resilience is centralized in `libs/resilience.py`, which provides `is_transient_error()` for error classification and `resilient_call()` / `async_resilient_call()` for retry with exponential backoff and jitter. Both the retrieval broker and indexing service accept an optional `RetryConfig`.

The bootstrap loads configs from `DRIFTER_*` env vars and applies optional `--config` CLI overrides (rejecting secret fields).

**Tests compose the same way** — with no env vars set, all services use in-memory/mock implementations and the orchestrator works identically without any external services.

### Where orchestrators live

```
drifter/
├── libs/          # pure libraries — no cross-library imports
├── orchestrators/
│   ├── __init__.py
│   ├── bootstrap.py     # ServiceRegistry + create_registry() (composition root)
│   ├── ingestion.py     # IngestionOrchestrator
│   ├── query.py         # QueryOrchestrator (sync)
│   └── async_query.py   # AsyncQueryOrchestrator (async retrieval, sync stages 2-4)
├── apps/
│   └── cli/           # thin presentation layer (argparse, output rendering)
```

The `orchestrators/bootstrap.py` module is the only place that knows about concrete adapter classes. `orchestrators/` only knows about library protocols and contracts. `libs/` knows about nothing except `libs/contracts`.

### Resilience

`libs/resilience.py` provides shared retry infrastructure:

- `is_transient_error(exc)` classifies network/timeout errors as retryable
- `RetryConfig` controls max retries, base delay, max delay, and jitter factor
- `resilient_call(fn, config)` and `async_resilient_call(fn, config)` wrap calls with exponential backoff
- Both `RetrievalBroker` and `IndexingService` accept optional `RetryConfig` for store/embedding calls

### Dependency direction

```
apps/cli/            (presentation — argparse, output)
      ↓
orchestrators/bootstrap.py  (knows everything — wiring only)
      ↓
orchestrators/       (knows library protocols + contracts)
      ↓
libs/*               (knows only libs/contracts)
      ↓
libs/contracts       (knows nothing — pure domain types)
```

No upward imports. No circular dependencies. Each layer depends only on the layer below it.

---

## Service vs. Library Breakdown

The system is designed as **libraries first, services later**. During development, everything runs in a single process. The architecture supports extracting services along natural boundaries when scaling requires it.

### Libraries (always)

All 14 subsystems under `libs/` are pure libraries:
- No HTTP servers, no main loops
- Accept configuration and adapter instances via constructor injection
- Communicate through typed contracts (the domain models above)
- Testable with in-memory adapters

### Services (deployment boundary, later)

When deployed, the system splits along two natural service boundaries:

```
┌──────────────────────┐     ┌──────────────────────┐
│   Ingestion Service  │     │    Query Service      │
│                      │     │                       │
│  - crawl coordinator │     │  - query normalization│
│  - ingestion         │     │  - retrieval broker   │
│  - parsing           │     │  - reranking          │
│  - chunking          │     │  - context builder    │
│  - embedding         │     │  - generation         │
│  - indexing          │     │  - verification       │
│                      │     │                       │
│  writes → Data Plane │     │  reads ← Data Plane   │
└──────────────────────┘     └──────────────────────┘
          ↑                            ↑
          └────── Control Plane ───────┘
```

- **Ingestion service**: offline, batch-oriented, write-heavy. Tolerates higher latency. Scales by parallelizing source ingestion.
- **Query service**: online, latency-sensitive, read-heavy. Scales by adding replicas behind a load balancer.
- **Control plane**: configuration API. Low traffic. Could be a third service or embedded in either.

This split is not required for development — it's a deployment concern. The library architecture makes the split mechanical when needed.

---

## Repository Structure

```
drifter/
├── app/
│   ├── factory.py           # wires adapters → libraries → orchestrators
│   └── config.py            # loads configuration
├── orchestrators/
│   ├── __init__.py
│   ├── ingestion.py         # IngestionOrchestrator
│   └── query.py             # QueryOrchestrator
├── libs/
│   ├── contracts/           # shared domain models (all types above)
│   │   ├── __init__.py
│   │   ├── documents.py     # SourceDocumentRef, CanonicalDocument, Block
│   │   ├── chunks.py        # Chunk, ChunkLineage
│   │   ├── embeddings.py    # ChunkEmbedding
│   │   ├── retrieval.py     # RetrievalCandidate, RankedCandidate
│   │   ├── context.py       # ContextPack, ContextItem
│   │   ├── generation.py    # GeneratedAnswer, Citation, TokenUsage
│   │   └── common.py        # shared enums, base classes
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── source_registry.py
│   │   ├── crawler.py
│   │   └── protocols.py     # SourceFetcher protocol
│   ├── parsing/
│   │   ├── __init__.py
│   │   ├── parser.py        # Parser protocol + registry
│   │   └── block_extractor.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── chunker.py       # Chunker protocol
│   │   ├── fixed_size.py
│   │   ├── semantic.py
│   │   └── parent_child.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── embedder.py      # Embedder protocol
│   │   └── batcher.py
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── index_writer.py  # IndexWriter protocol
│   │   └── consistency.py   # reconciler
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── stores/
│   │   │   ├── __init__.py
│   │   │   ├── vector_store.py    # VectorStore protocol
│   │   │   └── lexical_store.py   # LexicalStore protocol
│   │   └── broker/
│   │       ├── __init__.py
│   │       └── retrieval_broker.py
│   ├── reranking/
│   │   ├── __init__.py
│   │   └── reranker.py      # Reranker protocol
│   ├── context_builder/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   ├── token_budget.py
│   │   └── diversity.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── generator.py     # Generator protocol
│   │   ├── prompt_builder.py
│   │   └── citation_extractor.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py       # Recall@k, MRR, NDCG, faithfulness
│   │   ├── datasets.py      # dataset loading
│   │   └── reporter.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── experiment.py    # experiment definition and runner
│   │   └── registry.py
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── tracing.py       # OpenTelemetry span helpers
│   │   └── metrics.py       # counters, histograms
│   └── adapters/
│       ├── __init__.py
│       ├── qdrant.py         # VectorStore implementation
│       ├── opensearch.py     # LexicalStore implementation
│       ├── tei.py            # Embedder implementation
│       ├── vllm.py           # Generator implementation
│       ├── unstructured.py   # Parser implementation
│       ├── tika.py           # Parser implementation
│       └── memory/           # in-memory implementations for testing
│           ├── __init__.py
│           ├── vector_store.py
│           ├── lexical_store.py
│           ├── embedder.py
│           └── generator.py
├── tests/
│   ├── fixtures/             # fixture documents, gold sets
│   ├── unit/                 # per-subsystem unit tests
│   └── integration/          # cross-subsystem tests (still local, using memory adapters)
├── docs/
│   ├── ARCHITECTURE_GUIDE.md
│   ├── IMPLEMENTATION_PRINCIPLES.md
│   ├── EVALUATION_STRATEGY.md
│   ├── CLAUDE_WORKFLOW.md
│   └── 00_SYSTEM_DESIGN.md   # this document
├── prompts/                   # numbered implementation prompts
├── AGENTS.md
├── MASTER_PROMPT.md
└── CLAUDE.md
```

---

## Implementation Order

Each step must include contracts, implementation, tests, and documentation before moving to the next.

| Phase | Subsystem | Rationale |
|---|---|---|
| 1 | `libs/contracts` | Foundation — every other subsystem depends on these types |
| 2 | `libs/observability` | Cross-cutting — needed by all subsystems from the start |
| 3 | `libs/ingestion` + `libs/parsing` | First data flow: source → canonical document |
| 4 | `libs/chunking` | Transforms documents into retrieval-ready units |
| 5 | `libs/embeddings` + `libs/indexing` | Completes the ingestion pipeline: chunks → searchable indexes |
| 6 | `libs/retrieval` (stores + broker) | First query-path component: search indexes → candidates |
| 7 | `libs/reranking` | Precision ranking of candidates |
| 8 | `libs/context_builder` | Evidence selection under token budget |
| 9 | `libs/generation` | LLM reasoning over packed context |
| 10 | `libs/evaluation` + `libs/experiments` | Measure quality; enable experimentation |
| 11 | `libs/adapters` (real implementations) | Connect to Qdrant, OpenSearch, TEI, vLLM |

Memory adapters (`libs/adapters/memory/`) are built alongside each subsystem (phases 3–9) to enable deterministic testing.
