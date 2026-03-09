# Context Builder Subsystem

Stage 3 of the query plane. Consumes a precision-ranked shortlist of `RankedCandidate` objects from the reranker and produces a `ContextPack` — the final evidence set passed to generation.

This is often where retrieval quality is lost if token budgeting and redundancy handling are poor. The context builder enforces strict token limits, removes duplicate passages, and makes every inclusion/exclusion decision inspectable.

## Boundary

- **Input**: `list[RankedCandidate]` + `query: str` + `token_budget: int`
- **Output**: `BuilderResult` containing a `ContextPack`

The context builder is isolated from both reranking (stage 2) and generation (stage 4). It selects evidence, not prompts.

## Architecture

```
list[RankedCandidate]
    |
    v  dedup.py
deduplicate(by content_hash)
    |
    v  strategy selection
    |
    +-- GreedyContextBuilder     (rank-order packing)
    +-- DiversityAwareBuilder    (MMR-style, planned)
    |
    v  token budgeting (strict enforcement)
    |
    v  build ContextPack
    |
    v
BuilderResult (with outcome, exclusions, timing, debug)
```

## Components

### ContextBuilder Protocol (`protocols.py`)

Runtime-checkable protocol with a single method:

```python
def build(
    candidates: list[RankedCandidate],
    query: str,
    token_budget: int,
) -> BuilderResult
```

Any builder must satisfy this contract. The protocol depends on `TokenCounter` from the chunking subsystem for pluggable token counting.

### GreedyContextBuilder (`greedy_builder.py`)

Packs chunks in rank order until the token budget is exhausted. Higher-ranked candidates are always preferred.

Algorithm:
1. Deduplicate by `content_hash` (if configured)
2. Iterate candidates in rank order
3. Count tokens via pluggable `TokenCounter`
4. If the chunk fits within remaining budget, include it with `SelectionReason.TOP_RANKED`
5. If it exceeds budget, record an `ExclusionRecord` and continue (skip, do not truncate)
6. Stop early if `max_chunks` limit is reached

The greedy builder is deterministic and fast — suitable for baselines and production use when source diversity is not a concern.

### DiversityAwareBuilder (planned)

An MMR-style (Maximal Marginal Relevance) builder that balances relevance and novelty. For each candidate position, it scores:

```
score = (1 - diversity_weight) * relevance + diversity_weight * novelty
```

Where `novelty` measures how different a candidate is from already-selected evidence (e.g., by source ID, content similarity). Candidates selected for diversity reasons receive `SelectionReason.DIVERSITY`.

This builder will share the same deduplication and token budgeting infrastructure as the greedy builder.

### ContextBuilderService (planned)

Orchestrator that wraps any `ContextBuilder` implementation:
1. Validates input (empty check, budget sanity)
2. Calls `builder.build()` (catches exceptions)
3. Enriches the `BuilderResult` with trace metadata
4. Returns the result with timing and error context

Follows the same pattern as `RerankerService` in the reranking subsystem.

### Deduplication (`dedup.py`)

Removes chunks with identical `content_hash`, keeping the first (highest-ranked) occurrence. Excluded duplicates are recorded as `ExclusionRecord` entries with the reason `duplicate:content_hash=<hash>`.

Deduplication runs before packing so that budget is not wasted on redundant content.

## Configuration

`BuilderConfig` controls builder behavior:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `token_budget` | `int` | 3000 | Maximum tokens in the final context pack |
| `diversity_weight` | `float` | 0.3 | Weight for novelty vs relevance (DiversityAwareBuilder) |
| `max_chunks` | `int` | 0 | Maximum chunks to include (0 = no limit) |
| `max_chunks_per_source` | `int` | 2 | Maximum chunks from any single source (0 = no cap) |
| `deduplicate` | `bool` | `True` | Whether to remove duplicate chunks by content_hash |

The `max_chunks_per_source` cap prevents a single document from dominating the context pack. When a chunk exceeds the cap for its source, it is excluded with reason `source_cap:<source_id>`. This works alongside token budgeting to ensure both token-level and source-level diversity.

**Default token budget**: The bootstrap sets `token_budget=5000` by default. This is large enough to include sufficient evidence for multi-hop questions while staying within typical LLM context windows. Override with `--config token_budget=N`.

## Inspectability

Every inclusion and exclusion decision is recorded for debugging and evaluation.

### ExclusionRecord

Each excluded chunk gets a record with:
- `chunk_id` — which chunk was excluded
- `reason` — machine-readable reason string (e.g., `budget_exceeded:needed=45,remaining=12`, `duplicate:content_hash=abc`, `zero_tokens`)
- `token_count` — how many tokens the chunk would have consumed

### Debug Dict

The `BuilderResult.debug` dictionary includes:
- `builder_type` — which strategy was used (`"greedy"`)
- `input_count` — total candidates received
- `token_budget` — the configured budget
- `post_dedup_count` — candidates remaining after deduplication
- `selected_count` — chunks included in the context pack
- `excluded_count` — total exclusions (dedup + budget + zero tokens)
- `tokens_used` — actual token consumption
- `tokens_remaining` — unused budget
- `unique_sources` — number of distinct source IDs in the pack

### Diversity Score

The `ContextPack.diversity_score` field measures source coverage: `unique_sources / selected_count`, clamped to [0, 1]. A score of 1.0 means every chunk comes from a different source.

## Token Budgeting

Token budget enforcement is strict — the builder never exceeds the budget. Key design decisions:

- Tokens are counted per-chunk using a pluggable `TokenCounter` (protocol from `libs.chunking.protocols`)
- The built-in `WhitespaceTokenCounter` splits on whitespace (suitable for testing)
- Production deployments should use a model-specific tokenizer (e.g., tiktoken for GPT models)
- Chunks that exceed remaining budget are skipped entirely, not truncated
- Zero-token chunks are excluded with reason `zero_tokens`
- The `ContextPack` contract enforces `total_tokens <= token_budget` at construction time

## Usage

```python
from libs.chunking.token_counter import WhitespaceTokenCounter
from libs.context_builder.greedy_builder import GreedyContextBuilder
from libs.context_builder.models import BuilderConfig

# Create builder with a token counter
counter = WhitespaceTokenCounter()
config = BuilderConfig(token_budget=2000, deduplicate=True, max_chunks=10)
builder = GreedyContextBuilder(token_counter=counter, config=config)

# Build context from ranked candidates
result = builder.build(
    candidates=ranked_candidates,
    query="machine learning optimization",
    token_budget=2000,
)

# Inspect the result
print(f"Outcome: {result.outcome.value}")
print(f"Tokens used: {result.context_pack.total_tokens}/{result.context_pack.token_budget}")
print(f"Diversity: {result.context_pack.diversity_score:.2f}")

for item in result.context_pack.evidence:
    print(f"  Rank {item.rank}: {item.chunk.chunk_id} ({item.token_count} tokens, {item.selection_reason.value})")

for excl in result.exclusions:
    print(f"  Excluded {excl.chunk_id}: {excl.reason}")
```

## Models

- **`BuilderOutcome`**: `SUCCESS`, `EMPTY_CANDIDATES`, `BUDGET_EXHAUSTED`, `FAILED`
- **`BuilderConfig`**: frozen dataclass with `token_budget`, `diversity_weight`, `max_chunks`, `deduplicate`
- **`ExclusionRecord`**: frozen dataclass with `chunk_id`, `reason`, `token_count`
- **`BuilderResult`**: envelope with `context_pack`, `outcome`, `exclusions`, `input_count`, `dedup_removed`, `total_latency_ms`, `completed_at`, `errors`, `debug`

## Skills Applied

This subsystem was built following these skills:
- `rag/context_packing_analysis` — token-aware packing, inclusion/exclusion tracking
- `rag/context_diversity` — diversity scoring, MMR-style selection (planned)
- `rag/rag_trace_analysis` — debug dict with full trace metadata
- `reliability/failure_mode_analysis` — explicit outcomes, contextual error reporting
- `scalability/token_budget_management` — strict enforcement, pluggable counters
