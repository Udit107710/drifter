# Reranking Subsystem

Stage 2 of the query plane. Converts a high-recall candidate pool from the retrieval broker into a precision-oriented shortlist of `RankedCandidate` objects.

## Boundary

- **Input**: `list[RetrievalCandidate]` + `RetrievalQuery`
- **Output**: `list[RankedCandidate]` wrapped in `RerankerResult`

The reranking subsystem is isolated from both retrieval (stage 1) and generation (stage 4). It orders candidates, not prompts.

## Architecture

```
FusedCandidate (broker output)
    │
    ▼ converters.py
RetrievalCandidate
    │
    ▼ RerankerService.run()
    │
    ├── Reranker.rerank()  ← protocol dispatch
    │   ├── PassthroughReranker      (score-order baseline)
    │   ├── FeatureBasedReranker     (weighted multi-signal)
    │   ├── TeiCrossEncoderReranker  (TEI cross-encoder)
    │   ├── HuggingFaceReranker      (HF Inference API)
    │   └── CrossEncoderReranker     (legacy stub)
    │
    ▼ truncate to top_n
RerankerResult (with timing, outcome, debug)
```

## Components

### Reranker Protocol (`protocols.py`)

Runtime-checkable protocol with two requirements:
- `reranker_id: str` property
- `rerank(candidates, query) -> list[RankedCandidate]`

### PassthroughReranker (`mock_reranker.py`)

Sorts by descending retrieval score, assigns sequential ranks 1..N. Deterministic — useful for testing and baselines. ID: `passthrough-v1`.

### FeatureBasedReranker (`feature_reranker.py`)

Computes a weighted sum of six signals, each normalized to [0, 1]:

| Feature | Source | Default Weight |
|---------|--------|---------------|
| retrieval_score | min-max normalized across batch | 1.0 |
| lexical_overlap | fraction of query terms in chunk content | 0.3 |
| source_authority | `chunk.metadata["authority"]`, clamped | 0.2 |
| freshness | `exp(-age_days / 365)` from `chunk.metadata["updated_at"]` | 0.1 |
| title_match | 1.0 if query is substring of `chunk.metadata["title"]` | 0.5 |
| source_type | `chunk.metadata["source_type_score"]`, clamped | 0.1 |
| source_reference | 1.0 if query mentions "chapter N" / "section N" and chunk matches | 2.0 |

The `source_reference` feature detects chapter/section/part/appendix references in the query (e.g. "What happens in chapter 20?") and boosts chunks whose `source_id` or content matches that reference. This is the highest-weighted feature — when a user explicitly references a source, that signal should dominate.

Tie-breaking: by original retrieval score descending. ID: `feature-based-v1`.

### TeiCrossEncoderReranker (`libs/adapters/tei/cross_encoder.py`)

Cross-encoder reranker backed by a TEI server. Sends `(query, document)` pairs to `POST /rerank` and uses the returned relevance scores to rank candidates. ID: `tei-cross-encoder:{model_name}`.

Requires a separate TEI instance running a cross-encoder model (e.g. `BAAI/bge-reranker-base`). Configure via `DRIFTER_TEI_RERANKER_URL`. When not configured, the bootstrap falls back to `FeatureBasedReranker`.

### HuggingFaceReranker (`libs/adapters/huggingface/reranker.py`)

Cross-encoder reranker backed by the HuggingFace Inference API. Uses `huggingface_hub.InferenceClient.text_classification()` to score `(query, document)` pairs. ID: `huggingface:{model_name}`.

Configure via `DRIFTER_HF_TOKEN` and `DRIFTER_HF_RERANKER_MODEL`. The bootstrap prefers TEI when available, then falls back to HuggingFace if configured. Both are checked via the `HealthCheckable` protocol (`isinstance(reranker, HealthCheckable)`) before being selected — the bootstrap never uses `hasattr` for lifecycle dispatch.

The `create_reranker()` factory returns a typed `Reranker` protocol instance, enabling static type checking at all call sites.

### CrossEncoderReranker (`cross_encoder_stub.py`)

Legacy stub that raises `NotImplementedError`. Retained for backwards compatibility. Use `TeiCrossEncoderReranker` or `HuggingFaceReranker` for real cross-encoder reranking.

### RerankerService (`service.py`)

Orchestrator that wraps any `Reranker` implementation:
1. Empty check → `NO_CANDIDATES`
2. Calls `reranker.rerank()` (catches exceptions → `FAILED`)
3. Truncates to `top_n` if configured (0 = no truncation)
4. Builds `RerankerResult` with timing, outcome, and debug info

### Converters (`converters.py`)

Bridges the retrieval broker's `FusedCandidate` to `RetrievalCandidate`:
- `fused_score` → `score`
- First `contributing_stores` entry → `store_id`

## Usage

```python
from libs.reranking import (
    FeatureBasedReranker,
    FeatureWeights,
    RerankerService,
    fused_list_to_retrieval_candidates,
)

# Convert broker output
candidates = fused_list_to_retrieval_candidates(broker_result.candidates)

# Create service with feature-based reranker
service = RerankerService(
    reranker=FeatureBasedReranker(weights=FeatureWeights(source_authority=2.0)),
    top_n=10,
)

# Run reranking
result = service.run(candidates, query)
# result.ranked_candidates is the precision-oriented shortlist
```

## Diagnostics and Inspectability

The service and rerankers are designed for observability per the `reranker_design`, `rag_trace_analysis`, and `failure_mode_analysis` skills:

**Debug payload** (`RerankerResult.debug`) includes:
- `trace_id` — correlates with upstream query trace
- `reranker_id`, `top_n`, `input_count`, `output_count`
- `pre_rerank_score_min`, `pre_rerank_score_max` — enables pre-vs-post comparison
- `truncated_count` — how many candidates were dropped by `top_n`

**Error context** follows `failure_mode_analysis` — errors include reranker ID, trace ID, and candidate count, not just the exception message.

**Score breakdown** — `FeatureBasedReranker.score_breakdown()` returns per-feature raw and weighted scores for a single candidate, enabling debugging of why a candidate ranked where it did.

## Latency Considerations

Reranking is typically the most expensive retrieval-path stage. The `RerankerResult.total_latency_ms` field tracks wall-clock time. When evaluating candidate count vs latency tradeoffs:

- PassthroughReranker: O(n log n) sort, negligible latency
- FeatureBasedReranker: O(n * q) where q = query term count, still fast
- TeiCrossEncoderReranker: O(n) HTTP call to TEI server — use `top_n` to cap input size

Use `top_n` on `RerankerService` to control the tradeoff between precision and latency.

## Models

- **`RerankerOutcome`**: `SUCCESS`, `NO_CANDIDATES`, `FAILED`
- **`FeatureWeights`**: frozen dataclass with per-feature weight floats
- **`RerankerResult`**: envelope with query, ranked candidates, outcome, timing, errors, debug

## Skills Applied

This subsystem was built following these skills:
- `rag/reranker_design` — protocol-based adapter, latency tracking, mock for testing
- `backend/service_boundary_design` — isolated from retrieval and generation
- `reliability/failure_mode_analysis` — contextual errors, explicit error propagation
- `rag/rag_trace_analysis` — trace_id in debug, score breakdown inspectability
- `scalability/latency_budgeting` — latency tracking, truncation controls
