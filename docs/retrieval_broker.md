# Retrieval Broker

## Architecture Role

The retrieval broker sits in the **query plane**, between retrieval stores and the reranker:

```
query → normalize → fanout (dense + lexical) → RRF fusion → source caps → top_k → BrokerResult
```

It orchestrates dense and lexical retrieval backends, fuses their outputs using Reciprocal Rank Fusion (RRF), applies source-diversity caps, and returns a unified `BrokerResult` with full provenance.

## Pipeline Flow

```
                         ┌──────────────────┐
                         │  RetrievalQuery   │
                         └────────┬─────────┘
                                  │
                         ┌────────▼─────────┐
                         │ QueryNormalizer   │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
           ┌────────▼─────────┐       ┌────────▼─────────┐
           │   QueryEmbedder  │       │                   │
           │  embed_query()   │       │                   │
           └────────┬─────────┘       │                   │
                    │                 │                   │
           ┌────────▼─────────┐  ┌───▼───────────────┐   │
           │   VectorStore    │  │   LexicalStore     │   │
           │   .search()      │  │   .search()        │   │
           └────────┬─────────┘  └───┬───────────────┘   │
                    │                │                     │
                    └────────┬───────┘                     │
                             │                             │
                    ┌────────▼─────────┐                   │
                    │   RRF Fusion     │                   │
                    └────────┬─────────┘                   │
                             │                             │
                    ┌────────▼─────────┐                   │
                    │   Source Caps     │                   │
                    └────────┬─────────┘                   │
                             │                             │
                    ┌────────▼─────────┐                   │
                    │   top_k truncate │                   │
                    └────────┬─────────┘                   │
                             │
                    ┌────────▼─────────┐
                    │   BrokerResult   │
                    └──────────────────┘
```

## Reciprocal Rank Fusion (RRF)

RRF merges multiple ranked lists into a single fused ranking. For each candidate appearing in any list:

```
score(d) = Σ weight_i / (k + rank_i)
```

- **k**: Smoothing constant (default 60). Higher values flatten rank differences, giving less advantage to top-ranked items. Lower values amplify rank differences.
- **weight_i**: Per-store weight multiplier (default 1.0 for both dense and lexical).
- **rank_i**: 1-based rank of the candidate in list *i*.

Candidates appearing in multiple lists accumulate contributions from each, naturally boosting items found by both dense and lexical retrieval.

## Query Normalization

The `QueryNormalizer` protocol preprocesses query text before retrieval:

```python
class QueryNormalizer(Protocol):
    def normalize(self, raw_query: str) -> str: ...
```

`PassthroughNormalizer` is the default implementation, returning query text unchanged. Custom normalizers can lowercase, strip punctuation, expand abbreviations, or apply stemming.

## Query Embedding

The `QueryEmbedder` protocol converts query text to a dense vector for vector store search:

```python
class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...
```

`DeterministicQueryEmbedder` (in `libs/embeddings/query_embedder.py`) wraps `DeterministicEmbeddingProvider` by creating a synthetic chunk from the query text and embedding it. For testing only. Production implementations should call a real embedding service (e.g., TEI).

## BrokerConfig

All fields on `BrokerConfig` (`libs/retrieval/broker/models.py`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | `RetrievalMode` | `HYBRID` | Which backends to query |
| `rrf_k` | `int` | `60` | RRF smoothing constant |
| `max_candidates_per_source` | `int` | `0` | Max candidates per `source_id` (0 = no cap) |
| `dense_weight` | `float` | `1.0` | RRF weight for dense results |
| `lexical_weight` | `float` | `1.5` | RRF weight for lexical results |

**Why lexical_weight defaults to 1.5**: Empirical tuning showed that lexical retrieval produces higher-precision matches for queries with specific terms (names, chapter references, technical keywords). A 1.5x weight gives BM25 results a modest boost in RRF fusion without drowning out semantic matches. Override with `--config lexical_weight=N`.

## RetrievalMode

| Mode | Backends Used | When to Use |
|------|---------------|-------------|
| `DENSE` | VectorStore only | Semantic similarity queries, conceptual matching |
| `LEXICAL` | LexicalStore only | Exact term matching, keyword lookups |
| `HYBRID` | Both | Default; combines semantic and keyword signals |

## BrokerResult

Top-level return type of `RetrievalBroker.run()`:

| Field | Type | Description |
|-------|------|-------------|
| `query` | `RetrievalQuery` | The effective query (after normalization) |
| `mode` | `RetrievalMode` | Mode used for this run |
| `candidates` | `list[FusedCandidate]` | Final ranked candidates |
| `candidate_count` | `int` | Length of `candidates` |
| `store_results` | `list[StoreResult]` | Per-store intermediate results |
| `outcome` | `BrokerOutcome` | Overall outcome classification |
| `total_latency_ms` | `float` | Wall-clock time for the full run |
| `completed_at` | `datetime` | Completion timestamp (UTC) |
| `errors` | `list[str]` | Error messages from failed stores |
| `debug` | `dict[str, Any]` | Debug/tracing payload |

## FusedCandidate

A candidate after RRF fusion with full provenance:

| Field | Type | Description |
|-------|------|-------------|
| `chunk` | `Chunk` | The matched chunk with lineage |
| `fused_score` | `float` | Combined RRF score |
| `retrieval_method` | `RetrievalMethod` | `DENSE`, `LEXICAL`, or `HYBRID` (if found by both) |
| `contributing_stores` | `list[str]` | Store IDs that returned this candidate |
| `per_store_ranks` | `dict[str, int]` | Rank in each contributing store |
| `per_store_scores` | `dict[str, float]` | Raw score from each contributing store |

## StoreResult

Per-store intermediate result for traceability:

| Field | Type | Description |
|-------|------|-------------|
| `store_id` | `str` | Identifies the store |
| `retrieval_method` | `RetrievalMethod` | `DENSE` or `LEXICAL` |
| `candidates` | `list[RetrievalCandidate]` | Raw candidates from this store |
| `candidate_count` | `int` | Number of candidates returned |
| `latency_ms` | `float` | Time spent in this store |
| `error` | `str \| None` | Error message if the store failed |

## BrokerOutcome

| Outcome | Meaning |
|---------|---------|
| `SUCCESS` | All active stores returned results |
| `PARTIAL` | At least one store failed but some results were returned |
| `FAILED` | All active stores failed; no results |
| `NO_RESULTS` | Stores succeeded but returned no matching candidates |

## Source Caps

Source caps limit how many candidates can come from any single `source_id`, enforcing diversity across documents:

- Set `max_candidates_per_source` in `BrokerConfig` (default 0 = no cap).
- Applied after RRF fusion, before `top_k` truncation.
- Iterates candidates in fused-score order, counting per `source_id`, skipping any that exceed the cap.

Use source caps when a single document dominates results and you want diverse source coverage.

## Debug Payload

The `debug` dict on `BrokerResult` contains tracing keys:

| Key | Type | Description |
|-----|------|-------------|
| `mode` | `str` | Retrieval mode used |
| `rrf_k` | `int` | RRF k parameter |
| `pre_fusion_dense_count` | `int` | Dense candidates before fusion |
| `pre_fusion_lexical_count` | `int` | Lexical candidates before fusion |
| `query_vector` | `list[float]` | First 8 dimensions of the query vector (if dense) |
| `query_vector_dimensions` | `int` | Full dimensionality of query vector (if dense) |
| `post_fusion_count` | `int` | Candidates after RRF fusion |
| `post_source_cap_count` | `int` | Candidates after source cap filtering |
| `source_cap_removals` | `int` | Number of candidates removed by source caps |

## Running the Example

```bash
uv run python examples/retrieval_broker_sample.py
```

The example creates six test chunks (three tech, three food), embeds them, populates both stores, and demonstrates:

- **Hybrid retrieval**: dense + lexical fused with RRF.
- **Dense-only retrieval**: vector similarity search.
- **Lexical-only retrieval**: term-overlap search.
- **Metadata filtering**: restricting results by `category`.
- **Source caps**: limiting candidates per source for diversity.
- **Debug payload**: inspecting internal tracing data.

## Async Retrieval Broker

`AsyncRetrievalBroker` (`libs/retrieval/broker/async_service.py`) provides the same retrieval pipeline but uses `asyncio.gather()` to run dense and lexical fanout in parallel. In hybrid mode this cuts latency from the sum of both stores to approximately the latency of the slower store.

### Async Protocols

| Protocol | File | Method |
|----------|------|--------|
| `AsyncVectorStore` | `libs/retrieval/stores/async_protocols.py` | `async_search(query, query_vector)` |
| `AsyncLexicalStore` | `libs/retrieval/stores/async_protocols.py` | `async_search(query)` |
| `AsyncQueryEmbedder` | `libs/retrieval/broker/async_protocols.py` | `async_embed_query(text)` |

### Async Memory Store Wrappers

For local testing, async wrappers delegate to the sync in-memory stores:

- `AsyncMemoryVectorStore` (`libs/retrieval/stores/async_memory_vector_store.py`)
- `AsyncMemoryLexicalStore` (`libs/retrieval/stores/async_memory_lexical_store.py`)

### Bootstrap Wiring

The `ServiceRegistry` includes an `async_retrieval_broker` field. The bootstrap creates it by wrapping the same underlying stores with async wrappers and a `_SyncToAsyncEmbedder` adapter.

### Retry Support

Both the sync `RetrievalBroker` and `AsyncRetrievalBroker` accept an optional `RetryConfig` from `libs/resilience.py`. When configured, store calls are wrapped with `resilient_call()` (sync) or `async_resilient_call()` (async) for exponential backoff on transient errors.

## Source Files

| File | Description |
|------|-------------|
| `libs/retrieval/broker/__init__.py` | Public API re-exports |
| `libs/retrieval/broker/models.py` | `BrokerConfig`, `BrokerResult`, `FusedCandidate`, `StoreResult`, `RetrievalMode`, `BrokerOutcome` |
| `libs/retrieval/broker/protocols.py` | `QueryEmbedder`, `QueryNormalizer`, `PassthroughNormalizer` |
| `libs/retrieval/broker/async_protocols.py` | `AsyncQueryEmbedder` |
| `libs/retrieval/broker/fusion.py` | `reciprocal_rank_fusion` implementation |
| `libs/retrieval/broker/dedup.py` | `apply_source_caps` implementation |
| `libs/retrieval/broker/service.py` | `RetrievalBroker` (sync) orchestrator |
| `libs/retrieval/broker/async_service.py` | `AsyncRetrievalBroker` (async, parallel fanout) |
| `libs/retrieval/stores/async_protocols.py` | `AsyncVectorStore`, `AsyncLexicalStore` |
| `libs/retrieval/stores/async_memory_vector_store.py` | Async wrapper around any `VectorStore` |
| `libs/retrieval/stores/async_memory_lexical_store.py` | Async wrapper around any `LexicalStore` |
| `libs/embeddings/query_embedder.py` | `DeterministicQueryEmbedder` for testing |
| `examples/retrieval_broker_sample.py` | Runnable demo |
