# libs/retrieval/

Query execution layer. Contains two sub-packages: **stores** (backend protocols and implementations) and **broker** (orchestration and hybrid fusion).

## Boundary

- **Consumes:** RetrievalQuery (normalized query + trace ID + top_k + filters)
- **Produces:** BrokerResult (fused candidates with per-store scores)
- **Rule:** Returns candidates, never answers.

## Sub-packages

### `stores/` — Retrieval Store Protocols

Defines `VectorStore` and `LexicalStore` protocols plus implementations.

```python
class VectorStore(Protocol):
    def store_id(self) -> str: ...
    def add(self, embedding: ChunkEmbedding, chunk: Chunk) -> None: ...
    def search(self, query: RetrievalQuery, query_vector: list[float]) -> list[RetrievalCandidate]: ...

class LexicalStore(Protocol):
    def store_id(self) -> str: ...
    def add(self, chunk: Chunk) -> None: ...
    def search(self, query: RetrievalQuery) -> list[RetrievalCandidate]: ...
```

| Implementation | Backend | Location |
|----------------|---------|----------|
| `MemoryVectorStore` | In-memory (cosine similarity) | `stores/memory_vector_store.py` |
| `MemoryLexicalStore` | In-memory (term matching) | `stores/memory_lexical_store.py` |
| `QdrantVectorStore` | Qdrant | `libs/adapters/qdrant/` |
| `OpenSearchVectorStore` | OpenSearch | `libs/adapters/opensearch/` |
| `OpenSearchLexicalStore` | OpenSearch | `libs/adapters/opensearch/` |

### `broker/` — Retrieval Broker

Orchestrates dense, lexical, and hybrid retrieval with Reciprocal Rank Fusion (RRF).

```python
class RetrievalBroker:
    def __init__(
        self,
        vector_store: VectorStore,
        lexical_store: LexicalStore,
        query_embedder: QueryEmbedder,
        config: BrokerConfig | None = None,
    ) -> None: ...

    def run(self, query: RetrievalQuery) -> BrokerResult: ...
```

Pipeline: normalize query → embed query → fan out to stores → fuse results (RRF) → apply caps → return.

## Retrieval Modes

| Mode | Stores Used | Use Case |
|------|------------|----------|
| `DENSE` | VectorStore only | Semantic similarity search |
| `LEXICAL` | LexicalStore only | Keyword/term matching |
| `HYBRID` | Both + RRF fusion | Best of both worlds (default) |

## RRF Fusion

`fusion.py` implements Reciprocal Rank Fusion:
- Combines rankings from multiple stores into a single fused score
- Configurable `rrf_k` parameter (default 60)
- Supports asymmetric weights (`dense_weight`, `lexical_weight`)
- Deterministic tie-breaking

## Key Result Types

| Type | Fields |
|------|--------|
| `BrokerResult` | query, mode, candidates (fused), store_results, outcome, latency, errors, debug |
| `FusedCandidate` | chunk, fused_score, retrieval_method, contributing_stores, per_store_ranks/scores |
| `StoreResult` | store_id, method, candidates, count, latency, error |
| `BrokerOutcome` | SUCCESS, PARTIAL, FAILED, NO_RESULTS |

## Broker Protocols

| Protocol | Purpose |
|----------|---------|
| `QueryEmbedder` | Embeds query text to a dense vector |
| `QueryNormalizer` | Preprocesses query text (default: passthrough) |

## Testing

Uses `MemoryVectorStore` and `MemoryLexicalStore`. RRF fusion has comprehensive tests covering overlap, asymmetric weights, and deterministic ordering.
