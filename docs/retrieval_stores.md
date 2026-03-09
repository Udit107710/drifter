# Retrieval Stores

## Architecture Role

Retrieval stores sit between **indexing** and the **retrieval broker** in the RAG pipeline:

```
ingestion → parsing → chunking → embeddings → indexing → [retrieval stores] → reranking
```

A retrieval store accepts a `RetrievalQuery` and returns scored `RetrievalCandidate` objects. Two protocol interfaces define the contract: `VectorStore` for dense (embedding-based) search and `LexicalStore` for keyword-based search. Each protocol is independently implementable, so backends can be swapped without changing query-plane logic.

## VectorStore Protocol

Defined in `libs/retrieval/stores/protocols.py`. Any class implementing these methods satisfies the `@runtime_checkable` protocol:

```python
class VectorStore(Protocol):
    @property
    def store_id(self) -> str: ...

    def search(self, query: RetrievalQuery, query_vector: list[float]) -> list[RetrievalCandidate]: ...

    def count(self) -> int: ...
```

The `search` method takes a `RetrievalQuery` (with `top_k` and optional `filters`) and a dense `query_vector`. It computes similarity between the query vector and all indexed vectors, applies metadata filters, and returns up to `top_k` candidates sorted by descending score.

The in-memory implementation uses **cosine similarity**: `dot(a, b) / (||a|| * ||b||)`.

## LexicalStore Protocol

```python
class LexicalStore(Protocol):
    @property
    def store_id(self) -> str: ...

    def search(self, query: RetrievalQuery) -> list[RetrievalCandidate]: ...

    def count(self) -> int: ...
```

The `search` method uses term matching against `query.normalized_query`. No query vector is needed. The in-memory implementation tokenizes content and query into lowercase terms and scores by the fraction of query terms found in each chunk:

```
score = (matching query terms) / (total query terms)
```

Chunks with zero matching terms are excluded.

## Metadata Filtering

Both protocols support metadata filtering via `query.filters`, a `dict[str, Any]`. A chunk passes the filter only if every key-value pair in `filters` matches the chunk's `metadata` dictionary. An empty `filters` dict matches all chunks.

Example — retrieve only chunks with `category == "tech"`:

```python
query = RetrievalQuery(
    raw_query="machine learning",
    normalized_query="machine learning",
    trace_id="trace-001",
    top_k=5,
    filters={"category": "tech"},
)
```

## RetrievalCandidate Output

Every search returns a list of `RetrievalCandidate` objects (`libs/contracts/retrieval.py`):

| Field | Type | Description |
|-------|------|-------------|
| `chunk` | `Chunk` | The matched chunk with full lineage |
| `score` | `float` | Relevance score (higher is better) |
| `retrieval_method` | `RetrievalMethod` | `DENSE` for vector stores, `LEXICAL` for lexical stores |
| `store_id` | `str` | Identifies which store produced the result |

## In-Memory Adapters

### MemoryVectorStore

`libs/retrieval/stores/memory_vector_store.py`

Dict-backed store keyed by `ChunkId`. Each entry holds a `(ChunkEmbedding, Chunk)` tuple. Supports `add()` for single items and `add_batch()` for bulk indexing. Search computes cosine similarity between the query vector and every stored vector, applies metadata filters, sorts descending, and truncates to `top_k`.

### MemoryLexicalStore

`libs/retrieval/stores/memory_lexical_store.py`

Dict-backed store keyed by `ChunkId`. Supports `add()` and `add_batch()`. Search tokenizes both query and chunk content into lowercase terms, scores by term overlap fraction, applies metadata filters, sorts descending, and truncates to `top_k`.

Both adapters require no external services and produce repeatable results. Use them for unit tests, integration tests, and example scripts.

## Adapter Stubs

Three placeholder classes demonstrate the protocol interface shape for production backends. Each raises `NotImplementedError` on all methods:

| Class | File | Backend |
|-------|------|---------|
| `QdrantVectorStore` | `libs/retrieval/stores/qdrant_vector_store.py` | Qdrant dense retrieval |
| `OpenSearchLexicalStore` | `libs/retrieval/stores/opensearch_lexical_store.py` | OpenSearch full-text search |
| `OpenSearchVectorStore` | `libs/retrieval/stores/opensearch_vector_store.py` | OpenSearch k-NN plugin |

These stubs exist so the codebase has concrete adapter files ready for implementation. Each satisfies the relevant protocol interface (`VectorStore` or `LexicalStore`).

## Running the Example

```bash
uv run python examples/retrieval_stores_sample.py
```

The example creates six test chunks (three about machine learning, three about cooking), embeds them with `DeterministicEmbeddingProvider`, populates both stores, and demonstrates:

- **Vector search**: similarity-based retrieval using an embedded query.
- **Lexical search**: term-overlap retrieval using normalized query text.
- **Metadata filtering**: restricting results by `category` metadata.
- **Empty results**: a query with no matching terms returns an empty list.

## Source Files

| File | Description |
|------|-------------|
| `libs/retrieval/stores/__init__.py` | Public API for retrieval store protocols |
| `libs/retrieval/stores/protocols.py` | `VectorStore` and `LexicalStore` protocols |
| `libs/retrieval/stores/memory_vector_store.py` | `MemoryVectorStore` in-memory adapter |
| `libs/retrieval/stores/memory_lexical_store.py` | `MemoryLexicalStore` in-memory adapter |
| `libs/retrieval/stores/qdrant_vector_store.py` | `QdrantVectorStore` placeholder |
| `libs/retrieval/stores/opensearch_lexical_store.py` | `OpenSearchLexicalStore` placeholder |
| `libs/retrieval/stores/opensearch_vector_store.py` | `OpenSearchVectorStore` placeholder |
| `libs/contracts/retrieval.py` | `RetrievalQuery`, `RetrievalCandidate`, `RankedCandidate` |
| `examples/retrieval_stores_sample.py` | Runnable demo |
