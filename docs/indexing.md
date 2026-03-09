# Embedding and Indexing Pipeline

## Architecture Role

Embedding and indexing sit between **chunking** and **retrieval** in the RAG pipeline:

```
ingestion → parsing → chunking → [embeddings → indexing] → retrieval
```

The embedding subsystem converts `Chunk` objects into dense vector representations (`ChunkEmbedding`). The indexing subsystem orchestrates persisting chunks, embeddings, and writing to both vector and lexical indexes. Together they prepare data for the query plane.

## EmbeddingProvider Protocol

All embedding logic is accessed through the `EmbeddingProvider` protocol (`libs/embeddings/protocols.py`):

```python
class EmbeddingProvider(Protocol):
    def model_info(self) -> EmbeddingModelInfo: ...
    def embed_chunks(self, chunks: list[Chunk]) -> list[ChunkEmbedding]: ...
```

Implementations must:

- Return an `EmbeddingModelInfo` describing the model's identity and capabilities.
- Accept a batch of `Chunk` objects and return one `ChunkEmbedding` per chunk.
- Be stateless with respect to chunk storage (the indexing service handles persistence).

### DeterministicEmbeddingProvider

A test provider (`libs/embeddings/mock_provider.py`) that hashes chunk content into repeatable vectors. Same input always produces the same output, making tests fully deterministic. Configured with `model_id`, `model_version`, and `dimensions`.

## EmbeddingModelInfo

Immutable descriptor for an embedding model (`libs/embeddings/models.py`):

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | `str` | Unique identifier for the model |
| `model_version` | `str` | Version string (used for idempotency checks) |
| `dimensions` | `int` | Vector dimensionality (must be >= 1) |
| `max_tokens` | `int` | Maximum input tokens the model supports |
| `metadata` | `dict` | Arbitrary extra metadata |

## Storage Protocols

The indexing subsystem defines four storage interfaces (`libs/indexing/protocols.py`). All are `@runtime_checkable` protocols.

### ChunkRepository

Persistence for chunk records:

| Method | Description |
|--------|-------------|
| `store(chunk)` | Store a single chunk |
| `store_batch(chunks)` | Store multiple chunks |
| `get(chunk_id)` | Retrieve by ID |
| `get_by_document(document_id)` | All chunks for a document |
| `delete_by_document(document_id)` | Remove all chunks for a document |

### EmbeddingRepository

Persistence for chunk embeddings:

| Method | Description |
|--------|-------------|
| `store(embedding)` | Store a single embedding |
| `store_batch(embeddings)` | Store multiple embeddings |
| `get_by_chunk(chunk_id)` | All embeddings for a chunk (any model) |
| `get_by_chunk_and_model(chunk_id, model_id, model_version)` | Embedding for a specific model version |
| `list_by_model(model_id, model_version)` | All embeddings for a model version |
| `delete_by_chunk(chunk_id)` | Remove all embeddings for a chunk |

### VectorIndexWriter

Writes embeddings to a vector store for dense retrieval:

| Method | Description |
|--------|-------------|
| `write_batch(embeddings, chunks)` | Write embedding vectors with chunk metadata; returns count |
| `delete_by_chunk_ids(chunk_ids)` | Remove entries by chunk ID |

### LexicalIndexWriter

Writes chunk text to a full-text index for lexical retrieval:

| Method | Description |
|--------|-------------|
| `write_batch(chunks)` | Index chunk text; returns count |
| `delete_by_chunk_ids(chunk_ids)` | Remove entries by chunk ID |

## IndexingService Orchestration

`IndexingService` (`libs/indexing/service.py`) wires together an `EmbeddingProvider` and the four storage protocols. It exposes two methods:

### `run(chunks, run_id) -> IndexingResult`

Executes the full indexing pipeline:

1. Get `model_info` from the embedding provider.
2. **Filter**: skip chunks already embedded for this `model_id` + `model_version` (idempotency).
3. **Embed**: call `embed_chunks()` on the filtered list.
4. **Store embeddings**: `embedding_repo.store_batch()` for new embeddings.
5. **Store chunks**: `chunk_repo.store_batch()` for all chunks.
6. **Vector index**: `vector_writer.write_batch()` with all embeddings (new and existing).
7. **Lexical index**: `lexical_writer.write_batch()` with all chunks.
8. **Return** `IndexingResult` with counts and outcome.

### `needs_reembedding(chunk) -> bool`

Returns `True` if a chunk has no embedding for the current model version but has an embedding for a different version. Useful for detecting when a model upgrade requires re-processing.

## IndexingResult and Outcomes

`IndexingResult` (`libs/indexing/models.py`) summarises a completed run:

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `RunId` | Identifier for this indexing run |
| `chunks_received` | `int` | Total chunks passed in |
| `chunks_embedded` | `int` | Chunks newly embedded (0 on idempotent re-run) |
| `chunks_indexed_vector` | `int` | Chunks written to the vector index |
| `chunks_indexed_lexical` | `int` | Chunks written to the lexical index |
| `outcome` | `IndexingOutcome` | SUCCESS, PARTIAL, FAILED, or SKIPPED |
| `model_info` | `EmbeddingModelInfo` | The embedding model used |
| `completed_at` | `datetime` | Completion timestamp |
| `errors` | `list[str]` | Any error messages |

**Outcome semantics:**

| Outcome | Meaning |
|---------|---------|
| `SUCCESS` | All chunks embedded and indexed |
| `PARTIAL` | Some chunks could not be embedded or indexed |
| `FAILED` | Embedding failed entirely |
| `SKIPPED` | Empty chunk list; nothing to do |

## In-Memory Adapters

The `libs/adapters/memory/` package provides in-memory implementations of all four storage protocols:

- `MemoryChunkRepository`
- `MemoryEmbeddingRepository`
- `MemoryVectorIndexWriter`
- `MemoryLexicalIndexWriter`

These require no external services and produce repeatable results. Use them for unit tests, integration tests, and the example script.

## Running the Example

```bash
uv run python examples/embedding_sample.py
```

The example creates three test chunks, embeds and indexes them with a `DeterministicEmbeddingProvider`, then demonstrates:

- **Idempotency**: running the same chunks again produces `chunks_embedded = 0`.
- **Model version detection**: `needs_reembedding()` returns `True` after upgrading the provider to a new model version.

## Source Files

| File | Description |
|------|-------------|
| `libs/embeddings/__init__.py` | Embeddings subsystem public API |
| `libs/embeddings/protocols.py` | `EmbeddingProvider` protocol |
| `libs/embeddings/models.py` | `EmbeddingModelInfo` dataclass |
| `libs/embeddings/mock_provider.py` | `DeterministicEmbeddingProvider` |
| `libs/indexing/__init__.py` | Indexing subsystem public API |
| `libs/indexing/protocols.py` | Storage and index writer protocols |
| `libs/indexing/models.py` | `IndexingResult`, `IndexingOutcome` |
| `libs/indexing/service.py` | `IndexingService` orchestrator |
| `libs/adapters/memory/` | In-memory adapter implementations |
| `examples/embedding_sample.py` | Runnable demo |
