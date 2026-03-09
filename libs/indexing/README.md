# libs/indexing/

Writes chunk embeddings to vector stores and chunk text to lexical stores.

## Boundary

- **Consumes:** list[Chunk] + RunId
- **Produces:** IndexingResult
- **Owns:** Embedding generation, chunk/embedding persistence, index writes

## Service

```python
class IndexingService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        chunk_repo: ChunkRepository,
        embedding_repo: EmbeddingRepository,
        vector_writer: VectorIndexWriter,
        lexical_writer: LexicalIndexWriter,
    ) -> None: ...

    def run(self, chunks: list[Chunk], run_id: RunId) -> IndexingResult: ...
    def needs_reembedding(self, chunk: Chunk) -> bool: ...
```

Pipeline: embed chunks → store embeddings → store chunks → write vector index → write lexical index.

## Protocols

| Protocol | Methods | Purpose |
|----------|---------|---------|
| `VectorIndexWriter` | `write_batch`, `delete_by_chunk_ids` | Writes to vector store |
| `LexicalIndexWriter` | `write_batch`, `delete_by_chunk_ids` | Writes to lexical store |
| `ChunkRepository` | `store`, `store_batch`, `get`, `get_by_document`, `delete_by_document` | Chunk persistence |
| `EmbeddingRepository` | `store`, `store_batch`, `get_by_chunk`, `get_by_chunk_and_model` | Embedding persistence |

## Result Types

| Type | Fields |
|------|--------|
| `IndexingResult` | run_id, chunks_received/embedded/indexed_vector/indexed_lexical, outcome, model_info, errors |
| `IndexingOutcome` | SUCCESS, PARTIAL, FAILED, SKIPPED |
| `ChunkError` | chunk_id, stage, error, classification (TRANSIENT/PERMANENT) |

## Index Lifecycle

`lifecycle.py` provides `IndexRegistry` and `IndexVersion` for tracking index state:
- Which model version an index was built with
- When it was last updated
- Whether it needs rebuilding

## In-Memory Adapters

All protocols have in-memory implementations in `libs/adapters/memory/`:
- `MemoryChunkRepository`
- `MemoryEmbeddingRepository`
- `MemoryVectorIndexWriter`
- `MemoryLexicalIndexWriter`

## Testing

Uses in-memory adapters with `DeterministicEmbeddingProvider`. No external stores required.
