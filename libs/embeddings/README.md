# libs/embeddings/

Dense vector generation for chunks. Provider-agnostic via protocol abstraction.

## Boundary

- **Consumes:** list[Chunk]
- **Produces:** list[ChunkEmbedding]
- **Must not** depend on a specific embedding provider.

## Protocol

```python
class EmbeddingProvider(Protocol):
    def model_info(self) -> EmbeddingModelInfo: ...
    def embed_chunks(self, chunks: list[Chunk]) -> list[ChunkEmbedding]: ...
```

## Implementations

| Class | Purpose |
|-------|---------|
| `DeterministicEmbeddingProvider` | Hash-based vectors for testing. Deterministic and reproducible. |
| `DeterministicQueryEmbedder` | Wraps the mock provider for query embedding |
| `TeiEmbeddingProvider` | Real embedding via TEI `/embed` (in `libs/adapters/tei/`) |
| `TeiQueryEmbedder` | Real query embedding via TEI |
| `OllamaEmbeddingProvider` | Real embedding via Ollama `/api/embed` (in `libs/adapters/ollama/`) |
| `OllamaQueryEmbedder` | Real query embedding via Ollama |
| `VllmEmbeddingProvider` | Real embedding via vLLM `/v1/embeddings` (in `libs/adapters/vllm/`) |
| `VllmQueryEmbedder` | Real query embedding via vLLM |
| `OpenRouterEmbeddingProvider` | Real embedding via OpenRouter API (in `libs/adapters/openrouter/`) |
| `OpenRouterQueryEmbedder` | Real query embedding via OpenRouter |

## Key Types

| Type | Purpose |
|------|---------|
| `ChunkEmbedding` | Vector + model ID + model version + dimensions + timestamp |
| `EmbeddingModelInfo` | Model metadata: ID, version, dimensions, max tokens |

## Model Versioning

Every `ChunkEmbedding` records the model ID and version that produced it. This enables:
- Detecting when chunks need re-embedding after a model change
- Ensuring query vectors match the model used for indexing

## Testing

`DeterministicEmbeddingProvider` produces vectors by hashing `chunk.content_hash` seeded with model identity. Fully reproducible, no network calls.
