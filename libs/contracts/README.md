# libs/contracts/

Shared domain models for the Drifter RAG pipeline. Every type is a frozen dataclass with validation, metadata, lineage, and version information.

## Type Flow

```
SourceDocumentRef -> RawDocument -> CanonicalDocument -> Block -> Chunk
  -> ChunkEmbedding -> RetrievalCandidate -> RankedCandidate
  -> ContextPack -> Citation -> GeneratedAnswer
```

No subsystem logic lives here — only data definitions.

## Modules

### `common.py`

Type aliases and shared enums:

```python
SourceId = str
DocumentId = str
BlockId = str
ChunkId = str
EmbeddingId = str
TraceId = str
RunId = str

class BlockType(Enum):      # HEADING, PARAGRAPH, TABLE, CODE, LIST, ...
class RetrievalMethod(Enum): # DENSE, LEXICAL, HYBRID
class SelectionReason(Enum): # TOP_RANKED, DIVERSITY, RECENCY, AUTHORITY
```

### `documents.py`

Document lifecycle types:

| Type | Purpose |
|------|---------|
| `SourceDocumentRef` | Reference to a fetched source (URI, content hash, version) |
| `RawDocument` | Raw bytes + MIME type before parsing |
| `Block` | Structural element within a document (paragraph, heading, table, etc.) |
| `CanonicalDocument` | Parsed document with ordered list of Blocks |

### `chunks.py`

| Type | Purpose |
|------|---------|
| `ChunkLineage` | Full provenance: source, document, blocks, strategy, parser version |
| `Chunk` | Text segment with content hash, token count, lineage, byte offsets |

### `embeddings.py`

| Type | Purpose |
|------|---------|
| `ChunkEmbedding` | Dense vector with model ID, version, dimensions |

### `retrieval.py`

| Type | Purpose |
|------|---------|
| `RetrievalQuery` | Normalized query with trace ID, top_k, filters |
| `RetrievalCandidate` | Chunk + score + retrieval method + store ID |
| `RankedCandidate` | Candidate + rank + rerank score + reranker ID |

### `context.py`

| Type | Purpose |
|------|---------|
| `ContextItem` | Chunk + rank + token count + selection reason |
| `ContextPack` | Evidence list + total tokens + budget + diversity score |

### `generation.py`

| Type | Purpose |
|------|---------|
| `TokenUsage` | Prompt tokens, completion tokens, total |
| `Citation` | Claim + chunk reference + source ID + confidence |
| `GeneratedAnswer` | Answer text + citations + model ID + token usage |

### `evaluation.py`

| Type | Purpose |
|------|---------|
| `EvaluationCase` | Ground-truth query + expected answer + relevant chunk IDs |
| `EvaluationResult` | Per-case metrics + config + timestamp |

## Design Principles

- All types are `@dataclass(frozen=True)` — immutable after creation
- All types validate in `__post_init__()` — fail fast on bad data
- Schema versioning via `schema_version` field enables forward compatibility
- ACL fields on documents and chunks enable access control
- Metadata dicts allow extensibility without schema changes
