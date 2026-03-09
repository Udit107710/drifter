# Data Contracts

Canonical domain models for the Drifter RAG pipeline. Every `libs/` subsystem depends on `libs/contracts/` and nothing else.

---

## Rationale

Loose dicts and ad-hoc strings break silently at subsystem boundaries. Typed dataclasses with `__post_init__` validation catch invalid data at construction time, before it propagates downstream. All models are frozen (immutable) to prevent accidental mutation during pipeline processing.

**Why not Pydantic?** We use stdlib `dataclasses` to keep the contracts dependency-free. Validation lives in `__post_init__`; mypy strict mode catches type errors at development time.

---

## Ownership Table

| Module | Models | Owning subsystem |
|---|---|---|
| `common.py` | `BlockType`, `SelectionReason`, `RetrievalMethod`, ID aliases | shared |
| `documents.py` | `SourceDocumentRef`, `RawDocument`, `CanonicalDocument`, `Block` | ingestion, parsing |
| `chunks.py` | `Chunk`, `ChunkLineage` | chunking |
| `embeddings.py` | `ChunkEmbedding` | embeddings |
| `retrieval.py` | `RetrievalQuery`, `RetrievalCandidate`, `RankedCandidate` | retrieval, reranking |
| `context.py` | `ContextPack`, `ContextItem` | context_builder |
| `generation.py` | `GeneratedAnswer`, `Citation`, `TokenUsage` | generation |
| `evaluation.py` | `EvaluationCase`, `EvaluationResult` | evaluation |

---

## Key Distinctions

### Document vs Block vs Chunk

- **SourceDocumentRef** — a pointer to an external source (URI, content hash, version). Created by ingestion. Does not contain document content.
- **RawDocument** — the fetched bytes plus MIME type. Bridges ingestion → parsing.
- **CanonicalDocument** — a parsed, structured representation containing a list of Blocks. Preserves document structure.
- **Block** — a structural element (heading, paragraph, table, code, list, image caption). Blocks are the parser's output units.
- **Chunk** — a retrieval-ready text unit derived from one or more Blocks. Carries token count, byte offsets, and full lineage. This is what gets embedded and indexed.

The split is intentional: documents preserve structure for display and navigation; chunks are optimized for retrieval granularity.

### Candidate vs Ranked Candidate

- **RetrievalCandidate** — a chunk returned by a retrieval store (vector or lexical) with a raw relevance score and the retrieval method used.
- **RankedCandidate** — a candidate after reranking with a precision score from a cross-encoder or feature-based model. Carries a 1-based rank.

Keeping these separate ensures that reranking is always an explicit, observable stage — never silently merged into retrieval.

### Context Pack vs Generated Answer

- **ContextPack** — the selected evidence set passed to the LLM. Contains items with token counts, selection reasons, and a diversity score. The context builder owns this.
- **GeneratedAnswer** — the LLM's response with citations back to specific chunks, token usage, and a trace ID for debugging.

The generation stage receives a ContextPack and produces a GeneratedAnswer. It never decides which evidence to include — that's the context builder's job.

---

## Lineage Flow

Every chunk maintains a complete chain of provenance back to its source:

```
SourceDocumentRef (source_id, uri, content_hash, version)
        ↓
RawDocument (raw bytes + mime type)
        ↓
CanonicalDocument (document_id, blocks[], parser_version)
        ↓
Block (block_id, block_type, position)
        ↓
Chunk (chunk_id, block_ids[], lineage: ChunkLineage)
        ↓
ChunkEmbedding (embedding_id, model_id, model_version, dimensions)
```

`ChunkLineage` captures: source_id, document_id, block_ids, chunk_strategy, parser_version, and created_at. This enables:
- Tracing any retrieval result back to its original source
- Re-chunking when strategies change (identify affected chunks by strategy field)
- Re-embedding when models change (identify affected embeddings by model_version)
- Tombstone propagation when sources are deleted

---

## Placeholder Fields

Several fields exist as placeholders for future capabilities:

- **`acl`** on `CanonicalDocument` and `Chunk` — access control lists for multi-tenant filtering. Currently defaults to empty list.
- **`authority_score`** and **`freshness_hint`** on `SourceDocumentRef` — for authority-weighted and recency-weighted retrieval. Currently optional/None.
- **`filters`** on `RetrievalQuery` — for metadata-based filtering at query time.

---

## Validation

All models validate in `__post_init__`:
- Required string fields must not be empty
- Numeric bounds are enforced (version >= 1, token_count >= 1, rank >= 1, etc.)
- Cross-field invariants are checked (vector length == dimensions, total_tokens <= token_budget, etc.)
- Confidence and diversity scores are bounded [0.0, 1.0]
