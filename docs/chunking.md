# Chunking Subsystem

## Architecture Role

Chunking sits between **parsing** and **embedding/indexing** in the RAG pipeline:

```
ingestion → parsing → [chunking] → embeddings → indexing → retrieval
```

It consumes a `CanonicalDocument` (an ordered list of typed `Block` objects produced by a parser) and produces a `list[Chunk]` with full lineage, token counts, and deterministic IDs. Chunks are the atomic units that get embedded and indexed for retrieval.

Chunking never flattens document structure. Every chunk records which blocks it spans, the section heading hierarchy at that point, and byte offsets back into the concatenated document.

## Strategies

Three chunking strategies are provided. All satisfy the `ChunkingStrategy` protocol.

### Fixed Window (`FixedWindowChunker`)

Slides a token-level window across the concatenated block contents with configurable overlap. Simple and predictable. Best for homogeneous documents where structure is not meaningful.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | `int` | 256 | Maximum tokens per chunk |
| `overlap` | `int` | 64 | Tokens shared between consecutive chunks |
| `min_chunk_size` | `int` | 32 | Trailing chunk below this size is discarded |

**Constraints:** `chunk_size > 0`, `0 <= overlap < chunk_size`, `0 < min_chunk_size <= chunk_size`.

### Recursive Structure-Aware (`RecursiveStructureChunker`)

Walks document blocks respecting structural boundaries (headings, block-type transitions). Produces chunks that align with document sections when possible. Includes runt-merging logic to avoid tiny trailing chunks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_chunk_size` | `int` | 512 | Maximum tokens per chunk |
| `min_chunk_size` | `int` | 64 | Chunks below this size are merged with neighbours |
| `prefer_structural` | `bool` | `True` | Break on block-type transitions when above `min_chunk_size` |

**Constraints:** `max_chunk_size > 0`, `0 < min_chunk_size <= max_chunk_size`.

**Behaviour details:**

- Headings always start a new chunk.
- When `prefer_structural` is enabled, a block-type transition (e.g., paragraph to list) triggers a flush if the buffer already meets `min_chunk_size`.
- Oversized single blocks are split into sub-chunks by token count.
- Runt merging: forward-merge first (merge runt into the next chunk), then backward-merge the last runt into the previous chunk.

### Parent-Child (`ParentChildChunker`)

Produces a two-level hierarchy: large parent chunks for broad context and smaller overlapping child chunks for precise retrieval. At query time, retrieve children for precision but expand to parents for context.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parent_chunk_size` | `int` | 1024 | Maximum tokens per parent chunk |
| `child_chunk_size` | `int` | 256 | Maximum tokens per child chunk |
| `child_overlap` | `int` | 64 | Token overlap between consecutive children |
| `min_child_size` | `int` | 32 | Trailing child below this size is discarded |

**Constraints:** `parent_chunk_size > child_chunk_size`, `child_chunk_size > 0`, `0 <= child_overlap < child_chunk_size`, `0 < min_child_size <= child_chunk_size`.

**Output order:** All parent chunks first, then all child chunks.

**Metadata:**

- Parent chunks carry `is_parent=True` and `child_chunk_ids` (list of child IDs).
- Child chunks carry `parent_chunk_id` and `child_index`.

## Chunk Lineage

Every `Chunk` carries a `ChunkLineage` record:

| Field | Description |
|-------|-------------|
| `source_id` | The original source document reference |
| `document_id` | The canonical document ID |
| `block_ids` | Ordered list of block IDs this chunk spans |
| `chunk_strategy` | Name of the strategy that produced this chunk |
| `parser_version` | Parser version that produced the source blocks |
| `created_at` | Timestamp of chunk creation |

This enables full traceability from any retrieved chunk back to the exact source document, parser version, and structural blocks.

## Section Path Semantics

Each chunk's metadata includes a `section_path`: a list of heading texts representing the heading hierarchy at the chunk's position in the document. The `SectionTracker` maintains a stack of `(level, heading_text)` tuples. When a new heading is encountered, all headings at the same or deeper level are popped before the new one is pushed.

Example for a document with:

```
# Introduction       → section_path: ["Introduction"]
## Background        → section_path: ["Introduction", "Background"]
## Methods           → section_path: ["Introduction", "Methods"]
# Results            → section_path: ["Results"]
```

For fixed-window chunks, the section path comes from the first block in the window. For recursive chunks, it comes from the first block in the accumulated buffer. This provides approximate section context without requiring a heading in every chunk.

## Byte Offset Semantics

Each chunk records `byte_offset_start` and `byte_offset_end` as UTF-8 byte positions within the concatenated document (blocks joined by newline separators). These offsets are computed from per-block cumulative byte offsets using `compute_block_byte_offsets`.

The byte range spans from the start of the first block in `block_ids` to the end of the last block. This allows downstream systems to map chunks back to exact positions in the original parsed content.

For parent-child chunks, child byte offsets are computed relative to the parent's position within the document, giving each child precise document-level coordinates.

## Token Counting

The default token counter is `WhitespaceTokenCounter`, which splits on whitespace and counts the resulting tokens. This is zero-dependency and suitable for testing and approximate budgeting.

For production use, implement the `TokenCounter` protocol with a model-specific tokenizer (e.g., tiktoken for OpenAI models, the HuggingFace tokenizer for your embedding model):

```python
from libs.chunking.protocols import TokenCounter

class TiktokenCounter:
    def __init__(self, model: str = "cl100k_base"):
        import tiktoken
        self._enc = tiktoken.get_encoding(model)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

chunker = FixedWindowChunker(
    config=FixedWindowConfig(chunk_size=256),
    token_counter=TiktokenCounter(),
)
```

All strategies accept an optional `token_counter` parameter. When omitted, `WhitespaceTokenCounter` is used.

## Deterministic ID Generation

Chunk IDs are generated by `generate_chunk_id` using SHA-256:

1. Concatenate: `"{document_id}|{strategy}|{sequence_index}|{content}"`
2. Compute SHA-256 hex digest, truncate to 24 characters.
3. Prefix with `chk:`.

Result format: `chk:a1b2c3d4e5f6a1b2c3d4e5f6`

Content hashes use the full SHA-256 digest prefixed with `sha256:`.

Both are fully deterministic: the same document, strategy, and content always produce the same IDs. This supports idempotent re-ingestion and deduplication.

## Evaluation Hooks

Changes to chunking parameters directly affect retrieval quality. Key considerations:

**Recall@k sensitivity to chunk size:**

- Smaller chunks increase the number of candidates, improving Recall@k but potentially fragmenting context.
- Larger chunks preserve more context per candidate but may dilute relevance, reducing Precision@k.
- Optimal chunk size depends on the embedding model's effective input length and the query type.

**Overlap effects:**

- Overlap ensures that information near chunk boundaries is not lost to splitting.
- Higher overlap increases the total number of chunks (storage and indexing cost) but improves boundary coverage.
- Zero overlap is acceptable for well-structured documents where section breaks are natural boundaries.

**Strategy-specific considerations:**

- Fixed-window: tune `chunk_size` and `overlap` against Recall@k and MRR on your evaluation dataset. Start with the embedding model's recommended input size.
- Recursive: `prefer_structural=True` tends to produce more semantically coherent chunks, improving faithfulness scores. Evaluate against `prefer_structural=False` on your dataset.
- Parent-child: evaluate children for retrieval (Precision@k) and parents for generation context (faithfulness, completeness). The two levels serve different quality dimensions.

**Failure modes:**

- `min_chunk_size` too high: aggressive merging can combine unrelated sections.
- `min_chunk_size` too low: tiny chunks produce poor embeddings and waste index capacity.
- `overlap >= chunk_size`: rejected by validation (would cause infinite loops).
- Single blocks exceeding `max_chunk_size`: handled by sub-chunking, but very long code blocks or tables may lose coherence when split.

**Recommended evaluation protocol:**

1. Fix your embedding model and retrieval method.
2. Sweep chunk sizes (e.g., 128, 256, 512, 1024 tokens) and overlaps (0, 25%, 50%).
3. Measure Recall@k, Precision@k, MRR, and NDCG on a held-out query set (BEIR, Natural Questions, or domain-specific).
4. Evaluate each strategy separately, then compare.
5. Record all parameters in experiment metadata for reproducibility.

## Usage Guide

### Basic usage

```python
from libs.chunking import FixedWindowChunker, FixedWindowConfig

chunker = FixedWindowChunker(FixedWindowConfig(chunk_size=256, overlap=64))
chunks = chunker.chunk(canonical_doc)

for chunk in chunks:
    print(chunk.chunk_id, chunk.token_count, chunk.content[:50])
```

### Recursive chunking with structural awareness

```python
from libs.chunking import RecursiveStructureChunker, RecursiveConfig

chunker = RecursiveStructureChunker(RecursiveConfig(
    max_chunk_size=512,
    min_chunk_size=64,
    prefer_structural=True,
))
chunks = chunker.chunk(canonical_doc)
```

### Parent-child for retrieval + context expansion

```python
from libs.chunking import ParentChildChunker, ParentChildConfig

chunker = ParentChildChunker(ParentChildConfig(
    parent_chunk_size=1024,
    child_chunk_size=256,
    child_overlap=64,
))
chunks = chunker.chunk(canonical_doc)

parents = [c for c in chunks if c.metadata.get("is_parent")]
children = [c for c in chunks if c.metadata.get("parent_chunk_id")]
```

### Custom token counter

```python
from libs.chunking import FixedWindowChunker, FixedWindowConfig

chunker = FixedWindowChunker(
    config=FixedWindowConfig(chunk_size=256),
    token_counter=my_custom_counter,  # implements TokenCounter protocol
)
```

### Running the demo

```bash
uv run python examples/chunking_sample.py
```

This parses `tests/fixtures/sample.md` with the Markdown parser and runs all three strategies with small configs to produce visible output.
