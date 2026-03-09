# libs/chunking/

Splits CanonicalDocuments into Chunks using configurable strategies, preserving full lineage.

## Boundary

- **Consumes:** CanonicalDocument (ordered list of Blocks)
- **Produces:** list[Chunk] with lineage, token counts, and deterministic IDs

## Protocols

```python
class ChunkingStrategy(Protocol):
    def chunk(self, doc: CanonicalDocument) -> list[Chunk]: ...
    def strategy_name(self) -> str: ...

class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...
```

## Strategies

| Strategy | Config | Description |
|----------|--------|-------------|
| `FixedWindowChunker` | `FixedWindowConfig` | Fixed token window with configurable overlap |
| `RecursiveStructureChunker` | `RecursiveConfig` | Splits recursively by document structure (headings, paragraphs) |
| `ParentChildChunker` | `ParentChildConfig` | Creates parent (context) and child (detail) chunks |

All strategies live in `strategies/` and implement the `ChunkingStrategy` protocol.

## Chunk Lineage

Every Chunk carries a `ChunkLineage` recording:
- `source_id` — Original source
- `document_id` — Parsed document
- `block_ids` — Contributing blocks
- `chunk_strategy` — Strategy name
- `parser_version` — Parser that produced the blocks
- `created_at` — Timestamp

## Supporting Modules

| Module | Purpose |
|--------|---------|
| `builder.py` | `build_chunk()` helper, byte offset computation |
| `chunk_id.py` | Deterministic chunk ID generation (content-addressed) |
| `section_tracker.py` | Tracks heading hierarchy during chunking |
| `token_counter.py` | `WhitespaceTokenCounter` (default, zero-dependency) |
| `config.py` | Strategy-specific configuration dataclasses |

## Token Counting

`WhitespaceTokenCounter` splits on whitespace for approximate token counts. For production, swap in a model-specific tokenizer behind the `TokenCounter` protocol.

## Testing

All strategies tested with fixture documents. Deterministic: same input always produces same chunks with same IDs.
