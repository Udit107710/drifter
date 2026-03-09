# Parsing Subsystem

## Pipeline

```
RawDocument → Parser → Normalizers → CanonicalDocument
```

The parsing subsystem converts raw bytes (`RawDocument`) into a structured `CanonicalDocument` composed of typed `Block` elements. Document structure (headings, paragraphs, code, tables, lists, quotes) is always preserved — never flattened to plain text.

## Block Type Mapping

| Block Type | Plain Text | Markdown |
|------------|-----------|----------|
| HEADING    | —         | `# ...`  |
| PARAGRAPH  | Blank-line-separated sections | Non-special text |
| CODE       | —         | Fenced `` ``` `` blocks |
| TABLE      | —         | `\|`-delimited lines |
| LIST       | —         | `- `, `* `, `1. ` items |
| QUOTE      | —         | `> ` lines |
| METADATA   | —         | (reserved for front-matter) |

## Parsers

### PlainTextParser

- **MIME types**: `text/plain`
- Decodes bytes (UTF-8, fallback latin-1)
- Splits on blank lines → `PARAGRAPH` blocks
- Normalises whitespace within each paragraph
- **Version**: `plain_text:1.0.0`

### MarkdownParser

- **MIME types**: `text/markdown`, `text/x-markdown`
- Line-by-line state machine — no external dependencies
- Handles headings, fenced code (with language metadata), blockquotes, lists, tables, paragraphs
- Consecutive same-type lines (quotes, lists, tables) are merged into single blocks
- **Version**: `markdown:1.0.0`

### PdfParserBase (abstract)

- **MIME types**: `application/pdf`
- Abstract base class — subclass and implement `_extract_blocks(raw_bytes) -> list[Block]`
- Provides normalisation and `CanonicalDocument` construction

## Normalisation Hooks

Pure functions in `libs/parsing/normalizers.py`:

| Function | Purpose |
|----------|---------|
| `normalize_whitespace(text)` | Collapse spaces/tabs to single space, strip edges |
| `collapse_blank_lines(text)` | Reduce 3+ consecutive newlines to 2 |
| `strip_header_footer(blocks, header_marker, footer_marker)` | Remove blocks before header / after footer marker |
| `reindex_positions(blocks)` | Renumber block positions 0..N-1 after filtering |

These are composable — call individually or chain:

```python
from libs.parsing.normalizers import strip_header_footer, reindex_positions

blocks = parser.parse(raw).blocks
blocks = strip_header_footer(blocks, header_marker="PAGE HEADER")
blocks = reindex_positions(blocks)
```

## Adding a New Parser

1. Create a module in `libs/parsing/parsers/`
2. Implement the `DocumentParser` protocol:
   ```python
   class MyParser:
       def supported_mime_types(self) -> list[str]:
           return ["application/x-myformat"]

       def parse(self, raw: RawDocument) -> CanonicalDocument:
           # Extract blocks from raw.raw_bytes
           # Build CanonicalDocument with source lineage
           ...
   ```
3. Re-export from `libs/parsing/parsers/__init__.py` and `libs/parsing/__init__.py`
4. Add tests in `tests/unit/test_parsing.py`

## Structural Preservation Guarantees

- Every block carries `block_type`, `position`, `block_id`, and optional `level`/`metadata`
- Block IDs encode source lineage: `{source_id}:blk:{position}`
- Document IDs encode version: `doc:{source_id}:{version}`
- Code blocks preserve language metadata
- Headings preserve level (1–6)
- No content is silently dropped — empty blocks are filtered, not created

## Usage

```python
from libs.parsing import PlainTextParser, MarkdownParser

parser = PlainTextParser()
doc = parser.parse(raw_document)

for block in doc.blocks:
    print(f"[{block.position}] {block.block_type.value}: {block.content[:60]}")
```

See `examples/parsing_sample.py` for a full runnable demonstration.
