# libs/parsing/

Converts raw document bytes into structured CanonicalDocuments with preserved document structure.

## Boundary

- **Consumes:** RawDocument (raw bytes + MIME type)
- **Produces:** CanonicalDocument (ordered list of Blocks)
- **Rule:** Never flatten documents to plain text. Preserve headings, paragraphs, tables, code blocks, etc.

## Key Types

| Type | Purpose |
|------|---------|
| `Block` | Structural element: heading, paragraph, table, code, list, etc. |
| `CanonicalDocument` | Document ID + source ref + ordered blocks + parser version |

## Protocol

```python
class DocumentParser(Protocol):
    def parse(self, raw: RawDocument) -> CanonicalDocument: ...
    def supported_mime_types(self) -> list[str]: ...
```

## Parsers

| Parser | MIME Types | Location |
|--------|-----------|----------|
| `PlainTextParser` | `text/plain` | `parsers/plain_text.py` |
| `MarkdownParser` | `text/markdown` | `parsers/markdown.py` |
| `PdfParserBase` | `application/pdf` | `parsers/pdf.py` (abstract, implemented by adapters) |

PDF parsing is handled by adapter implementations:
- `UnstructuredPdfParser` (`libs/adapters/unstructured/`)
- `TikaPdfParser` (`libs/adapters/tika/`)

## Normalizers

`normalizers.py` provides text normalization utilities:

| Function | Purpose |
|----------|---------|
| `normalize_whitespace` | Collapse multiple whitespace to single spaces |
| `collapse_blank_lines` | Reduce multiple blank lines to one |
| `strip_header_footer` | Remove repeated headers/footers from pages |
| `reindex_positions` | Renumber block positions after filtering |

## Block Types

```python
class BlockType(Enum):
    HEADING, PARAGRAPH, TABLE, CODE, LIST, IMAGE_CAPTION, QUOTE, METADATA
```

Each Block records its `block_type`, `content`, `position`, and optional `level` (for headings).

## Testing

Uses `PlainTextParser` and `MarkdownParser` with fixture files in `tests/fixtures/`. No external parsing services required.
