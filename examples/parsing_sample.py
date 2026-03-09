#!/usr/bin/env python3
"""Demonstrate the parsing pipeline: RawDocument → Parser → CanonicalDocument."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from libs.contracts.documents import RawDocument, SourceDocumentRef
from libs.parsing import (
    MarkdownParser,
    PlainTextParser,
    normalize_whitespace,
    reindex_positions,
    strip_header_footer,
)

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _make_ref(source_id: str, uri: str) -> SourceDocumentRef:
    return SourceDocumentRef(
        source_id=source_id,
        uri=uri,
        content_hash="sha256:example",
        fetched_at=datetime.now(timezone.utc),
        version=1,
    )


def _print_doc(doc: object) -> None:
    """Print CanonicalDocument blocks as inspectable JSON-like output."""
    from libs.contracts.documents import CanonicalDocument

    assert isinstance(doc, CanonicalDocument)
    print(f"Document ID: {doc.document_id}")
    print(f"Parser:      {doc.parser_version}")
    print(f"Blocks:      {len(doc.blocks)}")
    print("-" * 60)
    for block in doc.blocks:
        preview = block.content[:80].replace("\n", "\\n")
        meta = json.dumps(block.metadata) if block.metadata else ""
        level = f" level={block.level}" if block.level else ""
        print(f"  [{block.position}] {block.block_type.value}{level}: {preview!r} {meta}")
    print()


def main() -> None:
    # ── Plain text ──
    print("=" * 60)
    print("PLAIN TEXT PARSING")
    print("=" * 60)
    txt_bytes = (FIXTURES / "sample.txt").read_bytes()
    raw_txt = RawDocument(
        source_ref=_make_ref("txt-001", "file:///sample.txt"),
        raw_bytes=txt_bytes,
        mime_type="text/plain",
        size_bytes=len(txt_bytes),
    )
    txt_doc = PlainTextParser().parse(raw_txt)
    _print_doc(txt_doc)

    # ── Markdown ──
    print("=" * 60)
    print("MARKDOWN PARSING")
    print("=" * 60)
    md_bytes = (FIXTURES / "sample.md").read_bytes()
    raw_md = RawDocument(
        source_ref=_make_ref("md-001", "file:///sample.md"),
        raw_bytes=md_bytes,
        mime_type="text/markdown",
        size_bytes=len(md_bytes),
    )
    md_doc = MarkdownParser().parse(raw_md)
    _print_doc(md_doc)

    # ── Normalizer demos ──
    print("=" * 60)
    print("NORMALIZER DEMOS")
    print("=" * 60)

    messy = "  lots   of   spaces   and\ttabs  "
    print(f"normalize_whitespace({messy!r})")
    print(f"  → {normalize_whitespace(messy)!r}")
    print()

    # Strip header/footer then reindex
    blocks = md_doc.blocks
    if len(blocks) > 2:
        stripped = strip_header_footer(blocks, header_marker="Main Title")
        reindexed = reindex_positions(stripped)
        print(f"After stripping header ('Main Title'): {len(stripped)} blocks")
        print(f"After reindexing: positions = {[b.position for b in reindexed]}")
    print()


if __name__ == "__main__":
    main()
