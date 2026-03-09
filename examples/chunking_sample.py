"""Demo: chunking strategies applied to a sample Markdown document."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from libs.chunking import (
    FixedWindowChunker,
    FixedWindowConfig,
    ParentChildChunker,
    ParentChildConfig,
    RecursiveConfig,
    RecursiveStructureChunker,
)
from libs.contracts.documents import RawDocument, SourceDocumentRef
from libs.parsing.parsers.markdown import MarkdownParser


def main() -> None:
    fixture = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample.md"
    raw_bytes = fixture.read_bytes()

    source_ref = SourceDocumentRef(
        source_id="sample-src",
        uri=str(fixture),
        content_hash="sha256:demo",
        fetched_at=datetime.now(UTC),
        version=1,
    )
    raw = RawDocument(
        source_ref=source_ref,
        raw_bytes=raw_bytes,
        mime_type="text/markdown",
        size_bytes=len(raw_bytes),
    )

    parser = MarkdownParser()
    doc = parser.parse(raw)

    # Use small configs so the sample fixture produces multiple chunks
    fw_cfg = FixedWindowConfig(chunk_size=8, overlap=2, min_chunk_size=2)
    rc_cfg = RecursiveConfig(max_chunk_size=10, min_chunk_size=2)
    pc_cfg = ParentChildConfig(
        parent_chunk_size=20, child_chunk_size=6,
        child_overlap=2, min_child_size=2,
    )
    strategies = [
        ("Fixed Window", FixedWindowChunker(fw_cfg)),
        ("Recursive", RecursiveStructureChunker(rc_cfg)),
        ("Parent-Child", ParentChildChunker(pc_cfg)),
    ]

    for name, chunker in strategies:
        chunks = chunker.chunk(doc)
        print(f"\n{'='*60}")
        print(f"Strategy: {name} ({chunker.strategy_name()})")
        print(f"Total chunks: {len(chunks)}")
        print(f"{'='*60}")

        for chunk in chunks:
            preview = chunk.content[:80].replace("\n", "\\n")
            section = chunk.metadata.get("section_path", [])
            is_parent = chunk.metadata.get("is_parent", False)
            parent_id = chunk.metadata.get("parent_chunk_id", "")
            child_idx = chunk.metadata.get("child_index", "")

            print(f"\n  ID:        {chunk.chunk_id}")
            print(f"  Tokens:    {chunk.token_count}")
            print(f"  Bytes:     {chunk.byte_offset_start}-{chunk.byte_offset_end}")
            print(f"  Blocks:    {chunk.block_ids}")
            if section:
                print(f"  Section:   {' > '.join(section)}")
            if is_parent:
                child_ids = chunk.metadata.get("child_chunk_ids", [])
                print(f"  Parent:    yes ({len(child_ids)} children)")
            if parent_id:
                print(f"  Child of:  {parent_id} (index {child_idx})")
            print(f"  Content:   {preview}...")


if __name__ == "__main__":
    main()
