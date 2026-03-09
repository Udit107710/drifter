"""Tests for the chunking subsystem: strategies, helpers, and protocols."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from libs.chunking.chunk_id import content_hash, generate_chunk_id
from libs.chunking.config import FixedWindowConfig, ParentChildConfig, RecursiveConfig
from libs.chunking.protocols import ChunkingStrategy, TokenCounter
from libs.chunking.section_tracker import SectionTracker
from libs.chunking.strategies.fixed_window import FixedWindowChunker
from libs.chunking.strategies.parent_child import ParentChildChunker
from libs.chunking.strategies.recursive import RecursiveStructureChunker
from libs.chunking.token_counter import WhitespaceTokenCounter
from libs.contracts.common import BlockType
from libs.contracts.documents import Block, CanonicalDocument, RawDocument, SourceDocumentRef
from libs.parsing.parsers.markdown import MarkdownParser

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


# ── Factory helpers ─────────────────────────────────────────────────


def _make_source_ref(**overrides: object) -> SourceDocumentRef:
    defaults: dict[str, object] = {
        "source_id": "src-001",
        "uri": "https://example.com/doc.txt",
        "content_hash": "sha256:abc123",
        "fetched_at": datetime(2025, 1, 1, tzinfo=UTC),
        "version": 1,
    }
    defaults.update(overrides)
    return SourceDocumentRef(**defaults)  # type: ignore[arg-type]


def _make_doc(
    blocks: list[Block],
    acl: list[str] | None = None,
) -> CanonicalDocument:
    return CanonicalDocument(
        document_id="doc:src-001:1",
        source_ref=_make_source_ref(),
        blocks=blocks,
        parser_version="test:1.0.0",
        parsed_at=datetime(2025, 1, 1, tzinfo=UTC),
        acl=acl or [],
    )


def _make_block(
    block_id: str,
    content: str,
    block_type: BlockType = BlockType.PARAGRAPH,
    position: int = 0,
    level: int | None = None,
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        content=content,
        position=position,
        level=level,
    )


def _words(n: int) -> str:
    """Generate a string with exactly *n* whitespace-delimited tokens."""
    return " ".join(f"w{i}" for i in range(n))


# ── Token counter ──────────────────────────────────────────────────


class TestWhitespaceTokenCounter:
    def test_count_simple(self) -> None:
        tc = WhitespaceTokenCounter()
        assert tc.count("hello world") == 2

    def test_count_empty(self) -> None:
        tc = WhitespaceTokenCounter()
        assert tc.count("") == 0

    def test_count_whitespace_only(self) -> None:
        tc = WhitespaceTokenCounter()
        assert tc.count("   ") == 0

    def test_count_multiline(self) -> None:
        tc = WhitespaceTokenCounter()
        assert tc.count("multi\nline\ntext") == 3


# ── Chunk IDs ──────────────────────────────────────────────────────


class TestChunkId:
    def test_format(self) -> None:
        cid = generate_chunk_id("doc:1", "fixed_window", "hello world", 0)
        assert cid.startswith("chk:")
        hex_part = cid[len("chk:"):]
        assert len(hex_part) == 24
        # Must be valid hex characters.
        int(hex_part, 16)

    def test_deterministic(self) -> None:
        a = generate_chunk_id("doc:1", "fixed_window", "hello", 0)
        b = generate_chunk_id("doc:1", "fixed_window", "hello", 0)
        assert a == b

    def test_different_input_different_id(self) -> None:
        a = generate_chunk_id("doc:1", "fixed_window", "hello", 0)
        b = generate_chunk_id("doc:1", "fixed_window", "world", 0)
        assert a != b

    def test_content_hash_format(self) -> None:
        h = content_hash("some text")
        assert h.startswith("sha256:")
        hex_part = h[len("sha256:"):]
        assert len(hex_part) == 64  # full SHA-256


# ── Section tracker ────────────────────────────────────────────────


class TestSectionTracker:
    def test_nested_headings(self) -> None:
        tracker = SectionTracker()
        tracker.update(_make_block("b0", "A", BlockType.HEADING, 0, level=1))
        tracker.update(_make_block("b1", "B", BlockType.HEADING, 1, level=2))
        assert tracker.current_path() == ["A", "B"]

    def test_same_level_replacement(self) -> None:
        tracker = SectionTracker()
        tracker.update(_make_block("b0", "A", BlockType.HEADING, 0, level=1))
        tracker.update(_make_block("b1", "B", BlockType.HEADING, 1, level=2))
        tracker.update(_make_block("b2", "C", BlockType.HEADING, 2, level=2))
        assert tracker.current_path() == ["A", "C"]

    def test_shallower_clears_deeper(self) -> None:
        tracker = SectionTracker()
        tracker.update(_make_block("b0", "A", BlockType.HEADING, 0, level=1))
        tracker.update(_make_block("b1", "B", BlockType.HEADING, 1, level=2))
        tracker.update(_make_block("b2", "D", BlockType.HEADING, 2, level=1))
        assert tracker.current_path() == ["D"]

    def test_non_heading_no_change(self) -> None:
        tracker = SectionTracker()
        tracker.update(_make_block("b0", "A", BlockType.HEADING, 0, level=1))
        tracker.update(_make_block("b1", "some text", BlockType.PARAGRAPH, 1))
        assert tracker.current_path() == ["A"]


# ── Protocols ──────────────────────────────────────────────────────


class TestProtocols:
    def test_fixed_window_is_chunking_strategy(self) -> None:
        assert isinstance(FixedWindowChunker(), ChunkingStrategy)

    def test_recursive_is_chunking_strategy(self) -> None:
        assert isinstance(RecursiveStructureChunker(), ChunkingStrategy)

    def test_parent_child_is_chunking_strategy(self) -> None:
        assert isinstance(ParentChildChunker(), ChunkingStrategy)

    def test_whitespace_counter_is_token_counter(self) -> None:
        assert isinstance(WhitespaceTokenCounter(), TokenCounter)


# ── Config validation ──────────────────────────────────────────────


class TestConfigValidation:
    def test_fixed_window_defaults(self) -> None:
        cfg = FixedWindowConfig()
        assert cfg.chunk_size == 256

    def test_recursive_defaults(self) -> None:
        cfg = RecursiveConfig()
        assert cfg.max_chunk_size == 512

    def test_parent_child_defaults(self) -> None:
        cfg = ParentChildConfig()
        assert cfg.parent_chunk_size == 1024

    def test_fixed_window_zero_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size must be > 0"):
            FixedWindowConfig(chunk_size=0)

    def test_fixed_window_negative_overlap(self) -> None:
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            FixedWindowConfig(overlap=-1)

    def test_fixed_window_overlap_ge_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="overlap must be < chunk_size"):
            FixedWindowConfig(chunk_size=10, overlap=10)

    def test_fixed_window_min_chunk_size_zero(self) -> None:
        with pytest.raises(ValueError, match="min_chunk_size must be > 0"):
            FixedWindowConfig(min_chunk_size=0)

    def test_fixed_window_min_chunk_size_exceeds_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="min_chunk_size must be <= chunk_size"):
            FixedWindowConfig(chunk_size=10, overlap=2, min_chunk_size=20)

    def test_recursive_zero_max(self) -> None:
        with pytest.raises(ValueError, match="max_chunk_size must be > 0"):
            RecursiveConfig(max_chunk_size=0)

    def test_recursive_min_exceeds_max(self) -> None:
        with pytest.raises(ValueError, match="min_chunk_size must be <= max_chunk_size"):
            RecursiveConfig(max_chunk_size=10, min_chunk_size=20)

    def test_parent_child_parent_le_child(self) -> None:
        with pytest.raises(ValueError, match="parent_chunk_size must be > child_chunk_size"):
            ParentChildConfig(parent_chunk_size=100, child_chunk_size=100)

    def test_parent_child_overlap_ge_child(self) -> None:
        with pytest.raises(ValueError, match="child_overlap must be < child_chunk_size"):
            ParentChildConfig(child_chunk_size=50, child_overlap=50)


# ── FixedWindowChunker ─────────────────────────────────────────────


class TestFixedWindowChunker:
    def test_single_block_under_limit(self) -> None:
        block = _make_block("b0", _words(10), position=0)
        doc = _make_doc([block])
        chunker = FixedWindowChunker(FixedWindowConfig(chunk_size=20, overlap=0, min_chunk_size=1))
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == _words(10)

    def test_single_block_over_limit(self) -> None:
        block = _make_block("b0", _words(100), position=0)
        doc = _make_doc([block])
        chunker = FixedWindowChunker(
            FixedWindowConfig(chunk_size=10, overlap=2, min_chunk_size=1)
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1

    def test_overlap_correctness(self) -> None:
        # chunk_size=6, overlap=2 → step=4
        block = _make_block("b0", _words(14), position=0)
        doc = _make_doc([block])
        chunker = FixedWindowChunker(
            FixedWindowConfig(chunk_size=6, overlap=2, min_chunk_size=1)
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        # Check consecutive chunks share last 2 tokens of previous with first 2 of next.
        for i in range(len(chunks) - 1):
            prev_tokens = chunks[i].content.split()
            next_tokens = chunks[i + 1].content.split()
            assert prev_tokens[-2:] == next_tokens[:2]

    def test_min_chunk_discard(self) -> None:
        # 11 words, chunk_size=5, overlap=0, step=5, min_chunk_size=3
        # Produces windows: [0:5](5 tok), [5:10](5 tok), [10:11](1 tok) → 1 < 3, discard trailing
        block = _make_block("b0", _words(11), position=0)
        doc = _make_doc([block])
        chunker = FixedWindowChunker(
            FixedWindowConfig(chunk_size=5, overlap=0, min_chunk_size=3)
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2
        # But first chunk is always kept even if small.

    def test_first_chunk_always_kept(self) -> None:
        # Single word → 1 token, min_chunk_size=5 → first chunk still kept.
        block = _make_block("b0", "hello", position=0)
        doc = _make_doc([block])
        chunker = FixedWindowChunker(
            FixedWindowConfig(chunk_size=10, overlap=0, min_chunk_size=5)
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_multi_block_merge(self) -> None:
        blocks = [
            _make_block("b0", _words(3), position=0),
            _make_block("b1", _words(3), position=1),
            _make_block("b2", _words(3), position=2),
        ]
        doc = _make_doc(blocks)
        chunker = FixedWindowChunker(
            FixedWindowConfig(chunk_size=20, overlap=0, min_chunk_size=1)
        )
        chunks = chunker.chunk(doc)
        # All 9 tokens fit in one chunk.
        assert len(chunks) == 1

    def test_block_ids_in_lineage(self) -> None:
        blocks = [
            _make_block("b0", _words(5), position=0),
            _make_block("b1", _words(5), position=1),
        ]
        doc = _make_doc(blocks)
        chunker = FixedWindowChunker(
            FixedWindowConfig(chunk_size=20, overlap=0, min_chunk_size=1)
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert "b0" in chunks[0].block_ids
        assert "b1" in chunks[0].block_ids

    def test_lineage_fields(self) -> None:
        block = _make_block("b0", _words(5), position=0)
        doc = _make_doc([block])
        chunker = FixedWindowChunker(
            FixedWindowConfig(chunk_size=20, overlap=0, min_chunk_size=1)
        )
        chunks = chunker.chunk(doc)
        c = chunks[0]
        assert c.lineage.source_id == "src-001"
        assert c.lineage.document_id == "doc:src-001:1"
        assert c.lineage.chunk_strategy == "fixed_window"
        assert c.lineage.parser_version == "test:1.0.0"

    def test_byte_offsets(self) -> None:
        block = _make_block("b0", _words(10), position=0)
        doc = _make_doc([block])
        chunker = FixedWindowChunker(
            FixedWindowConfig(chunk_size=20, overlap=0, min_chunk_size=1)
        )
        chunks = chunker.chunk(doc)
        c = chunks[0]
        assert c.byte_offset_start >= 0
        assert c.byte_offset_end > c.byte_offset_start

    def test_acl_propagation(self) -> None:
        block = _make_block("b0", _words(5), position=0)
        doc = _make_doc([block], acl=["admin"])
        chunker = FixedWindowChunker(
            FixedWindowConfig(chunk_size=20, overlap=0, min_chunk_size=1)
        )
        chunks = chunker.chunk(doc)
        assert chunks[0].acl == ["admin"]

    def test_deterministic_ids(self) -> None:
        block = _make_block("b0", _words(10), position=0)
        doc = _make_doc([block])
        chunker = FixedWindowChunker(
            FixedWindowConfig(chunk_size=20, overlap=0, min_chunk_size=1)
        )
        chunks_a = chunker.chunk(doc)
        chunks_b = chunker.chunk(doc)
        assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]

    def test_section_path_in_metadata(self) -> None:
        blocks = [
            _make_block("b0", "Introduction", BlockType.HEADING, 0, level=1),
            _make_block("b1", _words(5), BlockType.PARAGRAPH, 1),
        ]
        doc = _make_doc(blocks)
        chunker = FixedWindowChunker(
            FixedWindowConfig(chunk_size=20, overlap=0, min_chunk_size=1)
        )
        chunks = chunker.chunk(doc)
        # At least one chunk should have a section_path.
        paths = [c.metadata.get("section_path", []) for c in chunks]
        assert any("Introduction" in p for p in paths)

    def test_strategy_name(self) -> None:
        assert FixedWindowChunker().strategy_name() == "fixed_window"


# ── RecursiveStructureChunker ──────────────────────────────────────


class TestRecursiveStructureChunker:
    def test_heading_flush(self) -> None:
        blocks = [
            _make_block("b0", _words(5), BlockType.PARAGRAPH, 0),
            _make_block("b1", "Title", BlockType.HEADING, 1, level=1),
            _make_block("b2", _words(5), BlockType.PARAGRAPH, 2),
        ]
        doc = _make_doc(blocks)
        chunker = RecursiveStructureChunker(
            RecursiveConfig(max_chunk_size=100, min_chunk_size=1)
        )
        chunks = chunker.chunk(doc)
        # Heading should start a new chunk — at least 2 chunks.
        assert len(chunks) >= 2

    def test_small_blocks_merge(self) -> None:
        blocks = [
            _make_block(f"b{i}", _words(3), BlockType.PARAGRAPH, i)
            for i in range(5)
        ]
        doc = _make_doc(blocks)
        chunker = RecursiveStructureChunker(
            RecursiveConfig(max_chunk_size=100, min_chunk_size=1, prefer_structural=False)
        )
        chunks = chunker.chunk(doc)
        # 15 tokens total, all same type, no structural split → 1 chunk.
        assert len(chunks) == 1

    def test_oversized_split(self) -> None:
        block = _make_block("b0", _words(100), BlockType.PARAGRAPH, 0)
        doc = _make_doc([block])
        chunker = RecursiveStructureChunker(
            RecursiveConfig(max_chunk_size=10, min_chunk_size=1)
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        # Each chunk's token count should be at most max_chunk_size.
        tc = WhitespaceTokenCounter()
        for c in chunks:
            assert tc.count(c.content) <= 10

    def test_section_path(self) -> None:
        blocks = [
            _make_block("b0", "Title", BlockType.HEADING, 0, level=1),
            _make_block("b1", _words(5), BlockType.PARAGRAPH, 1),
        ]
        doc = _make_doc(blocks)
        chunker = RecursiveStructureChunker(
            RecursiveConfig(max_chunk_size=100, min_chunk_size=1)
        )
        chunks = chunker.chunk(doc)
        # Find chunk containing the paragraph content.
        para_chunks = [c for c in chunks if _words(5) in c.content]
        assert para_chunks
        assert para_chunks[0].metadata.get("section_path") == ["Title"]

    def test_type_transition_with_prefer_structural(self) -> None:
        blocks = [
            _make_block("b0", _words(10), BlockType.PARAGRAPH, 0),
            _make_block("b1", _words(10), BlockType.CODE, 1),
        ]
        doc = _make_doc(blocks)
        chunker = RecursiveStructureChunker(
            RecursiveConfig(max_chunk_size=100, min_chunk_size=5, prefer_structural=True)
        )
        chunks = chunker.chunk(doc)
        # Type transition with enough tokens → separate chunks.
        assert len(chunks) == 2

    def test_type_transition_without_prefer_structural(self) -> None:
        blocks = [
            _make_block("b0", _words(10), BlockType.PARAGRAPH, 0),
            _make_block("b1", _words(10), BlockType.CODE, 1),
        ]
        doc = _make_doc(blocks)
        chunker = RecursiveStructureChunker(
            RecursiveConfig(max_chunk_size=100, min_chunk_size=5, prefer_structural=False)
        )
        chunks = chunker.chunk(doc)
        # No structural split → 1 chunk.
        assert len(chunks) == 1

    def test_runt_merge(self) -> None:
        # First block is large, second block is a tiny runt.
        blocks = [
            _make_block("b0", _words(20), BlockType.PARAGRAPH, 0),
            _make_block("b1", "tiny", BlockType.PARAGRAPH, 1),
        ]
        doc = _make_doc(blocks)
        chunker = RecursiveStructureChunker(
            RecursiveConfig(max_chunk_size=25, min_chunk_size=5, prefer_structural=False)
        )
        chunks = chunker.chunk(doc)
        # "tiny" (1 token) < min_chunk_size (5), so it should be merged.
        # All content should be present somewhere in chunks.
        all_content = "\n".join(c.content for c in chunks)
        assert "tiny" in all_content

    def test_strategy_name(self) -> None:
        assert RecursiveStructureChunker().strategy_name() == "recursive"


# ── ParentChildChunker ─────────────────────────────────────────────


class TestParentChildChunker:
    def _make_large_doc(self, n_words: int = 50) -> CanonicalDocument:
        """Create a doc with enough content for parent-child splitting."""
        blocks = [
            _make_block("b0", "Introduction", BlockType.HEADING, 0, level=1),
            _make_block("b1", _words(n_words), BlockType.PARAGRAPH, 1),
        ]
        return _make_doc(blocks)

    def test_both_types_produced(self) -> None:
        doc = self._make_large_doc(50)
        chunker = ParentChildChunker(
            ParentChildConfig(
                parent_chunk_size=100, child_chunk_size=10,
                child_overlap=2, min_child_size=1,
            )
        )
        chunks = chunker.chunk(doc)
        parents = [c for c in chunks if c.metadata.get("is_parent")]
        children = [c for c in chunks if "parent_chunk_id" in c.metadata]
        assert len(parents) >= 1
        assert len(children) >= 1

    def test_parent_metadata(self) -> None:
        doc = self._make_large_doc(50)
        chunker = ParentChildChunker(
            ParentChildConfig(
                parent_chunk_size=100, child_chunk_size=10,
                child_overlap=2, min_child_size=1,
            )
        )
        chunks = chunker.chunk(doc)
        parents = [c for c in chunks if c.metadata.get("is_parent")]
        assert parents
        p = parents[0]
        assert p.metadata["is_parent"] is True
        assert isinstance(p.metadata["child_chunk_ids"], list)
        assert len(p.metadata["child_chunk_ids"]) >= 1

    def test_child_metadata(self) -> None:
        doc = self._make_large_doc(50)
        chunker = ParentChildChunker(
            ParentChildConfig(
                parent_chunk_size=100, child_chunk_size=10,
                child_overlap=2, min_child_size=1,
            )
        )
        chunks = chunker.chunk(doc)
        children = [c for c in chunks if "parent_chunk_id" in c.metadata]
        assert children
        child = children[0]
        assert "parent_chunk_id" in child.metadata
        assert "child_index" in child.metadata

    def test_child_index_order(self) -> None:
        doc = self._make_large_doc(50)
        chunker = ParentChildChunker(
            ParentChildConfig(
                parent_chunk_size=100, child_chunk_size=10,
                child_overlap=2, min_child_size=1,
            )
        )
        chunks = chunker.chunk(doc)
        parents = [c for c in chunks if c.metadata.get("is_parent")]
        children = [c for c in chunks if "parent_chunk_id" in c.metadata]
        for parent in parents:
            parent_children = [
                c for c in children
                if c.metadata.get("parent_chunk_id") == parent.chunk_id
            ]
            indices = [c.metadata["child_index"] for c in parent_children]
            assert indices == list(range(len(parent_children)))

    def test_children_cover_parent(self) -> None:
        doc = self._make_large_doc(50)
        chunker = ParentChildChunker(
            ParentChildConfig(
                parent_chunk_size=100, child_chunk_size=10,
                child_overlap=2, min_child_size=1,
            )
        )
        chunks = chunker.chunk(doc)
        parents = [c for c in chunks if c.metadata.get("is_parent")]
        children = [c for c in chunks if "parent_chunk_id" in c.metadata]
        for parent in parents:
            parent_children = sorted(
                [c for c in children if c.metadata.get("parent_chunk_id") == parent.chunk_id],
                key=lambda c: c.metadata["child_index"],
            )
            # All parent tokens should appear in at least one child.
            parent_tokens = set(parent.content.split())
            child_tokens: set[str] = set()
            for c in parent_children:
                child_tokens.update(c.content.split())
            assert parent_tokens.issubset(child_tokens)

    def test_block_ids_inherited(self) -> None:
        doc = self._make_large_doc(50)
        chunker = ParentChildChunker(
            ParentChildConfig(
                parent_chunk_size=100, child_chunk_size=10,
                child_overlap=2, min_child_size=1,
            )
        )
        chunks = chunker.chunk(doc)
        parents = [c for c in chunks if c.metadata.get("is_parent")]
        children = [c for c in chunks if "parent_chunk_id" in c.metadata]
        for parent in parents:
            parent_children = [
                c for c in children
                if c.metadata.get("parent_chunk_id") == parent.chunk_id
            ]
            for child in parent_children:
                assert child.block_ids == parent.block_ids

    def test_strategy_name(self) -> None:
        assert ParentChildChunker().strategy_name() == "parent_child"


# ── Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_whitespace_only_blocks_fixed_window(self) -> None:
        """Whitespace-only blocks have 0 tokens; chunker should handle gracefully."""
        blocks = [
            _make_block("b0", _words(5), BlockType.PARAGRAPH, 0),
            _make_block("b1", "   ", BlockType.PARAGRAPH, 1),
            _make_block("b2", _words(5), BlockType.PARAGRAPH, 2),
        ]
        doc = _make_doc(blocks)
        chunker = FixedWindowChunker(
            FixedWindowConfig(chunk_size=20, overlap=0, min_chunk_size=1)
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1
        for c in chunks:
            assert c.token_count >= 1

    def test_single_token_doc(self) -> None:
        block = _make_block("b0", "hello", BlockType.PARAGRAPH, 0)
        doc = _make_doc([block])
        chunker = FixedWindowChunker(
            FixedWindowConfig(chunk_size=10, overlap=0, min_chunk_size=1)
        )
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "hello"
        assert chunks[0].token_count == 1


# ── Integration: parse fixture then chunk ──────────────────────────


class TestIntegration:
    def test_fixture_chunking_all_strategies(self) -> None:
        """Parse sample.md, then chunk with all three strategies and validate."""
        content = (FIXTURES / "sample.md").read_bytes()
        parser = MarkdownParser()
        raw = RawDocument(
            source_ref=_make_source_ref(),
            raw_bytes=content,
            mime_type="text/markdown",
            size_bytes=len(content),
        )
        doc = parser.parse(raw)

        Chunker = FixedWindowChunker | RecursiveStructureChunker | ParentChildChunker
        chunkers: list[tuple[str, Chunker]] = [
            ("fixed_window", FixedWindowChunker(
                FixedWindowConfig(chunk_size=50, overlap=5, min_chunk_size=1),
            )),
            ("recursive", RecursiveStructureChunker(
                RecursiveConfig(max_chunk_size=50, min_chunk_size=1),
            )),
            ("parent_child", ParentChildChunker(ParentChildConfig(
                parent_chunk_size=100, child_chunk_size=20,
                child_overlap=4, min_child_size=1,
            ))),
        ]

        for name, chunker in chunkers:
            chunks = chunker.chunk(doc)
            assert len(chunks) >= 1, f"{name} produced no chunks"
            for c in chunks:
                assert c.chunk_id, f"{name}: empty chunk_id"
                assert c.content, f"{name}: empty content"
                assert c.token_count >= 1, f"{name}: token_count < 1"
                assert c.byte_offset_start >= 0, f"{name}: negative byte_offset_start"
                assert c.byte_offset_end > c.byte_offset_start, f"{name}: bad byte offsets"
                assert c.lineage.source_id == "src-001", f"{name}: wrong source_id"
                assert c.lineage.document_id == doc.document_id, f"{name}: wrong document_id"
                assert c.strategy == name, f"{name}: wrong strategy"
                assert c.content_hash.startswith("sha256:"), f"{name}: bad content_hash"
