"""Tests for document contracts: SourceDocumentRef, RawDocument, CanonicalDocument, Block."""

from datetime import UTC, datetime

import pytest

from libs.contracts.common import BlockType
from libs.contracts.documents import (
    Block,
    CanonicalDocument,
    RawDocument,
    SourceDocumentRef,
)


def _make_source_ref(**overrides: object) -> SourceDocumentRef:
    defaults: dict[str, object] = {
        "source_id": "src-001",
        "uri": "https://example.com/doc.pdf",
        "content_hash": "sha256:abc123",
        "fetched_at": datetime(2025, 1, 1, tzinfo=UTC),
        "version": 1,
    }
    defaults.update(overrides)
    return SourceDocumentRef(**defaults)  # type: ignore[arg-type]


def _make_block(**overrides: object) -> Block:
    defaults: dict[str, object] = {
        "block_id": "blk-001",
        "block_type": BlockType.PARAGRAPH,
        "content": "Some paragraph text.",
        "position": 0,
    }
    defaults.update(overrides)
    return Block(**defaults)  # type: ignore[arg-type]


# ── SourceDocumentRef ────────────────────────────────────────────────


class TestSourceDocumentRef:
    def test_create_valid(self) -> None:
        ref = _make_source_ref()
        assert ref.source_id == "src-001"
        assert ref.version == 1
        assert ref.metadata == {}
        assert ref.authority_score is None
        assert ref.freshness_hint is None

    def test_with_authority_and_freshness(self) -> None:
        now = datetime(2025, 6, 1, tzinfo=UTC)
        ref = _make_source_ref(authority_score=0.9, freshness_hint=now)
        assert ref.authority_score == 0.9
        assert ref.freshness_hint == now

    def test_empty_source_id_raises(self) -> None:
        with pytest.raises(ValueError, match="source_id"):
            _make_source_ref(source_id="")

    def test_empty_uri_raises(self) -> None:
        with pytest.raises(ValueError, match="uri"):
            _make_source_ref(uri="")

    def test_empty_content_hash_raises(self) -> None:
        with pytest.raises(ValueError, match="content_hash"):
            _make_source_ref(content_hash="")

    def test_version_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="version"):
            _make_source_ref(version=0)

    def test_schema_version_default(self) -> None:
        ref = _make_source_ref()
        assert ref.schema_version == 1

    def test_schema_version_custom(self) -> None:
        ref = _make_source_ref(schema_version=2)
        assert ref.schema_version == 2

    def test_is_frozen(self) -> None:
        ref = _make_source_ref()
        with pytest.raises(AttributeError):
            ref.version = 2  # type: ignore[misc]


# ── RawDocument ──────────────────────────────────────────────────────


class TestRawDocument:
    def test_create_valid(self) -> None:
        ref = _make_source_ref()
        raw = RawDocument(
            source_ref=ref,
            raw_bytes=b"PDF content",
            mime_type="application/pdf",
            size_bytes=11,
        )
        assert raw.mime_type == "application/pdf"
        assert raw.size_bytes == 11

    def test_empty_bytes_raises(self) -> None:
        with pytest.raises(ValueError, match="raw_bytes"):
            RawDocument(
                source_ref=_make_source_ref(),
                raw_bytes=b"",
                mime_type="text/plain",
                size_bytes=0,
            )

    def test_empty_mime_type_raises(self) -> None:
        with pytest.raises(ValueError, match="mime_type"):
            RawDocument(
                source_ref=_make_source_ref(),
                raw_bytes=b"data",
                mime_type="",
                size_bytes=4,
            )

    def test_negative_size_raises(self) -> None:
        with pytest.raises(ValueError, match="size_bytes"):
            RawDocument(
                source_ref=_make_source_ref(),
                raw_bytes=b"data",
                mime_type="text/plain",
                size_bytes=-1,
            )


# ── Block ────────────────────────────────────────────────────────────


class TestBlock:
    def test_create_valid(self) -> None:
        block = _make_block()
        assert block.block_type == BlockType.PARAGRAPH
        assert block.level is None

    def test_heading_with_level(self) -> None:
        block = _make_block(block_type=BlockType.HEADING, level=2)
        assert block.level == 2

    def test_empty_block_id_raises(self) -> None:
        with pytest.raises(ValueError, match="block_id"):
            _make_block(block_id="")

    def test_negative_position_raises(self) -> None:
        with pytest.raises(ValueError, match="position"):
            _make_block(position=-1)

    def test_level_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="level"):
            _make_block(level=0)


# ── CanonicalDocument ────────────────────────────────────────────────


class TestCanonicalDocument:
    def test_create_valid(self) -> None:
        doc = CanonicalDocument(
            document_id="doc-001",
            source_ref=_make_source_ref(),
            blocks=[_make_block()],
            parser_version="1.0.0",
            parsed_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert doc.document_id == "doc-001"
        assert len(doc.blocks) == 1
        assert doc.acl == []

    def test_with_acl(self) -> None:
        doc = CanonicalDocument(
            document_id="doc-001",
            source_ref=_make_source_ref(),
            blocks=[_make_block()],
            parser_version="1.0.0",
            parsed_at=datetime(2025, 1, 1, tzinfo=UTC),
            acl=["team:engineering"],
        )
        assert doc.acl == ["team:engineering"]

    def test_schema_version_default(self) -> None:
        doc = CanonicalDocument(
            document_id="doc-001",
            source_ref=_make_source_ref(),
            blocks=[_make_block()],
            parser_version="1.0.0",
            parsed_at=datetime(2025, 1, 1, tzinfo=UTC),
        )
        assert doc.schema_version == 1

    def test_schema_version_custom(self) -> None:
        doc = CanonicalDocument(
            document_id="doc-001",
            source_ref=_make_source_ref(),
            blocks=[_make_block()],
            parser_version="1.0.0",
            parsed_at=datetime(2025, 1, 1, tzinfo=UTC),
            schema_version=2,
        )
        assert doc.schema_version == 2

    def test_empty_document_id_raises(self) -> None:
        with pytest.raises(ValueError, match="document_id"):
            CanonicalDocument(
                document_id="",
                source_ref=_make_source_ref(),
                blocks=[_make_block()],
                parser_version="1.0.0",
                parsed_at=datetime(2025, 1, 1, tzinfo=UTC),
            )

    def test_empty_blocks_raises(self) -> None:
        with pytest.raises(ValueError, match="blocks"):
            CanonicalDocument(
                document_id="doc-001",
                source_ref=_make_source_ref(),
                blocks=[],
                parser_version="1.0.0",
                parsed_at=datetime(2025, 1, 1, tzinfo=UTC),
            )

    def test_empty_parser_version_raises(self) -> None:
        with pytest.raises(ValueError, match="parser_version"):
            CanonicalDocument(
                document_id="doc-001",
                source_ref=_make_source_ref(),
                blocks=[_make_block()],
                parser_version="",
                parsed_at=datetime(2025, 1, 1, tzinfo=UTC),
            )
