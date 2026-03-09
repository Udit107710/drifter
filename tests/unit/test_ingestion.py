"""Tests for the ingestion subsystem: change detector, service, connector, adapters."""

from __future__ import annotations

from pathlib import Path

import pytest

from libs.adapters.memory import MemoryCrawlStateRepository, MemorySourceRepository
from libs.ingestion.change_detector import detect
from libs.ingestion.connectors.filesystem import LocalFilesystemConnector
from libs.ingestion.models import (
    CrawlState,
    CrawlStatus,
    IngestionOutcome,
    SourceConfig,
    SourceType,
    WorkAction,
)
from libs.ingestion.service import IngestionService

# ── Helpers ─────────────────────────────────────────────────────────


def _make_source_config(**overrides: object) -> SourceConfig:
    defaults: dict[str, object] = {
        "source_id": "src-001",
        "uri": "/tmp/test.txt",
        "source_type": SourceType.FILESYSTEM,
    }
    defaults.update(overrides)
    return SourceConfig(**defaults)  # type: ignore[arg-type]


def _make_crawl_state(**overrides: object) -> CrawlState:
    defaults: dict[str, object] = {
        "source_id": "src-001",
    }
    defaults.update(overrides)
    return CrawlState(**defaults)  # type: ignore[arg-type]


# ── TestChangeDetector ──────────────────────────────────────────────


class TestChangeDetector:
    def test_no_crawl_state_returns_ingest(self) -> None:
        assert detect(None, "sha256:abc") == WorkAction.INGEST

    def test_pending_status_returns_ingest(self) -> None:
        state = _make_crawl_state(status=CrawlStatus.PENDING)
        assert detect(state, "sha256:abc") == WorkAction.INGEST

    def test_same_hash_returns_skip(self) -> None:
        state = _make_crawl_state(
            last_content_hash="sha256:abc",
            status=CrawlStatus.COMPLETED,
        )
        assert detect(state, "sha256:abc") == WorkAction.SKIP

    def test_different_hash_returns_update(self) -> None:
        state = _make_crawl_state(
            last_content_hash="sha256:abc",
            status=CrawlStatus.COMPLETED,
        )
        assert detect(state, "sha256:def") == WorkAction.UPDATE

    def test_none_hash_returns_tombstone(self) -> None:
        state = _make_crawl_state(status=CrawlStatus.COMPLETED)
        assert detect(state, None) == WorkAction.TOMBSTONE

    def test_none_hash_no_state_returns_tombstone(self) -> None:
        assert detect(None, None) == WorkAction.TOMBSTONE


# ── TestIngestionService ────────────────────────────────────────────


class TestIngestionService:
    def _make_service(self) -> tuple[
        IngestionService, MemorySourceRepository, MemoryCrawlStateRepository
    ]:
        source_repo = MemorySourceRepository()
        crawl_state_repo = MemoryCrawlStateRepository()
        connector = LocalFilesystemConnector()
        service = IngestionService(source_repo, crawl_state_repo, connector, detect)
        return service, source_repo, crawl_state_repo

    def test_new_document_ingested(self, tmp_path: Path) -> None:
        service, source_repo, _crawl_state_repo = self._make_service()
        filepath = tmp_path / "doc.txt"
        filepath.write_text("hello world")

        config = _make_source_config(source_id="src-new", uri=str(filepath))
        source_repo.add(config)

        results = service.run("run-001")
        assert len(results) == 1
        r = results[0]
        assert r.outcome == IngestionOutcome.SUCCESS
        assert r.work_item.action == WorkAction.INGEST
        assert r.source_ref is not None
        assert r.source_ref.version == 1
        assert r.raw_document is not None
        assert r.raw_document.raw_bytes == b"hello world"

    def test_unchanged_document_skipped(self, tmp_path: Path) -> None:
        service, source_repo, _crawl_state_repo = self._make_service()
        filepath = tmp_path / "doc.txt"
        filepath.write_text("hello world")

        config = _make_source_config(source_id="src-same", uri=str(filepath))
        source_repo.add(config)

        # First run: ingest
        service.run("run-001")
        # Second run: skip
        results = service.run("run-002")
        assert len(results) == 1
        assert results[0].outcome == IngestionOutcome.SKIPPED
        assert results[0].work_item.action == WorkAction.SKIP

    def test_changed_document_updated(self, tmp_path: Path) -> None:
        service, source_repo, _crawl_state_repo = self._make_service()
        filepath = tmp_path / "doc.txt"
        filepath.write_text("version one")

        config = _make_source_config(source_id="src-chg", uri=str(filepath))
        source_repo.add(config)

        service.run("run-001")

        # Modify file
        filepath.write_text("version two")
        results = service.run("run-002")
        assert len(results) == 1
        r = results[0]
        assert r.outcome == IngestionOutcome.SUCCESS
        assert r.work_item.action == WorkAction.UPDATE
        assert r.source_ref is not None
        assert r.source_ref.version == 2

    def test_deleted_source_tombstoned(self, tmp_path: Path) -> None:
        service, source_repo, crawl_state_repo = self._make_service()
        filepath = tmp_path / "doc.txt"
        filepath.write_text("to be deleted")

        config = _make_source_config(source_id="src-del", uri=str(filepath))
        source_repo.add(config)

        service.run("run-001")

        # Delete file
        filepath.unlink()
        results = service.run("run-002")
        assert len(results) == 1
        r = results[0]
        assert r.outcome == IngestionOutcome.SUCCESS
        assert r.work_item.action == WorkAction.TOMBSTONE

        state = crawl_state_repo.get("src-del")
        assert state is not None
        assert state.status == CrawlStatus.TOMBSTONED

    def test_replay_safe_rerun(self, tmp_path: Path) -> None:
        service, source_repo, _crawl_state_repo = self._make_service()
        filepath = tmp_path / "doc.txt"
        filepath.write_text("replay test")

        config = _make_source_config(source_id="src-replay", uri=str(filepath))
        source_repo.add(config)

        # First run
        service.run("run-001")
        # Same run_id again — should skip due to replay safety
        results = service.run("run-001")
        assert len(results) == 1
        assert results[0].outcome == IngestionOutcome.SKIPPED


# ── TestLocalFilesystemConnector ────────────────────────────────────


class TestLocalFilesystemConnector:
    def test_fetch_existing_file(self, tmp_path: Path) -> None:
        filepath = tmp_path / "test.txt"
        filepath.write_text("content here")

        connector = LocalFilesystemConnector()
        config = _make_source_config(uri=str(filepath))
        result = connector.fetch(config)

        assert result is not None
        assert result.raw_bytes == b"content here"
        assert result.mime_type == "text/plain"
        assert result.size_bytes == len(b"content here")

    def test_fetch_missing_file_returns_none(self) -> None:
        connector = LocalFilesystemConnector()
        config = _make_source_config(uri="/nonexistent/path/file.txt")
        result = connector.fetch(config)
        assert result is None

    def test_list_source_ids_directory(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.txt").write_text("c")

        connector = LocalFilesystemConnector()
        config = _make_source_config(uri=str(tmp_path))
        ids = connector.list_source_ids(config)

        assert len(ids) == 3
        assert any("a.txt" in s for s in ids)
        assert any("c.txt" in s for s in ids)


# ── TestMemorySourceRepository ──────────────────────────────────────


class TestMemorySourceRepository:
    def test_add_and_get(self) -> None:
        repo = MemorySourceRepository()
        config = _make_source_config()
        repo.add(config)
        assert repo.get("src-001") is config

    def test_list_enabled_excludes_disabled(self) -> None:
        repo = MemorySourceRepository()
        repo.add(_make_source_config(source_id="e1", enabled=True))
        repo.add(_make_source_config(source_id="e2", enabled=False))
        enabled = repo.list_enabled()
        assert len(enabled) == 1
        assert enabled[0].source_id == "e1"

    def test_remove(self) -> None:
        repo = MemorySourceRepository()
        repo.add(_make_source_config())
        repo.remove("src-001")
        assert repo.get("src-001") is None

    def test_add_duplicate_raises(self) -> None:
        repo = MemorySourceRepository()
        repo.add(_make_source_config())
        with pytest.raises(ValueError, match="already registered"):
            repo.add(_make_source_config())


# ── TestMemoryCrawlStateRepository ──────────────────────────────────


class TestMemoryCrawlStateRepository:
    def test_save_and_get(self) -> None:
        repo = MemoryCrawlStateRepository()
        state = _make_crawl_state()
        repo.save(state)
        assert repo.get("src-001") is state

    def test_missing_returns_none(self) -> None:
        repo = MemoryCrawlStateRepository()
        assert repo.get("nonexistent") is None

    def test_save_overwrites(self) -> None:
        repo = MemoryCrawlStateRepository()
        state1 = _make_crawl_state()
        state2 = _make_crawl_state(last_content_hash="sha256:new")
        repo.save(state1)
        repo.save(state2)
        assert repo.get("src-001") is state2
