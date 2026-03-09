"""Tests for the indexing subsystem: adapters, protocols, and service."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from libs.adapters.memory.chunk_repository import MemoryChunkRepository
from libs.adapters.memory.embedding_repository import MemoryEmbeddingRepository
from libs.adapters.memory.lexical_index_writer import MemoryLexicalIndexWriter
from libs.adapters.memory.vector_index_writer import MemoryVectorIndexWriter
from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.embeddings import ChunkEmbedding
from libs.embeddings.mock_provider import DeterministicEmbeddingProvider
from libs.indexing.lifecycle import IndexFreshness, MemoryIndexRegistry
from libs.indexing.models import ErrorClassification, IndexingOutcome
from libs.indexing.protocols import (
    ChunkRepository,
    EmbeddingRepository,
    LexicalIndexWriter,
    VectorIndexWriter,
)
from libs.indexing.service import IndexingService

# ── Helpers ─────────────────────────────────────────────────────────


def _make_chunk(
    content: str = "hello world",
    chunk_id: str = "chk-001",
    document_id: str = "doc-001",
    source_id: str = "src-001",
) -> Chunk:
    content_hash = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_id=source_id,
        block_ids=["blk-001"],
        content=content,
        content_hash=content_hash,
        token_count=len(content.split()),
        strategy="fixed_window",
        byte_offset_start=0,
        byte_offset_end=len(content.encode()),
        lineage=ChunkLineage(
            source_id=source_id,
            document_id=document_id,
            block_ids=["blk-001"],
            chunk_strategy="fixed_window",
            parser_version="test:1.0.0",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
        ),
    )


def _make_embedding(
    embedding_id: str = "emb-001",
    chunk_id: str = "chk-001",
    model_id: str = "test-model",
    model_version: str = "1.0",
    dimensions: int = 4,
) -> ChunkEmbedding:
    vector = [0.1 * (i + 1) for i in range(dimensions)]
    return ChunkEmbedding(
        embedding_id=embedding_id,
        chunk_id=chunk_id,
        vector=vector,
        model_id=model_id,
        model_version=model_version,
        dimensions=dimensions,
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _make_service() -> tuple[
    IndexingService,
    DeterministicEmbeddingProvider,
    MemoryChunkRepository,
    MemoryEmbeddingRepository,
    MemoryVectorIndexWriter,
    MemoryLexicalIndexWriter,
]:
    provider = DeterministicEmbeddingProvider()
    chunk_repo = MemoryChunkRepository()
    embedding_repo = MemoryEmbeddingRepository()
    vector_writer = MemoryVectorIndexWriter()
    lexical_writer = MemoryLexicalIndexWriter()
    service = IndexingService(
        embedding_provider=provider,
        chunk_repo=chunk_repo,
        embedding_repo=embedding_repo,
        vector_writer=vector_writer,
        lexical_writer=lexical_writer,
    )
    return service, provider, chunk_repo, embedding_repo, vector_writer, lexical_writer


# ── Protocol conformance ──────────────────────────────────────────


class TestProtocolConformance:
    def test_memory_chunk_repository_is_chunk_repository(self) -> None:
        assert isinstance(MemoryChunkRepository(), ChunkRepository)

    def test_memory_embedding_repository_is_embedding_repository(self) -> None:
        assert isinstance(MemoryEmbeddingRepository(), EmbeddingRepository)

    def test_memory_vector_index_writer_is_vector_index_writer(self) -> None:
        assert isinstance(MemoryVectorIndexWriter(), VectorIndexWriter)

    def test_memory_lexical_index_writer_is_lexical_index_writer(self) -> None:
        assert isinstance(MemoryLexicalIndexWriter(), LexicalIndexWriter)


# ── MemoryChunkRepository ─────────────────────────────────────────


class TestMemoryChunkRepository:
    def test_store_and_get(self) -> None:
        repo = MemoryChunkRepository()
        chunk = _make_chunk()
        repo.store(chunk)
        assert repo.get("chk-001") is chunk

    def test_get_missing_returns_none(self) -> None:
        repo = MemoryChunkRepository()
        assert repo.get("nonexistent") is None

    def test_store_batch(self) -> None:
        repo = MemoryChunkRepository()
        chunks = [
            _make_chunk(content="first", chunk_id="chk-1"),
            _make_chunk(content="second", chunk_id="chk-2"),
        ]
        repo.store_batch(chunks)
        assert repo.get("chk-1") is chunks[0]
        assert repo.get("chk-2") is chunks[1]

    def test_get_by_document(self) -> None:
        repo = MemoryChunkRepository()
        c1 = _make_chunk(content="alpha", chunk_id="chk-1", document_id="doc-A")
        c2 = _make_chunk(content="beta", chunk_id="chk-2", document_id="doc-A")
        c3 = _make_chunk(content="gamma", chunk_id="chk-3", document_id="doc-B")
        repo.store_batch([c1, c2, c3])
        result = repo.get_by_document("doc-A")
        assert len(result) == 2
        ids = {c.chunk_id for c in result}
        assert ids == {"chk-1", "chk-2"}

    def test_delete_by_document(self) -> None:
        repo = MemoryChunkRepository()
        c1 = _make_chunk(content="alpha", chunk_id="chk-1", document_id="doc-A")
        c2 = _make_chunk(content="beta", chunk_id="chk-2", document_id="doc-A")
        c3 = _make_chunk(content="gamma", chunk_id="chk-3", document_id="doc-B")
        repo.store_batch([c1, c2, c3])
        deleted = repo.delete_by_document("doc-A")
        assert deleted == 2
        assert repo.get("chk-1") is None
        assert repo.get("chk-2") is None
        assert repo.get("chk-3") is c3


# ── MemoryEmbeddingRepository ─────────────────────────────────────


class TestMemoryEmbeddingRepository:
    def test_store_and_get_by_chunk(self) -> None:
        repo = MemoryEmbeddingRepository()
        emb = _make_embedding()
        repo.store(emb)
        result = repo.get_by_chunk("chk-001")
        assert len(result) == 1
        assert result[0] is emb

    def test_get_by_chunk_and_model(self) -> None:
        repo = MemoryEmbeddingRepository()
        emb = _make_embedding()
        repo.store(emb)
        result = repo.get_by_chunk_and_model("chk-001", "test-model", "1.0")
        assert result is emb

    def test_get_by_chunk_and_model_missing(self) -> None:
        repo = MemoryEmbeddingRepository()
        emb = _make_embedding()
        repo.store(emb)
        result = repo.get_by_chunk_and_model("chk-001", "test-model", "2.0")
        assert result is None

    def test_list_by_model(self) -> None:
        repo = MemoryEmbeddingRepository()
        e1 = _make_embedding(embedding_id="emb-1", chunk_id="chk-1", model_version="1.0")
        e2 = _make_embedding(embedding_id="emb-2", chunk_id="chk-2", model_version="1.0")
        e3 = _make_embedding(embedding_id="emb-3", chunk_id="chk-3", model_version="2.0")
        repo.store_batch([e1, e2, e3])
        result = repo.list_by_model("test-model", "1.0")
        assert len(result) == 2

    def test_delete_by_chunk(self) -> None:
        repo = MemoryEmbeddingRepository()
        e1 = _make_embedding(embedding_id="emb-1", chunk_id="chk-1")
        e2 = _make_embedding(embedding_id="emb-2", chunk_id="chk-2")
        repo.store_batch([e1, e2])
        deleted = repo.delete_by_chunk("chk-1")
        assert deleted == 1
        assert repo.get_by_chunk("chk-1") == []
        assert len(repo.get_by_chunk("chk-2")) == 1

    def test_store_batch(self) -> None:
        repo = MemoryEmbeddingRepository()
        embeddings = [
            _make_embedding(embedding_id="emb-1", chunk_id="chk-1"),
            _make_embedding(embedding_id="emb-2", chunk_id="chk-2"),
        ]
        repo.store_batch(embeddings)
        assert repo.get_by_chunk_and_model("chk-1", "test-model", "1.0") is embeddings[0]
        assert repo.get_by_chunk_and_model("chk-2", "test-model", "1.0") is embeddings[1]


# ── MemoryVectorIndexWriter ───────────────────────────────────────


class TestMemoryVectorIndexWriter:
    def test_write_batch(self) -> None:
        writer = MemoryVectorIndexWriter()
        chunk = _make_chunk()
        emb = _make_embedding()
        count = writer.write_batch([emb], [chunk])
        assert count == 1

    def test_delete_by_chunk_ids(self) -> None:
        writer = MemoryVectorIndexWriter()
        c1 = _make_chunk(content="alpha", chunk_id="chk-1")
        c2 = _make_chunk(content="beta", chunk_id="chk-2")
        e1 = _make_embedding(embedding_id="emb-1", chunk_id="chk-1")
        e2 = _make_embedding(embedding_id="emb-2", chunk_id="chk-2")
        writer.write_batch([e1, e2], [c1, c2])
        deleted = writer.delete_by_chunk_ids(["chk-1"])
        assert deleted == 1


# ── MemoryLexicalIndexWriter ──────────────────────────────────────


class TestMemoryLexicalIndexWriter:
    def test_write_batch(self) -> None:
        writer = MemoryLexicalIndexWriter()
        chunks = [
            _make_chunk(content="alpha", chunk_id="chk-1"),
            _make_chunk(content="beta", chunk_id="chk-2"),
        ]
        count = writer.write_batch(chunks)
        assert count == 2

    def test_delete_by_chunk_ids(self) -> None:
        writer = MemoryLexicalIndexWriter()
        chunks = [
            _make_chunk(content="alpha", chunk_id="chk-1"),
            _make_chunk(content="beta", chunk_id="chk-2"),
        ]
        writer.write_batch(chunks)
        deleted = writer.delete_by_chunk_ids(["chk-1"])
        assert deleted == 1


# ── IndexingService ───────────────────────────────────────────────


class TestIndexingService:
    def test_full_pipeline_success(self) -> None:
        service, *_ = _make_service()
        chunks = [
            _make_chunk(content="alpha content", chunk_id="chk-1"),
            _make_chunk(content="beta content", chunk_id="chk-2"),
        ]
        result = service.run(chunks, run_id="run-001")
        assert result.outcome == IndexingOutcome.SUCCESS
        assert result.chunks_received == 2
        assert result.chunks_embedded == 2
        assert result.chunks_indexed_vector == 2
        assert result.chunks_indexed_lexical == 2

    def test_idempotent_skip(self) -> None:
        service, *_ = _make_service()
        chunks = [
            _make_chunk(content="alpha content", chunk_id="chk-1"),
            _make_chunk(content="beta content", chunk_id="chk-2"),
        ]
        service.run(chunks, run_id="run-001")
        result = service.run(chunks, run_id="run-002")
        assert result.chunks_embedded == 0

    def test_needs_reembedding_on_version_change(self) -> None:
        # Create service with v1 provider, embed a chunk, then check with v2 provider.
        provider_v1 = DeterministicEmbeddingProvider(model_version="1.0")
        chunk_repo = MemoryChunkRepository()
        embedding_repo = MemoryEmbeddingRepository()
        vector_writer = MemoryVectorIndexWriter()
        lexical_writer = MemoryLexicalIndexWriter()
        service_v1 = IndexingService(
            provider_v1, chunk_repo, embedding_repo, vector_writer, lexical_writer
        )
        chunk = _make_chunk(content="version test", chunk_id="chk-v")
        service_v1.run([chunk], run_id="run-v1")

        # Now create service with v2 provider
        provider_v2 = DeterministicEmbeddingProvider(model_version="2.0")
        service_v2 = IndexingService(
            provider_v2, chunk_repo, embedding_repo, vector_writer, lexical_writer
        )
        assert service_v2.needs_reembedding(chunk) is True

    def test_empty_input_returns_skipped(self) -> None:
        service, *_ = _make_service()
        result = service.run([], run_id="run-empty")
        assert result.outcome == IndexingOutcome.SKIPPED
        assert result.chunks_received == 0

    def test_chunks_stored_in_chunk_repo(self) -> None:
        service, _, chunk_repo, *_ = _make_service()
        chunks = [
            _make_chunk(content="stored chunk", chunk_id="chk-store"),
        ]
        service.run(chunks, run_id="run-store")
        assert chunk_repo.get("chk-store") is not None

    def test_embeddings_stored_in_embedding_repo(self) -> None:
        service, provider, _, embedding_repo, *_ = _make_service()
        chunk = _make_chunk(content="embed test", chunk_id="chk-emb")
        service.run([chunk], run_id="run-emb")
        info = provider.model_info()
        emb = embedding_repo.get_by_chunk_and_model(
            "chk-emb", info.model_id, info.model_version
        )
        assert emb is not None

    def test_vector_index_written(self) -> None:
        service, _, _, _, vector_writer, _ = _make_service()
        chunk = _make_chunk(content="vector test", chunk_id="chk-vec")
        result = service.run([chunk], run_id="run-vec")
        assert result.chunks_indexed_vector == 1
        # Verify data is in the writer's internal store
        assert "chk-vec" in vector_writer._store

    def test_lexical_index_written(self) -> None:
        service, _, _, _, _, lexical_writer = _make_service()
        chunk = _make_chunk(content="lexical test", chunk_id="chk-lex")
        result = service.run([chunk], run_id="run-lex")
        assert result.chunks_indexed_lexical == 1
        assert "chk-lex" in lexical_writer._store


# ── Failure handling ─────────────────────────────────────────────


class _FailingChunkRepository:
    """Chunk repository that raises on store_batch."""

    def store(self, chunk: Chunk) -> None:
        pass

    def store_batch(self, chunks: list[Chunk]) -> None:
        raise ConnectionError("chunk store unavailable")

    def get(self, chunk_id: str) -> Chunk | None:
        return None

    def get_by_document(self, document_id: str) -> list[Chunk]:
        return []

    def delete_by_document(self, document_id: str) -> int:
        return 0


class _FailingVectorWriter:
    """Vector writer that raises on write_batch."""

    def write_batch(self, embeddings: list[ChunkEmbedding], chunks: list[Chunk]) -> int:
        raise RuntimeError("vector index write error")

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
        return 0


class _FailingLexicalWriter:
    """Lexical writer that raises on write_batch."""

    def write_batch(self, chunks: list[Chunk]) -> int:
        raise OSError("lexical index disk full")

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
        return 0


class TestFailureHandling:
    """Tests for failure_mode_analysis skill compliance."""

    def test_chunk_store_failure_recorded(self) -> None:
        """chunk_repo.store_batch failure should be recorded with classification."""
        provider = DeterministicEmbeddingProvider()
        chunk_repo = _FailingChunkRepository()
        embedding_repo = MemoryEmbeddingRepository()
        vector_writer = MemoryVectorIndexWriter()
        lexical_writer = MemoryLexicalIndexWriter()
        service = IndexingService(
            embedding_provider=provider,
            chunk_repo=chunk_repo,
            embedding_repo=embedding_repo,
            vector_writer=vector_writer,
            lexical_writer=lexical_writer,
        )
        chunks = [_make_chunk(content="fail chunk", chunk_id="chk-fail")]
        result = service.run(chunks, run_id="run-fail-chunk")
        assert len(result.chunk_errors) >= 1
        err = result.chunk_errors[0]
        assert err.stage == "chunk_store"
        assert err.classification == ErrorClassification.TRANSIENT
        assert err.chunk_id == "chk-fail"

    def test_vector_write_failure_recorded(self) -> None:
        """vector_writer failure should produce PARTIAL/FAILED outcome with chunk_errors."""
        provider = DeterministicEmbeddingProvider()
        chunk_repo = MemoryChunkRepository()
        embedding_repo = MemoryEmbeddingRepository()
        vector_writer = _FailingVectorWriter()
        lexical_writer = MemoryLexicalIndexWriter()
        service = IndexingService(
            embedding_provider=provider,
            chunk_repo=chunk_repo,
            embedding_repo=embedding_repo,
            vector_writer=vector_writer,
            lexical_writer=lexical_writer,
        )
        chunks = [_make_chunk(content="fail vec", chunk_id="chk-fv")]
        result = service.run(chunks, run_id="run-fail-vec")
        assert result.chunks_indexed_vector == 0
        vec_errors = [e for e in result.chunk_errors if e.stage == "vector_index"]
        assert len(vec_errors) == 1
        assert vec_errors[0].classification == ErrorClassification.PERMANENT

    def test_lexical_write_failure_recorded(self) -> None:
        """lexical_writer failure should produce PARTIAL/FAILED with chunk_errors."""
        provider = DeterministicEmbeddingProvider()
        chunk_repo = MemoryChunkRepository()
        embedding_repo = MemoryEmbeddingRepository()
        vector_writer = MemoryVectorIndexWriter()
        lexical_writer = _FailingLexicalWriter()
        service = IndexingService(
            embedding_provider=provider,
            chunk_repo=chunk_repo,
            embedding_repo=embedding_repo,
            vector_writer=vector_writer,
            lexical_writer=lexical_writer,
        )
        chunks = [_make_chunk(content="fail lex", chunk_id="chk-fl")]
        result = service.run(chunks, run_id="run-fail-lex")
        assert result.chunks_indexed_lexical == 0
        lex_errors = [e for e in result.chunk_errors if e.stage == "lexical_index"]
        assert len(lex_errors) == 1
        assert lex_errors[0].classification == ErrorClassification.TRANSIENT

    def test_error_context_includes_run_id(self) -> None:
        """Error messages should include run_id for tracing."""
        provider = DeterministicEmbeddingProvider()
        chunk_repo = _FailingChunkRepository()
        embedding_repo = MemoryEmbeddingRepository()
        vector_writer = MemoryVectorIndexWriter()
        lexical_writer = MemoryLexicalIndexWriter()
        service = IndexingService(
            embedding_provider=provider,
            chunk_repo=chunk_repo,
            embedding_repo=embedding_repo,
            vector_writer=vector_writer,
            lexical_writer=lexical_writer,
        )
        chunks = [_make_chunk(content="trace test", chunk_id="chk-trace")]
        result = service.run(chunks, run_id="run-trace-42")
        assert any("run-trace-42" in e for e in result.errors)


# ── Index lifecycle ──────────────────────────────────────────────


class TestIndexLifecycle:
    """Tests for index_lifecycle_design skill compliance."""

    def test_create_version(self) -> None:
        registry = MemoryIndexRegistry()
        v = registry.create_version("idx-1", "model-a", "1.0", chunk_count=100)
        assert v.index_id == "idx-1"
        assert v.version == 1
        assert v.model_id == "model-a"
        assert v.chunk_count == 100
        assert v.is_active is False

    def test_activate_version(self) -> None:
        registry = MemoryIndexRegistry()
        registry.create_version("idx-1", "model-a", "1.0", chunk_count=50)
        registry.activate("idx-1", version=1)
        active = registry.get_active("idx-1")
        assert active is not None
        assert active.version == 1
        assert active.is_active is True

    def test_list_versions(self) -> None:
        registry = MemoryIndexRegistry()
        registry.create_version("idx-1", "model-a", "1.0", chunk_count=10)
        registry.create_version("idx-1", "model-a", "2.0", chunk_count=20)
        versions = registry.list_versions("idx-1")
        assert len(versions) == 2
        assert versions[0].version == 1
        assert versions[1].version == 2

    def test_get_active_returns_none_when_no_active(self) -> None:
        registry = MemoryIndexRegistry()
        registry.create_version("idx-1", "model-a", "1.0", chunk_count=10)
        assert registry.get_active("idx-1") is None

    def test_freshness_ratio(self) -> None:
        freshness = IndexFreshness(
            index_id="idx-1",
            last_updated_at=datetime(2025, 6, 1, tzinfo=UTC),
            document_count=100,
            stale_document_count=25,
        )
        assert freshness.freshness_ratio == 0.75

    def test_freshness_ratio_empty(self) -> None:
        freshness = IndexFreshness(
            index_id="idx-1",
            last_updated_at=datetime(2025, 6, 1, tzinfo=UTC),
            document_count=0,
            stale_document_count=0,
        )
        assert freshness.freshness_ratio == 1.0
