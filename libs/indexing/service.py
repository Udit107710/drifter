"""Indexing service: orchestrates embedding, storage, and index writing."""

from __future__ import annotations

from datetime import UTC, datetime

from libs.contracts.chunks import Chunk
from libs.contracts.common import RunId
from libs.embeddings.protocols import EmbeddingProvider
from libs.indexing.models import ChunkError, ErrorClassification, IndexingOutcome, IndexingResult
from libs.indexing.protocols import (
    ChunkRepository,
    EmbeddingRepository,
    LexicalIndexWriter,
    VectorIndexWriter,
)
from libs.resilience import is_transient_error


def _classify_error(exc: Exception) -> ErrorClassification:
    """Classify an exception as transient or permanent."""
    if is_transient_error(exc):
        return ErrorClassification.TRANSIENT
    return ErrorClassification.PERMANENT


class IndexingService:
    """Orchestrates the indexing pipeline: embed → store → index."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        chunk_repo: ChunkRepository,
        embedding_repo: EmbeddingRepository,
        vector_writer: VectorIndexWriter,
        lexical_writer: LexicalIndexWriter,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._chunk_repo = chunk_repo
        self._embedding_repo = embedding_repo
        self._vector_writer = vector_writer
        self._lexical_writer = lexical_writer

    def run(self, chunks: list[Chunk], run_id: RunId) -> IndexingResult:
        """Execute an indexing run for the given chunks.

        Algorithm:
        1. Get model_info from embedding provider
        2. Filter: skip chunks already embedded for this model_id + model_version (idempotency)
        3. embed_chunks() on filtered list
        4. embedding_repo.store_batch() new embeddings
        5. chunk_repo.store_batch() all chunks
        6. vector_writer.write_batch() with all embeddings + chunks
        7. lexical_writer.write_batch() with all chunks
        8. Return IndexingResult with counts
        """
        info = self._embedding_provider.model_info()
        errors: list[str] = []
        chunk_errors: list[ChunkError] = []

        if not chunks:
            return IndexingResult(
                run_id=run_id,
                chunks_received=0,
                chunks_embedded=0,
                chunks_indexed_vector=0,
                chunks_indexed_lexical=0,
                outcome=IndexingOutcome.SKIPPED,
                model_info=info,
                completed_at=datetime.now(UTC),
            )

        # Step 2: filter already-embedded chunks
        to_embed = [
            c for c in chunks
            if self._embedding_repo.get_by_chunk_and_model(
                c.chunk_id, info.model_id, info.model_version
            ) is None
        ]

        # Step 3: embed
        new_embeddings = []
        if to_embed:
            try:
                new_embeddings = self._embedding_provider.embed_chunks(to_embed)
            except Exception as exc:
                errors.append(f"Embedding failed: {exc}")
                return IndexingResult(
                    run_id=run_id,
                    chunks_received=len(chunks),
                    chunks_embedded=0,
                    chunks_indexed_vector=0,
                    chunks_indexed_lexical=0,
                    outcome=IndexingOutcome.FAILED,
                    model_info=info,
                    completed_at=datetime.now(UTC),
                    errors=errors,
                )

        # Step 4: store embeddings
        if new_embeddings:
            self._embedding_repo.store_batch(new_embeddings)

        # Step 5: store all chunks
        try:
            self._chunk_repo.store_batch(chunks)
        except Exception as exc:
            classification = _classify_error(exc)
            errors.append(f"run_id={run_id} chunk_store failed: {exc}")
            chunk_errors.extend(
                ChunkError(
                    chunk_id=c.chunk_id,
                    stage="chunk_store",
                    error=str(exc),
                    classification=classification,
                )
                for c in chunks
            )

        # Step 6: write to vector index (all embeddings for this run)
        # Gather all embeddings including previously stored ones
        all_embeddings = []
        for chunk in chunks:
            emb = self._embedding_repo.get_by_chunk_and_model(
                chunk.chunk_id, info.model_id, info.model_version
            )
            if emb is not None:
                all_embeddings.append(emb)

        try:
            vector_count = self._vector_writer.write_batch(all_embeddings, chunks)
        except Exception as exc:
            classification = _classify_error(exc)
            errors.append(f"run_id={run_id} vector_index failed: {exc}")
            chunk_errors.extend(
                ChunkError(
                    chunk_id=c.chunk_id,
                    stage="vector_index",
                    error=str(exc),
                    classification=classification,
                )
                for c in chunks
            )
            vector_count = 0

        # Step 7: write to lexical index
        try:
            lexical_count = self._lexical_writer.write_batch(chunks)
        except Exception as exc:
            classification = _classify_error(exc)
            errors.append(f"run_id={run_id} lexical_index failed: {exc}")
            chunk_errors.extend(
                ChunkError(
                    chunk_id=c.chunk_id,
                    stage="lexical_index",
                    error=str(exc),
                    classification=classification,
                )
                for c in chunks
            )
            lexical_count = 0

        # Determine outcome
        if errors or len(all_embeddings) < len(chunks):
            outcome = IndexingOutcome.PARTIAL
        else:
            outcome = IndexingOutcome.SUCCESS

        return IndexingResult(
            run_id=run_id,
            chunks_received=len(chunks),
            chunks_embedded=len(new_embeddings),
            chunks_indexed_vector=vector_count,
            chunks_indexed_lexical=lexical_count,
            outcome=outcome,
            model_info=info,
            completed_at=datetime.now(UTC),
            errors=errors,
            chunk_errors=chunk_errors,
        )

    def needs_reembedding(self, chunk: Chunk) -> bool:
        """Check if existing embedding model_version differs from provider's current version."""
        info = self._embedding_provider.model_info()
        existing = self._embedding_repo.get_by_chunk_and_model(
            chunk.chunk_id, info.model_id, info.model_version
        )
        if existing is None:
            # No embedding for current version — check if there's one for another version
            embeddings = self._embedding_repo.get_by_chunk(chunk.chunk_id)
            if not embeddings:
                return True  # No embedding at all
            # Has embedding but for different version
            return any(e.model_version != info.model_version for e in embeddings)
        return False
