"""Demo: embedding and indexing pipeline with in-memory adapters."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from libs.adapters.memory import (
    MemoryChunkRepository,
    MemoryEmbeddingRepository,
    MemoryLexicalIndexWriter,
    MemoryVectorIndexWriter,
)
from libs.contracts.chunks import Chunk, ChunkLineage
from libs.embeddings import DeterministicEmbeddingProvider
from libs.indexing import IndexingService


def _make_chunk(index: int, content: str) -> Chunk:
    """Build a minimal Chunk with valid lineage for demonstration."""
    content_hash = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
    doc_id = "doc-001"
    source_id = "src-001"
    block_id = f"blk-{index:03d}"
    chunk_id = f"chk-demo-{index:03d}"
    encoded = content.encode()

    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        source_id=source_id,
        block_ids=[block_id],
        content=content,
        content_hash=content_hash,
        token_count=len(content.split()),
        strategy="fixed_window",
        byte_offset_start=index * 200,
        byte_offset_end=index * 200 + len(encoded),
        lineage=ChunkLineage(
            source_id=source_id,
            document_id=doc_id,
            block_ids=[block_id],
            chunk_strategy="fixed_window",
            parser_version="markdown-1.0",
            created_at=datetime.now(UTC),
        ),
    )


def main() -> None:
    # ── Step 1: Create test chunks ───────────────────────────────────
    print("Step 1: Creating test chunks")
    print("-" * 50)

    chunks = [
        _make_chunk(0, "Retrieval augmented generation combines search with language models."),
        _make_chunk(1, "Vector embeddings encode semantic meaning into dense numerical arrays."),
        _make_chunk(2, "Lexical search uses term frequency for keyword matching."),
    ]

    for chunk in chunks:
        print(f"  {chunk.chunk_id}: {chunk.content[:60]}...")
    print()

    # ── Step 2: Create a deterministic embedding provider ────────────
    print("Step 2: Creating DeterministicEmbeddingProvider")
    print("-" * 50)

    provider = DeterministicEmbeddingProvider(
        model_id="demo-model",
        model_version="1.0",
        dimensions=32,
    )
    info = provider.model_info()
    print(f"  Model:      {info.model_id} v{info.model_version}")
    print(f"  Dimensions: {info.dimensions}")
    print(f"  Max tokens: {info.max_tokens}")
    print()

    # ── Step 3: Create in-memory adapters ────────────────────────────
    print("Step 3: Creating in-memory adapters")
    print("-" * 50)

    chunk_repo = MemoryChunkRepository()
    embedding_repo = MemoryEmbeddingRepository()
    vector_writer = MemoryVectorIndexWriter()
    lexical_writer = MemoryLexicalIndexWriter()

    print("  ChunkRepository:     MemoryChunkRepository")
    print("  EmbeddingRepository: MemoryEmbeddingRepository")
    print("  VectorIndexWriter:   MemoryVectorIndexWriter")
    print("  LexicalIndexWriter:  MemoryLexicalIndexWriter")
    print()

    # ── Step 4: Create the IndexingService ───────────────────────────
    print("Step 4: Creating IndexingService")
    print("-" * 50)

    service = IndexingService(
        embedding_provider=provider,
        chunk_repo=chunk_repo,
        embedding_repo=embedding_repo,
        vector_writer=vector_writer,
        lexical_writer=lexical_writer,
    )
    print("  IndexingService wired with all adapters.")
    print()

    # ── Step 5: Run the indexing pipeline ────────────────────────────
    print("Step 5: Running indexing pipeline (first run)")
    print("-" * 50)

    result = service.run(chunks, run_id="run-001")

    print(f"  Run ID:                {result.run_id}")
    print(f"  Outcome:               {result.outcome.value}")
    print(f"  Chunks received:       {result.chunks_received}")
    print(f"  Chunks embedded:       {result.chunks_embedded}")
    print(f"  Chunks indexed vector: {result.chunks_indexed_vector}")
    print(f"  Chunks indexed lexical:{result.chunks_indexed_lexical}")
    model = f"{result.model_info.model_id} v{result.model_info.model_version}"
    print(f"  Model:                 {model}")
    print(f"  Errors:                {result.errors or 'none'}")
    print()

    # ── Step 6: Demonstrate idempotency ──────────────────────────────
    print("Step 6: Running indexing pipeline again (idempotency check)")
    print("-" * 50)

    result2 = service.run(chunks, run_id="run-002")

    print(f"  Run ID:                {result2.run_id}")
    print(f"  Outcome:               {result2.outcome.value}")
    print(f"  Chunks received:       {result2.chunks_received}")
    print(f"  Chunks embedded:       {result2.chunks_embedded}  <-- 0 means already embedded")
    print(f"  Chunks indexed vector: {result2.chunks_indexed_vector}")
    print(f"  Chunks indexed lexical:{result2.chunks_indexed_lexical}")
    print()

    # ── Step 7: Show needs_reembedding ───────────────────────────────
    print("Step 7: Checking needs_reembedding")
    print("-" * 50)

    for chunk in chunks:
        needs = service.needs_reembedding(chunk)
        print(f"  {chunk.chunk_id}: needs_reembedding = {needs}")

    # Now create a service with a newer model version
    provider_v2 = DeterministicEmbeddingProvider(
        model_id="demo-model",
        model_version="2.0",
        dimensions=32,
    )
    service_v2 = IndexingService(
        embedding_provider=provider_v2,
        chunk_repo=chunk_repo,
        embedding_repo=embedding_repo,
        vector_writer=vector_writer,
        lexical_writer=lexical_writer,
    )

    print()
    print("  After upgrading provider to v2.0:")
    for chunk in chunks:
        needs = service_v2.needs_reembedding(chunk)
        print(f"  {chunk.chunk_id}: needs_reembedding = {needs}  <-- model version changed")


if __name__ == "__main__":
    main()
