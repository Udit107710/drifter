"""Demo: retrieval broker with hybrid, dense, lexical, filtering, and source caps."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.retrieval import RetrievalQuery
from libs.embeddings import DeterministicEmbeddingProvider, DeterministicQueryEmbedder
from libs.retrieval.broker import (
    BrokerConfig,
    RetrievalBroker,
    RetrievalMode,
)
from libs.retrieval.stores.memory_lexical_store import MemoryLexicalStore
from libs.retrieval.stores.memory_vector_store import MemoryVectorStore


def _make_chunk(
    index: int,
    content: str,
    *,
    source_id: str,
    category: str,
    doc_id: str = "doc-001",
) -> Chunk:
    """Build a minimal Chunk with valid lineage and metadata."""
    content_hash = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
    block_id = f"blk-{index:03d}"
    chunk_id = f"chk-brk-{index:03d}"
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
        metadata={"category": category},
    )


def _print_broker_result(result) -> None:
    """Pretty-print a BrokerResult."""
    print(f"  Outcome:    {result.outcome.value}")
    print(f"  Mode:       {result.mode.value}")
    print(f"  Candidates: {result.candidate_count}")
    print(f"  Latency:    {result.total_latency_ms:.2f} ms")
    if result.errors:
        print(f"  Errors:     {result.errors}")
    print()
    for i, fc in enumerate(result.candidates, 1):
        print(
            f"  {i}. [{fc.retrieval_method.value}] score={fc.fused_score:.6f}  "
            f"chunk={fc.chunk.chunk_id}  "
            f"stores={fc.contributing_stores}  "
            f'"{fc.chunk.content[:55]}..."'
        )
    print()


def main() -> None:
    # ── Step 1: Create test chunks ───────────────────────────────────
    print("=" * 70)
    print("Step 1: Creating test chunks")
    print("=" * 70)

    chunks = [
        _make_chunk(
            0, "Machine learning models learn patterns from training data",
            source_id="src-tech", category="tech",
        ),
        _make_chunk(
            1, "Neural networks use gradient descent for optimization",
            source_id="src-tech", category="tech",
        ),
        _make_chunk(
            2, "Deep learning requires large datasets and GPU compute",
            source_id="src-tech", category="tech",
        ),
        _make_chunk(
            3, "A simple pasta recipe starts with boiling salted water",
            source_id="src-food", category="food",
        ),
        _make_chunk(
            4, "Cooking recipes often include precise temperature and timing",
            source_id="src-food", category="food",
        ),
        _make_chunk(
            5, "Sourdough bread requires a starter culture and long fermentation",
            source_id="src-food", category="food",
        ),
    ]

    for chunk in chunks:
        cat = chunk.metadata["category"]
        print(f"  {chunk.chunk_id} [src={chunk.source_id}, cat={cat}]: {chunk.content[:50]}...")
    print()

    # ── Step 2: Embed chunks ─────────────────────────────────────────
    print("=" * 70)
    print("Step 2: Embedding chunks with DeterministicEmbeddingProvider (dims=32)")
    print("=" * 70)

    provider = DeterministicEmbeddingProvider(
        model_id="demo-model",
        model_version="1.0",
        dimensions=32,
    )
    embeddings = provider.embed_chunks(chunks)

    info = provider.model_info()
    print(f"  Model: {info.model_id} v{info.model_version}, dims={info.dimensions}")
    print(f"  Embedded {len(embeddings)} chunks")
    print()

    # ── Step 3: Populate stores ──────────────────────────────────────
    print("=" * 70)
    print("Step 3: Populating MemoryVectorStore and MemoryLexicalStore")
    print("=" * 70)

    vector_store = MemoryVectorStore(store_id="demo-vector")
    lexical_store = MemoryLexicalStore(store_id="demo-lexical")

    vector_store.add_batch(embeddings, chunks)
    lexical_store.add_batch(chunks)

    print(f"  VectorStore count:  {vector_store.count()}")
    print(f"  LexicalStore count: {lexical_store.count()}")
    print()

    # ── Step 4: Create query embedder ────────────────────────────────
    print("=" * 70)
    print("Step 4: Creating DeterministicQueryEmbedder")
    print("=" * 70)

    query_embedder = DeterministicQueryEmbedder(
        model_id="demo-model",
        model_version="1.0",
        dimensions=32,
    )
    print("  QueryEmbedder ready (same model_id, model_version, dimensions)")
    print()

    # ── Step 5: Hybrid query ─────────────────────────────────────────
    print("=" * 70)
    print("Step 5: Hybrid query — 'machine learning training data'")
    print("=" * 70)

    broker_hybrid = RetrievalBroker(
        vector_store=vector_store,
        lexical_store=lexical_store,
        query_embedder=query_embedder,
        config=BrokerConfig(mode=RetrievalMode.HYBRID),
    )

    hybrid_query = RetrievalQuery(
        raw_query="machine learning training data",
        normalized_query="machine learning training data",
        trace_id="trace-hybrid-001",
        top_k=5,
    )

    hybrid_result = broker_hybrid.run(hybrid_query)
    _print_broker_result(hybrid_result)

    # ── Step 6: Dense-only query ─────────────────────────────────────
    print("=" * 70)
    print("Step 6: Dense-only query — 'machine learning training data'")
    print("=" * 70)

    broker_dense = RetrievalBroker(
        vector_store=vector_store,
        lexical_store=lexical_store,
        query_embedder=query_embedder,
        config=BrokerConfig(mode=RetrievalMode.DENSE),
    )

    dense_query = RetrievalQuery(
        raw_query="machine learning training data",
        normalized_query="machine learning training data",
        trace_id="trace-dense-001",
        top_k=5,
    )

    dense_result = broker_dense.run(dense_query)
    _print_broker_result(dense_result)

    # ── Step 7: Lexical-only query ───────────────────────────────────
    print("=" * 70)
    print("Step 7: Lexical-only query — 'machine learning training data'")
    print("=" * 70)

    broker_lexical = RetrievalBroker(
        vector_store=vector_store,
        lexical_store=lexical_store,
        query_embedder=query_embedder,
        config=BrokerConfig(mode=RetrievalMode.LEXICAL),
    )

    lexical_query = RetrievalQuery(
        raw_query="machine learning training data",
        normalized_query="machine learning training data",
        trace_id="trace-lexical-001",
        top_k=5,
    )

    lexical_result = broker_lexical.run(lexical_query)
    _print_broker_result(lexical_result)

    # ── Step 8: Metadata-filtered query ──────────────────────────────
    print("=" * 70)
    print("Step 8: Hybrid query with metadata filter — category='tech'")
    print("=" * 70)

    filtered_query = RetrievalQuery(
        raw_query="machine learning training data",
        normalized_query="machine learning training data",
        trace_id="trace-filtered-001",
        top_k=5,
        filters={"category": "tech"},
    )

    filtered_result = broker_hybrid.run(filtered_query)
    _print_broker_result(filtered_result)

    # ── Step 9: Source caps ──────────────────────────────────────────
    print("=" * 70)
    print("Step 9: Hybrid query with source caps — max_candidates_per_source=1")
    print("=" * 70)

    broker_capped = RetrievalBroker(
        vector_store=vector_store,
        lexical_store=lexical_store,
        query_embedder=query_embedder,
        config=BrokerConfig(
            mode=RetrievalMode.HYBRID,
            max_candidates_per_source=1,
        ),
    )

    capped_query = RetrievalQuery(
        raw_query="machine learning training data",
        normalized_query="machine learning training data",
        trace_id="trace-capped-001",
        top_k=5,
    )

    capped_result = broker_capped.run(capped_query)
    _print_broker_result(capped_result)

    # ── Step 10: Debug payload ───────────────────────────────────────
    print("=" * 70)
    print("Step 10: Debug payload from the capped hybrid run")
    print("=" * 70)

    for key, value in capped_result.debug.items():
        print(f"  {key}: {value}")
    print()

    # Store results detail
    print("  Store results:")
    for sr in capped_result.store_results:
        print(
            f"    {sr.store_id} [{sr.retrieval_method.value}]: "
            f"{sr.candidate_count} candidates, {sr.latency_ms:.2f} ms"
            f"{', error=' + sr.error if sr.error else ''}"
        )
    print()

    print("Done.")


if __name__ == "__main__":
    main()
