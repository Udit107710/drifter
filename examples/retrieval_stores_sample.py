"""Demo: retrieval stores with vector and lexical search."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.retrieval import RetrievalQuery
from libs.embeddings import DeterministicEmbeddingProvider
from libs.retrieval.stores.memory_lexical_store import MemoryLexicalStore
from libs.retrieval.stores.memory_vector_store import MemoryVectorStore


def _make_chunk(
    index: int, content: str, *, category: str, doc_id: str = "doc-001"
) -> Chunk:
    """Build a minimal Chunk with valid lineage and metadata."""
    content_hash = f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"
    source_id = "src-001"
    block_id = f"blk-{index:03d}"
    chunk_id = f"chk-ret-{index:03d}"
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


def _print_candidates(candidates: list) -> None:
    """Pretty-print a list of RetrievalCandidate objects."""
    if not candidates:
        print("  (no results)")
        return
    for i, c in enumerate(candidates, 1):
        print(
            f"  {i}. [{c.retrieval_method.value}] score={c.score:.4f}  "
            f"chunk={c.chunk.chunk_id}  "
            f'"{c.chunk.content[:60]}..."'
        )


def main() -> None:
    # ── Step 1: Create test chunks ───────────────────────────────────
    print("Step 1: Creating test chunks")
    print("-" * 60)

    chunks = [
        _make_chunk(
            0, "Machine learning models learn patterns from training data",
            category="tech",
        ),
        _make_chunk(
            1, "Neural networks use gradient descent for optimization",
            category="tech",
        ),
        _make_chunk(
            2, "Deep learning requires large datasets and GPU compute",
            category="tech",
        ),
        _make_chunk(
            3, "A simple pasta recipe starts with boiling salted water",
            category="food",
        ),
        _make_chunk(
            4, "Cooking recipes often include precise temperature and timing",
            category="food",
        ),
        _make_chunk(
            5, "Sourdough bread requires a starter culture and long fermentation",
            category="food",
        ),
    ]

    for chunk in chunks:
        cat = chunk.metadata["category"]
        print(f"  {chunk.chunk_id} [{cat}]: {chunk.content[:55]}...")
    print()

    # ── Step 2: Create embeddings ────────────────────────────────────
    print("Step 2: Embedding chunks with DeterministicEmbeddingProvider")
    print("-" * 60)

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
    print("Step 3: Populating MemoryVectorStore and MemoryLexicalStore")
    print("-" * 60)

    vector_store = MemoryVectorStore(store_id="demo-vector")
    lexical_store = MemoryLexicalStore(store_id="demo-lexical")

    vector_store.add_batch(embeddings, chunks)
    lexical_store.add_batch(chunks)

    print(f"  VectorStore count:  {vector_store.count()}")
    print(f"  LexicalStore count: {lexical_store.count()}")
    print()

    # ── Step 4: Vector search ────────────────────────────────────────
    print("Step 4: Vector search — query about 'machine learning optimization'")
    print("-" * 60)

    # Embed a query-like chunk to get a query vector
    query_chunk = _make_chunk(
        99,
        "machine learning optimization and gradient methods",
        category="tech",
        doc_id="query-doc",
    )
    query_embedding = provider.embed_chunks([query_chunk])[0]

    query = RetrievalQuery(
        raw_query="machine learning optimization",
        normalized_query="machine learning optimization",
        trace_id="trace-demo-001",
        top_k=3,
    )

    vector_results = vector_store.search(query, query_embedding.vector)
    _print_candidates(vector_results)
    print()

    # ── Step 5: Lexical search ───────────────────────────────────────
    print("Step 5: Lexical search — query for 'learning models training data'")
    print("-" * 60)

    lexical_query = RetrievalQuery(
        raw_query="learning models training data",
        normalized_query="learning models training data",
        trace_id="trace-demo-002",
        top_k=3,
    )

    lexical_results = lexical_store.search(lexical_query)
    _print_candidates(lexical_results)
    print()

    # ── Step 6: Metadata-filtered search ─────────────────────────────
    print("Step 6: Lexical search with metadata filter — category='food'")
    print("-" * 60)

    food_query = RetrievalQuery(
        raw_query="recipe temperature cooking",
        normalized_query="recipe temperature cooking",
        trace_id="trace-demo-003",
        top_k=3,
        filters={"category": "food"},
    )

    food_results = lexical_store.search(food_query)
    _print_candidates(food_results)
    print()

    print("Step 6b: Vector search with metadata filter — category='tech'")
    print("-" * 60)

    tech_vector_query = RetrievalQuery(
        raw_query="deep learning compute",
        normalized_query="deep learning compute",
        trace_id="trace-demo-004",
        top_k=3,
        filters={"category": "tech"},
    )

    tech_results = vector_store.search(tech_vector_query, query_embedding.vector)
    _print_candidates(tech_results)
    print()

    # ── Step 7: Empty result case ────────────────────────────────────
    print("Step 7: Empty result — query with no matching terms")
    print("-" * 60)

    empty_query = RetrievalQuery(
        raw_query="quantum entanglement superconductor",
        normalized_query="quantum entanglement superconductor",
        trace_id="trace-demo-005",
        top_k=3,
    )

    empty_results = lexical_store.search(empty_query)
    _print_candidates(empty_results)
    print()

    print("Done.")


if __name__ == "__main__":
    main()
