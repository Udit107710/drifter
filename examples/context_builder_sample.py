"""Example: context builder pipeline from RankedCandidates to ContextPack."""

from __future__ import annotations

from datetime import UTC, datetime

from libs.chunking.token_counter import WhitespaceTokenCounter
from libs.context_builder.greedy_builder import GreedyContextBuilder
from libs.context_builder.models import BuilderConfig, BuilderResult
from libs.contracts.chunks import Chunk, ChunkLineage
from libs.contracts.common import RetrievalMethod
from libs.contracts.retrieval import RankedCandidate, RetrievalCandidate

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lineage(source_id: str = "src-1") -> ChunkLineage:
    return ChunkLineage(
        source_id=source_id,
        document_id=f"doc-{source_id}",
        block_ids=["b1"],
        chunk_strategy="fixed",
        parser_version="1.0",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
    )


def _make_chunk(
    chunk_id: str,
    content: str,
    source_id: str = "src-1",
    content_hash: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=f"doc-{source_id}",
        source_id=source_id,
        block_ids=["b1"],
        content=content,
        content_hash=content_hash or f"hash-{chunk_id}",
        token_count=len(content.split()),
        strategy="fixed",
        byte_offset_start=0,
        byte_offset_end=len(content),
        lineage=_make_lineage(source_id),
        metadata={},
    )


def _make_ranked(
    chunk_id: str,
    content: str,
    rank: int,
    score: float,
    source_id: str = "src-1",
    content_hash: str | None = None,
) -> RankedCandidate:
    chunk = _make_chunk(chunk_id, content, source_id=source_id, content_hash=content_hash)
    candidate = RetrievalCandidate(
        chunk=chunk,
        score=score,
        retrieval_method=RetrievalMethod.HYBRID,
        store_id="qdrant-1",
    )
    return RankedCandidate(
        candidate=candidate,
        rank=rank,
        rerank_score=score,
        reranker_id="feature-based-v1",
    )


def _print_result(result: BuilderResult, label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"Outcome       : {result.outcome.value}")
    print(f"Input count   : {result.input_count}")
    print(f"Dedup removed : {result.dedup_removed}")
    pack = result.context_pack
    print(f"Tokens used   : {pack.total_tokens} / {pack.token_budget}")
    print(f"Diversity     : {pack.diversity_score:.2f}")
    print(f"Latency       : {result.total_latency_ms:.2f} ms")

    print("\n--- Included chunks ---")
    for item in pack.evidence:
        print(
            f"  Rank {item.rank}: {item.chunk.chunk_id} "
            f"({item.token_count} tokens, reason={item.selection_reason.value}) "
            f"[source={item.chunk.source_id}]"
        )
        print(f"    Content: {item.chunk.content[:80]}...")

    print("\n--- Excluded chunks ---")
    if not result.exclusions:
        print("  (none)")
    for excl in result.exclusions:
        print(f"  {excl.chunk_id}: {excl.reason} ({excl.token_count} tokens)")

    print("\n--- Debug metadata ---")
    for key, value in result.debug.items():
        print(f"  {key}: {value}")


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def _build_candidates() -> list[RankedCandidate]:
    """Create ranked candidates from three different sources with varied content."""
    return [
        _make_ranked(
            "c1",
            "Neural networks are a class of machine learning models inspired by "
            "biological neural networks. They consist of layers of interconnected "
            "nodes that process information using learnable weights and biases.",
            rank=1,
            score=0.95,
            source_id="textbook-ml",
        ),
        _make_ranked(
            "c2",
            "Gradient descent is an optimization algorithm used to minimize loss "
            "functions. Variants include stochastic gradient descent, Adam, and "
            "RMSProp, each with different convergence properties.",
            rank=2,
            score=0.89,
            source_id="blog-optimization",
        ),
        _make_ranked(
            "c3",
            "Convolutional neural networks apply learnable filters to input data "
            "making them effective for image recognition tasks. Key components "
            "include convolutional layers pooling layers and fully connected layers.",
            rank=3,
            score=0.84,
            source_id="textbook-ml",
        ),
        _make_ranked(
            "c4",
            "Transfer learning allows models pretrained on large datasets to be "
            "fine-tuned for specific downstream tasks reducing the need for large "
            "labeled datasets. BERT and GPT are prominent examples in NLP.",
            rank=4,
            score=0.80,
            source_id="survey-paper",
        ),
        _make_ranked(
            "c5",
            "Regularization techniques such as dropout batch normalization and "
            "weight decay help prevent overfitting in deep learning models by "
            "adding constraints during training.",
            rank=5,
            score=0.76,
            source_id="blog-optimization",
        ),
        # Duplicate of c1 (same content_hash) to demonstrate dedup
        _make_ranked(
            "c6-dup",
            "Neural networks are a class of machine learning models inspired by "
            "biological neural networks. They consist of layers of interconnected "
            "nodes that process information using learnable weights and biases.",
            rank=6,
            score=0.70,
            source_id="textbook-ml",
            content_hash="hash-c1",  # same hash as c1
        ),
        _make_ranked(
            "c7",
            "Attention mechanisms allow models to focus on relevant parts of the "
            "input sequence. The transformer architecture built entirely on "
            "self-attention has become the foundation of modern NLP.",
            rank=7,
            score=0.65,
            source_id="research-paper",
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    counter = WhitespaceTokenCounter()
    candidates = _build_candidates()

    # 1. Greedy builder with a generous budget (fits most chunks)
    print("\n" + "#" * 60)
    print("# Scenario 1: Greedy builder, generous budget (200 tokens)")
    print("#" * 60)

    greedy = GreedyContextBuilder(
        token_counter=counter,
        config=BuilderConfig(token_budget=200, deduplicate=True),
    )
    result = greedy.build(candidates, query="neural network optimization", token_budget=200)
    _print_result(result, "Greedy Builder (200 token budget)")

    # 2. Greedy builder with a tight budget (forces exclusions)
    print("\n" + "#" * 60)
    print("# Scenario 2: Greedy builder, tight budget (80 tokens)")
    print("#" * 60)

    tight_greedy = GreedyContextBuilder(
        token_counter=counter,
        config=BuilderConfig(token_budget=80, deduplicate=True),
    )
    result_tight = tight_greedy.build(
        candidates, query="neural network optimization", token_budget=80,
    )
    _print_result(result_tight, "Greedy Builder (80 token budget)")

    # 3. Greedy builder with max_chunks limit
    print("\n" + "#" * 60)
    print("# Scenario 3: Greedy builder, max 3 chunks")
    print("#" * 60)

    limited_greedy = GreedyContextBuilder(
        token_counter=counter,
        config=BuilderConfig(token_budget=500, deduplicate=True, max_chunks=3),
    )
    result_limited = limited_greedy.build(
        candidates, query="neural network optimization", token_budget=500,
    )
    _print_result(result_limited, "Greedy Builder (max 3 chunks)")

    # 4. Greedy builder with dedup disabled (shows the duplicate getting through)
    print("\n" + "#" * 60)
    print("# Scenario 4: Greedy builder, dedup disabled")
    print("#" * 60)

    no_dedup_greedy = GreedyContextBuilder(
        token_counter=counter,
        config=BuilderConfig(token_budget=500, deduplicate=False),
    )
    result_no_dedup = no_dedup_greedy.build(
        candidates, query="neural network optimization", token_budget=500,
    )
    _print_result(result_no_dedup, "Greedy Builder (dedup disabled)")

    # 5. Show ContextPack tracing fields
    print("\n" + "#" * 60)
    print("# ContextPack tracing")
    print("#" * 60)
    final_pack = result.context_pack
    print(f"\nChunk IDs in pack : {final_pack.chunk_ids}")
    print(f"Source IDs in pack: {final_pack.source_ids}")
    print(f"Schema version    : {final_pack.schema_version}")


if __name__ == "__main__":
    main()
