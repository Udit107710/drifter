"""Demo: evaluation subsystem with mock retriever, metrics, and reports."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from libs.contracts.generation import Citation, GeneratedAnswer, TokenUsage
from libs.evaluation import (
    EvaluationConfig,
    RetrievalEvaluator,
    create_seed_dataset,
    detect_unsupported_claims,
    evaluate_citation_accuracy,
    evaluate_faithfulness,
    generate_markdown_summary,
    load_dataset,
    save_dataset,
    save_json_report,
)


class MockRetriever:
    """A retriever pre-loaded with canned results for evaluation.

    Returns a mix of relevant and irrelevant chunks so metrics
    are interesting (not all 0 or all 1).
    """

    def __init__(self, results: dict[str, list[str]]) -> None:
        self._results = results

    def retrieve(self, query: str, k: int) -> list[str]:
        return self._results.get(query, [])[:k]


def main() -> None:
    # ---------------------------------------------------------------
    # 1. Create seed dataset
    # ---------------------------------------------------------------
    print("=" * 60)
    print("1. Creating seed dataset")
    print("=" * 60)

    cases = create_seed_dataset()
    for case in cases:
        print(f"  {case.case_id}: {case.query}")
        print(f"    relevant: {case.relevant_chunk_ids}")
    print()

    # ---------------------------------------------------------------
    # 2. Set up mock retriever with partial overlap
    # ---------------------------------------------------------------
    print("=" * 60)
    print("2. Setting up mock retriever")
    print("=" * 60)

    # Each query returns some relevant chunks mixed with irrelevant ones.
    # This produces metrics that are neither 0 nor 1.
    mock_results: dict[str, list[str]] = {
        # seed-001: relevant = [chunk-ml-001, chunk-ml-002]
        # Returns 1 relevant at rank 1, 1 irrelevant, then 1 relevant at rank 3
        "What is machine learning?": [
            "chunk-ml-001",      # relevant
            "chunk-noise-01",    # irrelevant
            "chunk-ml-002",      # relevant
            "chunk-noise-02",    # irrelevant
            "chunk-noise-03",    # irrelevant
        ],
        # seed-002: relevant = [chunk-bp-001, chunk-bp-002, chunk-bp-003]
        # Returns 2 of 3 relevant, misses chunk-bp-003
        "How does backpropagation work?": [
            "chunk-noise-04",    # irrelevant
            "chunk-bp-001",      # relevant
            "chunk-bp-002",      # relevant
            "chunk-noise-05",    # irrelevant
            "chunk-noise-06",    # irrelevant
        ],
        # seed-003: relevant = [chunk-vdb-001]
        # Returns the relevant chunk at rank 2
        "What is a vector database?": [
            "chunk-noise-07",    # irrelevant
            "chunk-vdb-001",     # relevant
            "chunk-noise-08",    # irrelevant
            "chunk-noise-09",    # irrelevant
            "chunk-noise-10",    # irrelevant
        ],
        # seed-004: relevant = [chunk-tf-001, chunk-tf-002]
        # Returns 1 of 2 relevant, buried at rank 4
        "Explain the transformer architecture": [
            "chunk-noise-11",    # irrelevant
            "chunk-noise-12",    # irrelevant
            "chunk-noise-13",    # irrelevant
            "chunk-tf-001",      # relevant
            "chunk-noise-14",    # irrelevant
        ],
        # seed-005: relevant = [chunk-rag-001, chunk-rag-002, chunk-rag-003]
        # Returns all 3 relevant in top 5 but scattered
        "What is retrieval augmented generation?": [
            "chunk-rag-001",     # relevant
            "chunk-noise-15",    # irrelevant
            "chunk-rag-002",     # relevant
            "chunk-rag-003",     # relevant
            "chunk-noise-16",    # irrelevant
        ],
    }

    retriever = MockRetriever(mock_results)
    print("  Mock retriever configured with partially overlapping results")
    print()

    # ---------------------------------------------------------------
    # 3. Run RetrievalEvaluator
    # ---------------------------------------------------------------
    print("=" * 60)
    print("3. Running RetrievalEvaluator")
    print("=" * 60)

    config = EvaluationConfig(
        retrieval_mode="mock",
        embedding_model="none",
        chunking_strategy="fixed_window",
        k_values=[3, 5],
    )
    evaluator = RetrievalEvaluator(config=config, run_id="eval-demo-001")
    report = evaluator.evaluate(cases, retriever, k_values=[3, 5])

    print(f"  Run ID: {report.run_id}")
    print(f"  Dataset size: {report.dataset_size}")
    print(f"  Queries evaluated: {len(report.query_results)}")
    for sm in report.stage_metrics:
        print(f"\n  Stage: {sm.stage}")
        for key in sorted(sm.metric_means):
            print(f"    {key}: mean={sm.metric_means[key]:.4f}  "
                  f"median={sm.metric_medians.get(key, 0.0):.4f}")
    print()

    # ---------------------------------------------------------------
    # 4. Generate JSON and Markdown reports
    # ---------------------------------------------------------------
    print("=" * 60)
    print("4. Generating reports")
    print("=" * 60)

    # JSON report to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
    ) as tmp_json:
        json_path = Path(tmp_json.name)
    save_json_report(report, json_path)
    print(f"  JSON report saved to: {json_path}")

    # Verify JSON is valid
    json_data = json.loads(json_path.read_text())
    print(f"  JSON keys: {list(json_data.keys())}")

    # Markdown report to stdout
    print()
    md = generate_markdown_summary(report)
    print(md)

    # Clean up JSON temp file
    json_path.unlink()

    # ---------------------------------------------------------------
    # 5. Answer quality metrics
    # ---------------------------------------------------------------
    print("=" * 60)
    print("5. Computing answer metrics")
    print("=" * 60)

    # Simulate a generated answer with citations
    context_chunk_ids = {"chunk-ml-001", "chunk-ml-002", "chunk-noise-01"}

    answer = GeneratedAnswer(
        answer="Machine learning enables systems to learn from data. It is a subset of AI.",
        citations=[
            Citation(
                claim="Machine learning enables systems to learn from data",
                chunk_id="chunk-ml-001",
                chunk_content="ML is a field...",
                source_id="src-001",
                confidence=0.95,
            ),
            Citation(
                claim="It is a subset of AI",
                chunk_id="chunk-ml-002",
                chunk_content="ML is a subset...",
                source_id="src-001",
                confidence=0.90,
            ),
            Citation(
                claim="Neural networks are the backbone",
                chunk_id="chunk-nn-999",  # NOT in context -- fabricated
                chunk_content="Neural nets...",
                source_id="src-002",
                confidence=0.70,
            ),
        ],
        model_id="mock-llm",
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        trace_id="trace-demo-001",
    )

    # Citation accuracy
    accuracy = evaluate_citation_accuracy(answer, context_chunk_ids)
    print(f"\n  Citation accuracy: {accuracy.score:.2%}")
    print(f"    Total citations: {accuracy.total_citations}")
    print(f"    Valid citations: {accuracy.valid_citations}")
    if accuracy.invalid_citations:
        print(f"    Invalid citations: {accuracy.invalid_citations}")

    # Faithfulness
    claims = [
        "Machine learning enables systems to learn from data",
        "It is a subset of AI",
        "Deep learning requires GPUs",  # unsupported
    ]
    claim_to_chunk = {
        "Machine learning enables systems to learn from data": "chunk-ml-001",
        "It is a subset of AI": "chunk-ml-002",
        "Deep learning requires GPUs": "chunk-gpu-999",  # not in context
    }
    faithfulness = evaluate_faithfulness(claims, context_chunk_ids, claim_to_chunk)
    print(f"\n  Faithfulness: {faithfulness.score:.2%}")
    print(f"    Supported: {faithfulness.supported_claims}/{faithfulness.total_claims}")
    if faithfulness.unsupported_claims:
        print(f"    Unsupported: {faithfulness.unsupported_claims}")

    # Unsupported claim detection
    unsupported = detect_unsupported_claims(answer, context_chunk_ids)
    print(f"\n  Unsupported claims detected: {unsupported}")
    print()

    # ---------------------------------------------------------------
    # 6. Dataset save/load roundtrip
    # ---------------------------------------------------------------
    print("=" * 60)
    print("6. Dataset save/load roundtrip")
    print("=" * 60)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
    ) as tmp_ds:
        ds_path = Path(tmp_ds.name)

    save_dataset(cases, ds_path)
    print(f"  Saved {len(cases)} cases to: {ds_path}")

    loaded = load_dataset(ds_path)
    print(f"  Loaded {len(loaded)} cases from: {ds_path}")

    # Verify roundtrip
    for original, restored in zip(cases, loaded, strict=False):
        assert original.case_id == restored.case_id, (
            f"Mismatch: {original.case_id} != {restored.case_id}"
        )
        assert original.query == restored.query
        assert original.relevant_chunk_ids == restored.relevant_chunk_ids

    print("  Roundtrip verification passed: all cases match")

    # Clean up
    ds_path.unlink()
    print("  Temp file cleaned up")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
