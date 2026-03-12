"""Integration wiring sample — demonstrates adapter factory usage.

Runs with zero external services by using in-memory/mock fallbacks.

Usage::

    uv run python examples/integration_wiring_sample.py
"""

from __future__ import annotations

from libs.adapters.env import (
    load_ollama_config,
    load_opensearch_config,
    load_otel_config,
    load_qdrant_config,
    load_tei_config,
)
from libs.adapters.factory import (
    create_embedding_provider,
    create_generator,
    create_lexical_store,
    create_pdf_parser,
    create_query_embedder,
    create_reranker,
    create_span_collector,
    create_vector_store,
)


def main() -> None:
    # Load configs from env — returns None if env vars not set
    qdrant_cfg = load_qdrant_config()
    opensearch_cfg = load_opensearch_config()
    tei_cfg = load_tei_config()
    ollama_cfg = load_ollama_config()
    otel_cfg = load_otel_config()

    # Create adapters — falls back to in-memory/mock when config is None
    vector_store = create_vector_store(qdrant_cfg)
    lexical_store = create_lexical_store(opensearch_cfg)
    embedding_provider = create_embedding_provider(tei_cfg)
    query_embedder = create_query_embedder(tei_cfg)
    reranker = create_reranker(tei_cfg)
    generator = create_generator(ollama_cfg)
    span_collector = create_span_collector(otel_cfg)

    print("Adapter wiring complete:")
    print(f"  vector_store:       {type(vector_store).__name__}")
    print(f"  lexical_store:      {type(lexical_store).__name__}")
    print(f"  embedding_provider: {type(embedding_provider).__name__}")
    print(f"  query_embedder:     {type(query_embedder).__name__}")
    print(f"  reranker:           {type(reranker).__name__}")
    print(f"  generator:          {type(generator).__name__}")
    print(f"  span_collector:     {type(span_collector).__name__}")

    # PDF parser — always returns a stub (no in-memory fallback)
    pdf_parser = create_pdf_parser("unstructured")
    print(f"  pdf_parser:         {type(pdf_parser).__name__}")

    print("\nAll adapters created successfully with zero external services.")


if __name__ == "__main__":
    main()
