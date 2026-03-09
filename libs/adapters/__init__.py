"""Integration adapters.

Concrete implementations of subsystem protocols for external services:

- **Qdrant** — ``VectorStore`` (dense retrieval)
- **OpenSearch** — ``VectorStore`` (k-NN) and ``LexicalStore`` (BM25)
- **TEI** — ``EmbeddingProvider``, ``QueryEmbedder``, ``Reranker``
- **OpenAI** — ``Generator``
- **vLLM** — ``Generator``
- **Unstructured / Tika** — ``PdfParserBase``
- **Ragas** — answer evaluation (supplementary, no existing protocol)
- **OpenTelemetry** — ``SpanCollector``

The ``memory/`` sub-package provides in-memory implementations for
deterministic local testing.

Use the factory functions in :mod:`libs.adapters.factory` to obtain
adapter instances with automatic fallback to in-memory/mock defaults.
"""

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

__all__ = [
    "create_embedding_provider",
    "create_generator",
    "create_lexical_store",
    "create_pdf_parser",
    "create_query_embedder",
    "create_reranker",
    "create_span_collector",
    "create_vector_store",
]
