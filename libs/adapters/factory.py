"""Factory functions for creating adapter instances.

Each factory returns an in-memory / mock implementation when no config is
provided, and the real adapter stub when a config is given.  This keeps
the application wiring simple: call the factory, get a ready-to-use object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from libs.adapters.config import (
    GeminiConfig,
    LangfuseConfig,
    OpenAIConfig,
    OpenSearchConfig,
    OtelConfig,
    QdrantConfig,
    TeiConfig,
    TikaConfig,
    UnstructuredConfig,
    VllmConfig,
)

if TYPE_CHECKING:
    from libs.generation.protocols import Generator
    from libs.observability.collector import SpanCollector
    from libs.parsing.parsers.pdf import PdfParserBase
    from libs.retrieval.stores.protocols import LexicalStore, VectorStore


def create_vector_store(config: QdrantConfig | OpenSearchConfig | None = None) -> VectorStore:
    """Create a vector store instance.

    Returns :class:`MemoryVectorStore` when *config* is ``None``,
    :class:`QdrantVectorStore` for a :class:`QdrantConfig`, or
    :class:`OpenSearchVectorStore` for an :class:`OpenSearchConfig`.
    """
    if config is None:
        from libs.retrieval.stores.memory_vector_store import MemoryVectorStore

        return MemoryVectorStore()

    if isinstance(config, QdrantConfig):
        from libs.adapters.qdrant import QdrantVectorStore

        return QdrantVectorStore(config)

    if isinstance(config, OpenSearchConfig):
        from libs.adapters.opensearch import OpenSearchVectorStore

        return OpenSearchVectorStore(config)

    raise TypeError(f"Unsupported vector store config type: {type(config)}")


def create_lexical_store(config: OpenSearchConfig | None = None) -> LexicalStore:
    """Create a lexical store instance.

    Returns :class:`MemoryLexicalStore` when *config* is ``None``,
    or :class:`OpenSearchLexicalStore` for an :class:`OpenSearchConfig`.
    """
    if config is None:
        from libs.retrieval.stores.memory_lexical_store import MemoryLexicalStore

        return MemoryLexicalStore()

    from libs.adapters.opensearch import OpenSearchLexicalStore

    return OpenSearchLexicalStore(config)


def create_embedding_provider(config: TeiConfig | None = None) -> object:
    """Create an embedding provider instance.

    Returns :class:`DeterministicEmbeddingProvider` when *config* is
    ``None``, or :class:`TeiEmbeddingProvider` for a :class:`TeiConfig`.
    """
    if config is None:
        from libs.embeddings.mock_provider import DeterministicEmbeddingProvider

        return DeterministicEmbeddingProvider()

    from libs.adapters.tei import TeiEmbeddingProvider

    return TeiEmbeddingProvider(config)


def create_query_embedder(config: TeiConfig | None = None) -> object:
    """Create a query embedder instance.

    Returns a simple lambda-based embedder when *config* is ``None``,
    or :class:`TeiQueryEmbedder` for a :class:`TeiConfig`.
    """
    if config is None:
        from libs.embeddings.mock_provider import DeterministicEmbeddingProvider

        # Wrap the mock provider to satisfy QueryEmbedder protocol
        provider = DeterministicEmbeddingProvider()

        class _MockQueryEmbedder:
            def embed_query(self, text: str) -> list[float]:
                info = provider.model_info()
                return [hash(text) % 100 / 100.0] * info.dimensions

        return _MockQueryEmbedder()

    from libs.adapters.tei import TeiQueryEmbedder

    return TeiQueryEmbedder(config)


def create_reranker(config: TeiConfig | None = None, model_name: str = "cross-encoder") -> object:
    """Create a reranker instance.

    Returns a :class:`CrossEncoderReranker` placeholder when *config* is
    ``None`` or has no ``reranker_url``, or :class:`TeiCrossEncoderReranker`
    for a :class:`TeiConfig` with a reranker URL.
    """
    if config is None or not config.reranker_url:
        from libs.reranking.cross_encoder_stub import CrossEncoderReranker

        return CrossEncoderReranker(model_name)

    from libs.adapters.tei import TeiCrossEncoderReranker

    # Create a config pointing at the reranker endpoint
    reranker_config = TeiConfig(
        base_url=config.reranker_url,
        model_id=config.reranker_model_id or config.model_id,
        model_version=config.model_version,
        timeout_s=config.timeout_s,
        max_batch_size=config.max_batch_size,
    )
    return TeiCrossEncoderReranker(reranker_config, model_name)


def create_generator(
    config: VllmConfig | GeminiConfig | OpenAIConfig | None = None,
) -> Generator:
    """Create a generator instance.

    Returns :class:`MockGenerator` when *config* is ``None``,
    :class:`OpenAIGenerator` for an :class:`OpenAIConfig`,
    :class:`GeminiGenerator` for a :class:`GeminiConfig`, or
    :class:`VllmGenerator` for a :class:`VllmConfig`.
    """
    if config is None:
        from libs.generation.mock_generator import MockGenerator

        return MockGenerator()

    if isinstance(config, OpenAIConfig):
        from libs.adapters.openai import OpenAIGenerator

        return OpenAIGenerator(config)

    if isinstance(config, GeminiConfig):
        from libs.adapters.gemini import GeminiGenerator

        return GeminiGenerator(config)

    if isinstance(config, VllmConfig):
        from libs.adapters.vllm import VllmGenerator

        return VllmGenerator(config)

    raise TypeError(f"Unsupported generator config type: {type(config)}")


def create_span_collector(config: OtelConfig | LangfuseConfig | None = None) -> SpanCollector:
    """Create a span collector instance.

    Returns :class:`NoOpCollector` when *config* is ``None``,
    :class:`LangfuseSpanExporter` for a :class:`LangfuseConfig`,
    or :class:`OtelSpanExporter` for an :class:`OtelConfig`.
    """
    if config is None:
        from libs.observability.collector import NoOpCollector

        return NoOpCollector()

    if isinstance(config, LangfuseConfig):
        from libs.adapters.langfuse import LangfuseSpanExporter

        return LangfuseSpanExporter(config)

    if isinstance(config, OtelConfig):
        from libs.adapters.otel import OtelSpanExporter

        return OtelSpanExporter(config)

    raise TypeError(f"Unsupported span collector config type: {type(config)}")


def create_pdf_parser(
    provider: str = "unstructured",
    config: UnstructuredConfig | TikaConfig | None = None,
) -> PdfParserBase:
    """Create a PDF parser instance.

    Args:
        provider: Parser backend — ``"unstructured"`` or ``"tika"``.
        config: Provider-specific configuration.  Required for real
                adapters; ignored when no matching config is given.

    Raises:
        ValueError: If *provider* is not recognised.
        TypeError: If *config* type does not match *provider*.
    """
    if provider == "unstructured":
        if config is None:
            config = UnstructuredConfig()
        if not isinstance(config, UnstructuredConfig):
            raise TypeError(f"Expected UnstructuredConfig, got {type(config)}")
        from libs.adapters.unstructured import UnstructuredPdfParser

        return UnstructuredPdfParser(config)

    if provider == "tika":
        if config is None:
            config = TikaConfig()
        if not isinstance(config, TikaConfig):
            raise TypeError(f"Expected TikaConfig, got {type(config)}")
        from libs.adapters.tika import TikaPdfParser

        return TikaPdfParser(config)

    raise ValueError(f"Unknown PDF parser provider: {provider!r}")
