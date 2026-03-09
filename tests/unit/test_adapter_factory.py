"""Tests for adapter factory functions."""

from __future__ import annotations

from libs.adapters.config import (
    OpenAIConfig,
    OpenSearchConfig,
    OtelConfig,
    QdrantConfig,
    TeiConfig,
    VllmConfig,
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
from libs.adapters.openai import OpenAIGenerator
from libs.adapters.opensearch import OpenSearchLexicalStore, OpenSearchVectorStore
from libs.adapters.otel import OtelSpanExporter
from libs.adapters.qdrant import QdrantVectorStore
from libs.adapters.tei import TeiCrossEncoderReranker, TeiEmbeddingProvider, TeiQueryEmbedder
from libs.adapters.vllm import VllmGenerator
from libs.generation.mock_generator import MockGenerator
from libs.observability.collector import NoOpCollector, SpanCollector
from libs.retrieval.stores.memory_lexical_store import MemoryLexicalStore
from libs.retrieval.stores.memory_vector_store import MemoryVectorStore
from libs.retrieval.stores.protocols import LexicalStore, VectorStore


class TestCreateVectorStore:
    def test_no_config_returns_memory(self) -> None:
        store = create_vector_store()
        assert isinstance(store, MemoryVectorStore)
        assert isinstance(store, VectorStore)

    def test_qdrant_config_returns_qdrant(self) -> None:
        store = create_vector_store(QdrantConfig())
        assert isinstance(store, QdrantVectorStore)
        assert isinstance(store, VectorStore)

    def test_opensearch_config_returns_opensearch(self) -> None:
        store = create_vector_store(OpenSearchConfig())
        assert isinstance(store, OpenSearchVectorStore)
        assert isinstance(store, VectorStore)


class TestCreateLexicalStore:
    def test_no_config_returns_memory(self) -> None:
        store = create_lexical_store()
        assert isinstance(store, MemoryLexicalStore)
        assert isinstance(store, LexicalStore)

    def test_opensearch_config_returns_opensearch(self) -> None:
        store = create_lexical_store(OpenSearchConfig())
        assert isinstance(store, OpenSearchLexicalStore)
        assert isinstance(store, LexicalStore)


class TestCreateEmbeddingProvider:
    def test_no_config_returns_mock(self) -> None:
        provider = create_embedding_provider()
        assert hasattr(provider, "embed_chunks")
        assert hasattr(provider, "model_info")

    def test_tei_config_returns_tei(self) -> None:
        provider = create_embedding_provider(TeiConfig())
        assert isinstance(provider, TeiEmbeddingProvider)


class TestCreateQueryEmbedder:
    def test_no_config_returns_mock(self) -> None:
        embedder = create_query_embedder()
        assert hasattr(embedder, "embed_query")

    def test_tei_config_returns_tei(self) -> None:
        embedder = create_query_embedder(TeiConfig())
        assert isinstance(embedder, TeiQueryEmbedder)


class TestCreateReranker:
    def test_no_config_returns_cross_encoder_stub(self) -> None:
        reranker = create_reranker()
        assert hasattr(reranker, "rerank")
        assert hasattr(reranker, "reranker_id")

    def test_tei_config_without_reranker_url_returns_stub(self) -> None:
        reranker = create_reranker(TeiConfig(), model_name="my-model")
        assert not isinstance(reranker, TeiCrossEncoderReranker)
        assert hasattr(reranker, "reranker_id")

    def test_tei_config_with_reranker_url_returns_tei(self) -> None:
        config = TeiConfig(reranker_url="http://localhost:8081")
        reranker = create_reranker(config, model_name="my-model")
        assert isinstance(reranker, TeiCrossEncoderReranker)


class TestCreateGenerator:
    def test_no_config_returns_mock(self) -> None:
        gen = create_generator()
        assert isinstance(gen, MockGenerator)

    def test_openai_config_returns_openai(self) -> None:
        gen = create_generator(OpenAIConfig(api_key="test-key"))
        assert isinstance(gen, OpenAIGenerator)

    def test_vllm_config_returns_vllm(self) -> None:
        gen = create_generator(VllmConfig())
        assert isinstance(gen, VllmGenerator)


class TestCreateSpanCollector:
    def test_no_config_returns_noop(self) -> None:
        collector = create_span_collector()
        assert isinstance(collector, NoOpCollector)
        assert isinstance(collector, SpanCollector)

    def test_otel_config_returns_otel(self) -> None:
        collector = create_span_collector(OtelConfig())
        assert isinstance(collector, OtelSpanExporter)
        assert isinstance(collector, SpanCollector)


class TestCreatePdfParser:
    def test_unstructured_default(self) -> None:
        from libs.adapters.unstructured import UnstructuredPdfParser

        parser = create_pdf_parser("unstructured")
        assert isinstance(parser, UnstructuredPdfParser)

    def test_tika(self) -> None:
        from libs.adapters.tika import TikaPdfParser

        parser = create_pdf_parser("tika")
        assert isinstance(parser, TikaPdfParser)

    def test_unknown_provider_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Unknown PDF parser provider"):
            create_pdf_parser("bogus")
