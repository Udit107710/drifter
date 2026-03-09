"""Tests for adapter factory functions."""

from __future__ import annotations

from libs.adapters.config import (
    HuggingFaceConfig,
    OpenAIConfig,
    OpenRouterConfig,
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
from libs.adapters.openrouter import OpenRouterEmbeddingProvider, OpenRouterQueryEmbedder
from libs.adapters.opensearch import OpenSearchLexicalStore, OpenSearchVectorStore
from libs.adapters.otel import OtelSpanExporter
from libs.adapters.qdrant import QdrantVectorStore
from libs.adapters.tei import TeiCrossEncoderReranker, TeiEmbeddingProvider, TeiQueryEmbedder
from libs.adapters.vllm import VllmGenerator
from libs.embeddings.protocols import EmbeddingProvider
from libs.generation.mock_generator import MockGenerator
from libs.observability.collector import NoOpCollector, SpanCollector
from libs.reranking.protocols import Reranker
from libs.retrieval.broker.protocols import QueryEmbedder
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
        assert isinstance(provider, EmbeddingProvider)

    def test_tei_config_returns_tei(self) -> None:
        provider = create_embedding_provider(TeiConfig())
        assert isinstance(provider, TeiEmbeddingProvider)

    def test_openrouter_config_returns_openrouter(self) -> None:
        cfg = OpenRouterConfig(api_key="test", embedding_model="test/model")
        provider = create_embedding_provider(cfg)
        assert isinstance(provider, OpenRouterEmbeddingProvider)


class TestCreateQueryEmbedder:
    def test_no_config_returns_mock(self) -> None:
        embedder = create_query_embedder()
        assert isinstance(embedder, QueryEmbedder)

    def test_tei_config_returns_tei(self) -> None:
        embedder = create_query_embedder(TeiConfig())
        assert isinstance(embedder, TeiQueryEmbedder)

    def test_openrouter_config_returns_openrouter(self) -> None:
        cfg = OpenRouterConfig(api_key="test", embedding_model="test/model")
        embedder = create_query_embedder(cfg)
        assert isinstance(embedder, OpenRouterQueryEmbedder)


class TestCreateReranker:
    def test_no_config_returns_cross_encoder_stub(self) -> None:
        reranker = create_reranker()
        assert isinstance(reranker, Reranker)

    def test_tei_config_without_reranker_url_returns_stub(self) -> None:
        reranker = create_reranker(TeiConfig(), model_name="my-model")
        assert not isinstance(reranker, TeiCrossEncoderReranker)
        assert isinstance(reranker, Reranker)

    def test_tei_config_with_reranker_url_returns_tei(self) -> None:
        config = TeiConfig(reranker_url="http://localhost:8081")
        reranker = create_reranker(config, model_name="my-model")
        assert isinstance(reranker, TeiCrossEncoderReranker)

    def test_hf_config_returns_huggingface_reranker(self) -> None:
        from libs.adapters.huggingface import HuggingFaceReranker

        config = HuggingFaceConfig(api_key="test-key")
        reranker = create_reranker(config, model_name="my-model")
        assert isinstance(reranker, HuggingFaceReranker)
        assert reranker.reranker_id == "hf-reranker:my-model"


class TestCreateGenerator:
    def test_no_config_returns_mock(self) -> None:
        gen = create_generator()
        assert isinstance(gen, MockGenerator)

    def test_openai_config_returns_openai(self) -> None:
        gen = create_generator(OpenAIConfig(api_key="test-key"))
        assert isinstance(gen, OpenAIGenerator)
        assert gen.generator_id == "openai:gpt-4o"

    def test_openrouter_config_returns_openai_generator(self) -> None:
        gen = create_generator(
            OpenRouterConfig(api_key="test-key", model_id="meta/llama-3-70b")
        )
        assert isinstance(gen, OpenAIGenerator)
        assert gen.generator_id == "openrouter:meta/llama-3-70b"

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
