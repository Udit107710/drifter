"""Tests for adapter stubs: protocol compliance, lifecycle, NotImplementedError.

Adapters that have been implemented (Qdrant, OpenSearch, OTEL) are tested
via integration tests.  This file tests the remaining stubs.
"""

from __future__ import annotations

import pytest

from libs.adapters.config import (
    OpenAIConfig,
    OpenSearchConfig,
    OtelConfig,
    QdrantConfig,
    RagasConfig,
    TeiConfig,
    TikaConfig,
    UnstructuredConfig,
    VllmConfig,
)
from libs.adapters.openai import OpenAIGenerator
from libs.adapters.opensearch import OpenSearchLexicalStore, OpenSearchVectorStore
from libs.adapters.otel import OtelSpanExporter
from libs.adapters.qdrant import QdrantVectorStore
from libs.adapters.ragas import RagasAnswerEvaluator
from libs.adapters.tei import TeiCrossEncoderReranker, TeiEmbeddingProvider, TeiQueryEmbedder
from libs.adapters.tika import TikaPdfParser
from libs.adapters.unstructured import UnstructuredPdfParser
from libs.adapters.vllm import VllmGenerator
from libs.observability.collector import SpanCollector
from libs.parsing.parsers.pdf import PdfParserBase
from libs.retrieval.stores.protocols import LexicalStore, VectorStore

# ── Protocol compliance ───────────────────────────────────────────────


class TestProtocolCompliance:
    def test_qdrant_is_vector_store(self) -> None:
        store = QdrantVectorStore(QdrantConfig())
        assert isinstance(store, VectorStore)

    def test_opensearch_vector_is_vector_store(self) -> None:
        store = OpenSearchVectorStore(OpenSearchConfig())
        assert isinstance(store, VectorStore)

    def test_opensearch_lexical_is_lexical_store(self) -> None:
        store = OpenSearchLexicalStore(OpenSearchConfig())
        assert isinstance(store, LexicalStore)

    def test_otel_is_span_collector(self) -> None:
        exporter = OtelSpanExporter(OtelConfig())
        assert isinstance(exporter, SpanCollector)

    def test_unstructured_is_pdf_parser(self) -> None:
        parser = UnstructuredPdfParser(UnstructuredConfig())
        assert isinstance(parser, PdfParserBase)

    def test_tika_is_pdf_parser(self) -> None:
        parser = TikaPdfParser(TikaConfig())
        assert isinstance(parser, PdfParserBase)


# ── Lifecycle for stubs (not yet implemented) ────────────────────────


class TestStubLifecycle:
    def test_tei_embedding_lifecycle(self) -> None:
        p = TeiEmbeddingProvider(TeiConfig())
        p.connect()
        assert p.health_check() is False
        p.close()

    def test_tei_query_embedder_lifecycle(self) -> None:
        e = TeiQueryEmbedder(TeiConfig())
        e.connect()
        assert e.health_check() is False
        e.close()

    def test_tei_cross_encoder_lifecycle(self) -> None:
        r = TeiCrossEncoderReranker(TeiConfig(), "model")
        r.connect()
        assert r.health_check() is False
        r.close()

    def test_vllm_lifecycle(self) -> None:
        g = VllmGenerator(VllmConfig())
        g.connect()
        assert g.health_check() is False
        g.close()

    def test_unstructured_lifecycle(self) -> None:
        p = UnstructuredPdfParser(UnstructuredConfig())
        p.connect()
        assert p.health_check() is False
        p.close()

    def test_tika_lifecycle(self) -> None:
        p = TikaPdfParser(TikaConfig())
        p.connect()
        assert p.health_check() is False
        p.close()

    def test_openai_lifecycle(self) -> None:
        g = OpenAIGenerator(OpenAIConfig(api_key="test-key"))
        g.connect()
        assert g.health_check() is True
        g.close()
        assert g.health_check() is False

    def test_ragas_lifecycle(self) -> None:
        e = RagasAnswerEvaluator(RagasConfig())
        e.connect()
        assert e.health_check() is False
        e.close()


# ── Lifecycle for implemented adapters (without real services) ───────


class TestImplementedAdapterNoService:
    """Verify that implemented adapters behave correctly when not connected."""

    def test_qdrant_health_check_without_connect(self) -> None:
        store = QdrantVectorStore(QdrantConfig())
        assert store.health_check() is False

    def test_opensearch_vector_health_check_without_connect(self) -> None:
        store = OpenSearchVectorStore(OpenSearchConfig())
        assert store.health_check() is False

    def test_opensearch_lexical_health_check_without_connect(self) -> None:
        store = OpenSearchLexicalStore(OpenSearchConfig())
        assert store.health_check() is False

    def test_otel_health_check_without_connect(self) -> None:
        e = OtelSpanExporter(OtelConfig())
        assert e.health_check() is False

    def test_qdrant_add_without_connect_raises(self) -> None:
        store = QdrantVectorStore(QdrantConfig())
        with pytest.raises(RuntimeError, match="not connected"):
            store.add(None, None)  # type: ignore[arg-type]

    def test_opensearch_lexical_add_without_connect_raises(self) -> None:
        store = OpenSearchLexicalStore(OpenSearchConfig())
        with pytest.raises(RuntimeError, match="not connected"):
            store.add(None)  # type: ignore[arg-type]

    def test_otel_collect_without_connect_raises(self) -> None:
        e = OtelSpanExporter(OtelConfig())
        with pytest.raises(RuntimeError, match="not connected"):
            e.collect(None)  # type: ignore[arg-type]


# ── NotImplementedError on stub data methods ─────────────────────────


class TestNotImplemented:
    def test_tei_embed_chunks_raises_without_connect(self) -> None:
        p = TeiEmbeddingProvider(TeiConfig())
        with pytest.raises(RuntimeError, match="not connected"):
            p.embed_chunks([])

    def test_tei_embed_query_raises_without_connect(self) -> None:
        e = TeiQueryEmbedder(TeiConfig())
        with pytest.raises(RuntimeError, match="not connected"):
            e.embed_query("test")

    def test_tei_rerank_raises_without_connect(self) -> None:
        r = TeiCrossEncoderReranker(TeiConfig(), "model")
        with pytest.raises(RuntimeError, match="not connected"):
            r.rerank([], None)  # type: ignore[arg-type]

    def test_vllm_generate_raises(self) -> None:
        g = VllmGenerator(VllmConfig())
        with pytest.raises(NotImplementedError):
            g.generate(None)  # type: ignore[arg-type]

    def test_unstructured_extract_raises(self) -> None:
        p = UnstructuredPdfParser(UnstructuredConfig())
        with pytest.raises(NotImplementedError):
            p._extract_blocks(b"")

    def test_tika_extract_raises(self) -> None:
        p = TikaPdfParser(TikaConfig())
        with pytest.raises(NotImplementedError):
            p._extract_blocks(b"")

    def test_ragas_evaluate_raises(self) -> None:
        e = RagasAnswerEvaluator(RagasConfig())
        with pytest.raises(NotImplementedError):
            e.evaluate("q", "a", ["ctx"])


# ── Store IDs and identifiers ─────────────────────────────────────────


class TestIdentifiers:
    def test_qdrant_store_id(self) -> None:
        store = QdrantVectorStore(QdrantConfig(collection_name="my_col"))
        assert store.store_id == "qdrant:my_col"

    def test_opensearch_vector_store_id(self) -> None:
        store = OpenSearchVectorStore(OpenSearchConfig(index_prefix="test"))
        assert store.store_id == "opensearch-vector:test"

    def test_opensearch_lexical_store_id(self) -> None:
        store = OpenSearchLexicalStore(OpenSearchConfig(index_prefix="test"))
        assert store.store_id == "opensearch-lexical:test"

    def test_tei_reranker_id(self) -> None:
        r = TeiCrossEncoderReranker(TeiConfig(), "my-model")
        assert r.reranker_id == "tei-cross-encoder:my-model"

    def test_openai_generator_id(self) -> None:
        g = OpenAIGenerator(OpenAIConfig(api_key="test-key", model_id="gpt-5-nano"))
        assert g.generator_id == "openai:gpt-5-nano"

    def test_openai_generate_without_connect_raises(self) -> None:
        g = OpenAIGenerator(OpenAIConfig(api_key="test-key"))
        with pytest.raises(RuntimeError, match="not connected"):
            g.generate(None)  # type: ignore[arg-type]

    def test_vllm_generator_id(self) -> None:
        g = VllmGenerator(VllmConfig(model_id="llama-2"))
        assert g.generator_id == "vllm:llama-2"
