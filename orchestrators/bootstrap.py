"""Service registry and composition root.

Creates all library services from environment configuration and adapter factories.
The registry is the single place where concrete implementations are chosen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from libs.adapters.env import (
    load_gemini_config,
    load_huggingface_config,
    load_langfuse_config,
    load_openai_config,
    load_openrouter_config,
    load_opensearch_config,
    load_otel_config,
    load_qdrant_config,
    load_tei_config,
    load_vllm_config,
)
from libs.adapters.factory import (
    create_embedding_provider,
    create_generator,
    create_lexical_store,
    create_query_embedder,
    create_reranker,
    create_span_collector,
    create_vector_store,
)
from libs.adapters.memory.chunk_repository import MemoryChunkRepository
from libs.adapters.memory.crawl_state_repository import MemoryCrawlStateRepository
from libs.adapters.memory.embedding_repository import MemoryEmbeddingRepository
from libs.adapters.memory.source_repository import MemorySourceRepository
from libs.adapters.store_writers import LexicalStoreWriter, VectorStoreWriter
from libs.chunking.strategies.recursive import RecursiveStructureChunker
from libs.chunking.token_counter import WhitespaceTokenCounter
from libs.context_builder.greedy_builder import GreedyContextBuilder
from libs.context_builder.service import ContextBuilderService
from libs.evaluation.evaluator import RetrievalEvaluator
from libs.experiments.runner import ExperimentRunner
from libs.experiments.store import InMemoryExperimentStore
from libs.generation.citation_validator import DefaultCitationValidator
from libs.generation.request_builder import GenerationRequestBuilder
from libs.generation.service import GenerationService
from libs.indexing.service import IndexingService
from libs.ingestion.change_detector import detect
from libs.ingestion.connectors.filesystem import LocalFilesystemConnector
from libs.ingestion.service import IngestionService
from libs.observability.tracer import Tracer
from libs.parsing.parsers.markdown import MarkdownParser
from libs.parsing.parsers.plain_text import PlainTextParser
from libs.reranking.feature_reranker import FeatureBasedReranker
from libs.reranking.service import RerankerService
from libs.retrieval.broker.models import BrokerConfig
from libs.retrieval.broker.service import RetrievalBroker
from orchestrators.ingestion import IngestionOrchestrator

# Fields that must not be overridden via --config for security.
_SECRET_FIELDS = frozenset({
    "api_key",
    "password",
    "auth",
    "secret",
})


@dataclass
class ServiceRegistry:
    """Holds all wired services for the application layer."""

    tracer: Tracer
    retrieval_broker: RetrievalBroker
    reranker_service: RerankerService
    context_builder_service: ContextBuilderService
    generation_service: GenerationService
    indexing_service: IndexingService | None
    ingestion_orchestrator: IngestionOrchestrator | None
    source_repo: Any
    evaluator: RetrievalEvaluator
    experiment_runner: ExperimentRunner
    token_budget: int


def _reject_secret_overrides(overrides: dict[str, Any]) -> None:
    """Raise ValueError if any override key touches a secret field."""
    for key in overrides:
        leaf = key.rsplit(".", 1)[-1] if "." in key else key
        if leaf.lower() in _SECRET_FIELDS:
            raise ValueError(
                f"Cannot override secret field {key!r} via --config. "
                "Use environment variables instead."
            )


def create_registry(overrides: dict[str, Any] | None = None) -> ServiceRegistry:
    """Composition root: load configs, create adapters, wire services.

    1. Load configs from DRIFTER_* env vars.
    2. Apply overrides (reject secret fields).
    3. Call adapter factories.
    4. Construct library services.
    5. Return ServiceRegistry.
    """
    overrides = overrides or {}
    _reject_secret_overrides(overrides)

    # --- Configs from environment ---
    qdrant_config = load_qdrant_config()
    opensearch_config = load_opensearch_config()
    tei_config = load_tei_config()
    hf_config = load_huggingface_config()
    vllm_config = load_vllm_config()
    openrouter_config = load_openrouter_config()
    openai_config = load_openai_config()
    gemini_config = load_gemini_config()
    otel_config = load_otel_config()
    langfuse_config = load_langfuse_config()

    # --- Token budget (overridable) ---
    token_budget = int(overrides.get("token_budget", 5000))

    # --- Observability (prefer Langfuse over OTel) ---
    collector_config = langfuse_config or otel_config
    collector = create_span_collector(collector_config)
    tracer = Tracer(collector=collector)

    # --- Observability: connect if real ---
    if hasattr(collector, "connect"):
        collector.connect()

    # --- Retrieval stores ---
    vector_store = create_vector_store(qdrant_config)
    lexical_store = create_lexical_store(opensearch_config)

    # Connect real stores
    if hasattr(vector_store, "connect"):
        vector_store.connect()
    if hasattr(lexical_store, "connect"):
        lexical_store.connect()

    # --- Embeddings (prefer OpenRouter if embedding_model set, else TEI) ---
    embedding_config: Any = None
    if openrouter_config and openrouter_config.embedding_model:
        embedding_config = openrouter_config
    elif tei_config:
        embedding_config = tei_config
    query_embedder: Any = create_query_embedder(embedding_config)
    if hasattr(query_embedder, "connect"):
        query_embedder.connect()

    # --- Retrieval broker ---
    broker_config = BrokerConfig(
        lexical_weight=float(overrides.get("lexical_weight", 1.5)),
    )
    retrieval_broker = RetrievalBroker(
        vector_store=vector_store,
        lexical_store=lexical_store,
        query_embedder=query_embedder,
        config=broker_config,
    )

    # --- Reranking (TEI > HuggingFace > FeatureBasedReranker) ---
    reranker: Any = None

    # Try TEI cross-encoder first
    tei_reranker = create_reranker(tei_config)
    if hasattr(tei_reranker, "connect"):
        tei_reranker.connect()
    if hasattr(tei_reranker, "health_check") and tei_reranker.health_check():
        reranker = tei_reranker

    # Fall back to HuggingFace Inference API
    if reranker is None and hf_config is not None:
        hf_reranker = create_reranker(hf_config)
        if hasattr(hf_reranker, "connect"):
            hf_reranker.connect()
        if hasattr(hf_reranker, "health_check") and hf_reranker.health_check():
            reranker = hf_reranker

    # Final fallback: feature-based
    if reranker is None:
        reranker = FeatureBasedReranker()
    top_n = int(overrides.get("reranker_top_n", 0))
    reranker_service = RerankerService(reranker=reranker, top_n=top_n)

    # --- Context builder ---
    counter = WhitespaceTokenCounter()
    builder = GreedyContextBuilder(token_counter=counter)
    context_builder_service = ContextBuilderService(builder=builder)

    # --- Generation (prefer OpenRouter > OpenAI > Gemini > vLLM) ---
    generator_config = (
        openrouter_config or openai_config or gemini_config or vllm_config
    )
    generator = create_generator(generator_config)
    if hasattr(generator, "connect"):
        generator.connect()
    request_builder = GenerationRequestBuilder()
    citation_validator = DefaultCitationValidator()
    generation_service = GenerationService(
        generator=generator,
        request_builder=request_builder,
        citation_validator=citation_validator,
    )

    # --- Indexing ---
    embedding_provider = create_embedding_provider(embedding_config)
    if hasattr(embedding_provider, "connect"):
        embedding_provider.connect()
    chunk_repo = MemoryChunkRepository()
    embedding_repo = MemoryEmbeddingRepository()
    vector_writer = VectorStoreWriter(vector_store)
    lexical_writer = LexicalStoreWriter(lexical_store)

    indexing_service = IndexingService(
        embedding_provider=embedding_provider,
        chunk_repo=chunk_repo,
        embedding_repo=embedding_repo,
        vector_writer=vector_writer,
        lexical_writer=lexical_writer,
    )

    # --- Ingestion orchestrator ---
    source_repo = MemorySourceRepository()
    crawl_state_repo = MemoryCrawlStateRepository()
    connector = LocalFilesystemConnector()

    ingestion_service = IngestionService(
        source_repo=source_repo,
        crawl_state_repo=crawl_state_repo,
        connector=connector,
        change_detector=detect,
    )

    parser_registry = {
        "text/markdown": MarkdownParser(),
        "text/x-markdown": MarkdownParser(),
        "text/plain": PlainTextParser(),
    }
    chunking_strategy = RecursiveStructureChunker()

    ingestion_orchestrator = IngestionOrchestrator(
        tracer=tracer,
        ingestion_service=ingestion_service,
        parser_registry=parser_registry,
        chunking_strategy=chunking_strategy,
        indexing_service=indexing_service,
    )

    # --- Evaluation ---
    evaluator = RetrievalEvaluator()
    experiment_store = InMemoryExperimentStore()
    experiment_runner = ExperimentRunner(store=experiment_store)

    return ServiceRegistry(
        tracer=tracer,
        retrieval_broker=retrieval_broker,
        reranker_service=reranker_service,
        context_builder_service=context_builder_service,
        generation_service=generation_service,
        indexing_service=indexing_service,
        ingestion_orchestrator=ingestion_orchestrator,
        source_repo=source_repo,
        evaluator=evaluator,
        experiment_runner=experiment_runner,
        token_budget=token_budget,
    )
