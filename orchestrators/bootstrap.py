"""Service registry and composition root.

Creates all library services from environment configuration and adapter factories.
The registry is the single place where concrete implementations are chosen.
"""

from __future__ import annotations

import logging
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
from libs.adapters.protocols import Connectable, HealthCheckable
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
from libs.resilience import RetryConfig
from libs.retrieval.broker.async_service import AsyncRetrievalBroker
from libs.retrieval.broker.models import BrokerConfig
from libs.retrieval.broker.service import RetrievalBroker
from libs.retrieval.stores.async_memory_lexical_store import (
    AsyncMemoryLexicalStore,
)
from libs.retrieval.stores.async_memory_vector_store import (
    AsyncMemoryVectorStore,
)
from orchestrators.ingestion import IngestionOrchestrator

_logger = logging.getLogger(__name__)

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
    async_retrieval_broker: AsyncRetrievalBroker | None
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
    if isinstance(collector, Connectable):
        collector.connect()

    # --- Retrieval stores ---
    vector_store = create_vector_store(qdrant_config)
    lexical_store = create_lexical_store(opensearch_config)

    # Connect real stores
    if isinstance(vector_store, Connectable):
        vector_store.connect()
    if isinstance(lexical_store, Connectable):
        lexical_store.connect()

    # --- Embeddings (prefer OpenRouter if embedding_model set, else TEI) ---
    embedding_config = None
    if openrouter_config and openrouter_config.embedding_model:
        embedding_config = openrouter_config
    elif tei_config:
        embedding_config = tei_config
    query_embedder = create_query_embedder(embedding_config)
    if isinstance(query_embedder, Connectable):
        query_embedder.connect()

    # --- Retry config (optional) ---
    retry_max = int(overrides.get("retry.max_retries", 0))
    retry_config: RetryConfig | None = None
    if retry_max > 0:
        retry_config = RetryConfig(
            max_retries=retry_max,
            base_delay_s=float(overrides.get("retry.base_delay_s", 0.5)),
            max_delay_s=float(overrides.get("retry.max_delay_s", 30.0)),
            jitter_factor=float(
                overrides.get("retry.jitter_factor", 0.5),
            ),
        )

    # --- Retrieval broker ---
    broker_config = BrokerConfig(
        lexical_weight=float(overrides.get("lexical_weight", 1.5)),
    )
    retrieval_broker = RetrievalBroker(
        vector_store=vector_store,
        lexical_store=lexical_store,
        query_embedder=query_embedder,
        config=broker_config,
        retry_config=retry_config,
    )

    # --- Async retrieval broker (wraps same stores) ---
    async_vector = AsyncMemoryVectorStore(vector_store)
    async_lexical = AsyncMemoryLexicalStore(lexical_store)

    class _SyncToAsyncEmbedder:
        """Wraps a sync QueryEmbedder as AsyncQueryEmbedder."""

        def __init__(self, inner: object) -> None:
            self._inner = inner

        async def async_embed_query(self, text: str) -> list[float]:
            return self._inner.embed_query(text)  # type: ignore[union-attr]

    async_retrieval_broker = AsyncRetrievalBroker(
        vector_store=async_vector,
        lexical_store=async_lexical,
        query_embedder=_SyncToAsyncEmbedder(query_embedder),
        config=broker_config,
        retry_config=retry_config,
    )

    # --- Reranking (TEI > HuggingFace > FeatureBasedReranker) ---
    reranker = None

    # Try TEI cross-encoder first
    tei_reranker = create_reranker(tei_config)
    if isinstance(tei_reranker, Connectable):
        tei_reranker.connect()
    if isinstance(tei_reranker, HealthCheckable) and tei_reranker.health_check():
        reranker = tei_reranker

    # Fall back to HuggingFace Inference API
    if reranker is None and hf_config is not None:
        hf_reranker = create_reranker(hf_config)
        if isinstance(hf_reranker, Connectable):
            hf_reranker.connect()
        if isinstance(hf_reranker, HealthCheckable) and hf_reranker.health_check():
            reranker = hf_reranker

    # Final fallback: feature-based
    if reranker is None:
        reranker = FeatureBasedReranker()
    top_n = int(overrides.get("reranker_top_n", 0))
    reranker_service = RerankerService(reranker=reranker, top_n=top_n)

    # --- Context builder ---
    try:
        from libs.chunking.token_counter import TiktokenTokenCounter

        counter = TiktokenTokenCounter()
    except ImportError:
        _logger.warning("tiktoken not installed, falling back to WhitespaceTokenCounter")
        counter = WhitespaceTokenCounter()
    builder = GreedyContextBuilder(token_counter=counter)
    context_builder_service = ContextBuilderService(builder=builder)

    # --- Generation (prefer OpenRouter > OpenAI > Gemini > vLLM) ---
    generator_config = (
        openrouter_config or openai_config or gemini_config or vllm_config
    )
    generator = create_generator(generator_config)
    if isinstance(generator, Connectable):
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
    if isinstance(embedding_provider, Connectable):
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
        retry_config=retry_config,
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
        async_retrieval_broker=async_retrieval_broker,
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
