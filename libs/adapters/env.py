"""Environment variable loaders for provider configurations."""

from __future__ import annotations

import os

from libs.adapters.config import (
    GeminiConfig,
    HuggingFaceConfig,
    LangfuseConfig,
    OllamaConfig,
    OpenAIConfig,
    OpenRouterConfig,
    OpenSearchConfig,
    OtelConfig,
    QdrantConfig,
    RagasConfig,
    TeiConfig,
    TikaConfig,
    UnstructuredConfig,
)


def _parse_bool(value: str) -> bool:
    """Parse a boolean from an environment variable string."""
    return value.lower() in ("true", "1")


def load_qdrant_config() -> QdrantConfig | None:
    """Load Qdrant config from DRIFTER_QDRANT_* env vars.

    Returns None if DRIFTER_QDRANT_HOST is not set.
    """
    host = os.environ.get("DRIFTER_QDRANT_HOST")
    if host is None:
        return None
    return QdrantConfig(
        host=host,
        port=int(os.environ.get("DRIFTER_QDRANT_PORT", "6333")),
        grpc_port=int(os.environ.get("DRIFTER_QDRANT_GRPC_PORT", "6334")),
        api_key=os.environ.get("DRIFTER_QDRANT_API_KEY"),
        collection_name=os.environ.get(
            "DRIFTER_QDRANT_COLLECTION", "drifter"
        ),
        timeout_s=float(os.environ.get("DRIFTER_QDRANT_TIMEOUT_S", "10.0")),
        use_tls=_parse_bool(
            os.environ.get("DRIFTER_QDRANT_USE_TLS", "false")
        ),
    )


def load_opensearch_config() -> OpenSearchConfig | None:
    """Load OpenSearch config from DRIFTER_OPENSEARCH_* env vars.

    Returns None if DRIFTER_OPENSEARCH_HOSTS is not set.
    Hosts should be comma-separated.
    """
    hosts_raw = os.environ.get("DRIFTER_OPENSEARCH_HOSTS")
    if hosts_raw is None:
        return None
    hosts = [h.strip() for h in hosts_raw.split(",") if h.strip()]
    return OpenSearchConfig(
        hosts=hosts,
        username=os.environ.get("DRIFTER_OPENSEARCH_USERNAME", "admin"),
        password=os.environ.get("DRIFTER_OPENSEARCH_PASSWORD", "admin"),
        index_prefix=os.environ.get(
            "DRIFTER_OPENSEARCH_INDEX_PREFIX", "drifter"
        ),
        use_ssl=_parse_bool(
            os.environ.get("DRIFTER_OPENSEARCH_USE_SSL", "true")
        ),
        timeout_s=float(
            os.environ.get("DRIFTER_OPENSEARCH_TIMEOUT_S", "10.0")
        ),
    )


def load_tei_config() -> TeiConfig | None:
    """Load TEI config from DRIFTER_TEI_* env vars.

    Returns None if DRIFTER_TEI_URL is not set.
    """
    base_url = os.environ.get("DRIFTER_TEI_URL")
    if base_url is None:
        return None
    return TeiConfig(
        base_url=base_url,
        reranker_url=os.environ.get("DRIFTER_TEI_RERANKER_URL", ""),
        model_id=os.environ.get("DRIFTER_TEI_MODEL_ID", ""),
        model_version=os.environ.get("DRIFTER_TEI_MODEL_VERSION", ""),
        reranker_model_id=os.environ.get("DRIFTER_TEI_RERANKER_MODEL_ID", ""),
        timeout_s=float(os.environ.get("DRIFTER_TEI_TIMEOUT_S", "30.0")),
        max_batch_size=int(
            os.environ.get("DRIFTER_TEI_MAX_BATCH_SIZE", "32")
        ),
    )


def load_ollama_config() -> OllamaConfig | None:
    """Load Ollama config from DRIFTER_OLLAMA_* env vars.

    Returns None if DRIFTER_OLLAMA_URL is not set.
    """
    base_url = os.environ.get("DRIFTER_OLLAMA_URL")
    if base_url is None:
        return None
    kwargs: dict[str, object] = {
        "base_url": base_url,
        "model_id": os.environ.get("DRIFTER_OLLAMA_MODEL", "llama3.2"),
        "timeout_s": float(os.environ.get("DRIFTER_OLLAMA_TIMEOUT_S", "120.0")),
        "num_predict": int(os.environ.get("DRIFTER_OLLAMA_NUM_PREDICT", "4096")),
        "num_ctx": int(os.environ.get("DRIFTER_OLLAMA_NUM_CTX", "2048")),
        "temperature": float(os.environ.get("DRIFTER_OLLAMA_TEMPERATURE", "0.1")),
        "top_k": int(os.environ.get("DRIFTER_OLLAMA_TOP_K", "40")),
        "top_p": float(os.environ.get("DRIFTER_OLLAMA_TOP_P", "0.9")),
        "min_p": float(os.environ.get("DRIFTER_OLLAMA_MIN_P", "0.0")),
        "repeat_penalty": float(
            os.environ.get("DRIFTER_OLLAMA_REPEAT_PENALTY", "1.1")
        ),
        "repeat_last_n": int(os.environ.get("DRIFTER_OLLAMA_REPEAT_LAST_N", "64")),
        "keep_alive": os.environ.get("DRIFTER_OLLAMA_KEEP_ALIVE", "5m"),
    }
    seed_raw = os.environ.get("DRIFTER_OLLAMA_SEED")
    if seed_raw is not None:
        kwargs["seed"] = int(seed_raw)
    stop_raw = os.environ.get("DRIFTER_OLLAMA_STOP")
    if stop_raw is not None:
        kwargs["stop"] = [s.strip() for s in stop_raw.split(",") if s.strip()]
    return OllamaConfig(**kwargs)  # type: ignore[arg-type]


def load_unstructured_config() -> UnstructuredConfig | None:
    """Load Unstructured config from DRIFTER_UNSTRUCTURED_* env vars.

    Returns None if DRIFTER_UNSTRUCTURED_URL is not set.
    """
    base_url = os.environ.get("DRIFTER_UNSTRUCTURED_URL")
    if base_url is None:
        return None
    return UnstructuredConfig(
        base_url=base_url,
        strategy=os.environ.get("DRIFTER_UNSTRUCTURED_STRATEGY", "auto"),
        timeout_s=float(
            os.environ.get("DRIFTER_UNSTRUCTURED_TIMEOUT_S", "60.0")
        ),
    )


def load_tika_config() -> TikaConfig | None:
    """Load Tika config from DRIFTER_TIKA_* env vars.

    Returns None if DRIFTER_TIKA_URL is not set.
    """
    base_url = os.environ.get("DRIFTER_TIKA_URL")
    if base_url is None:
        return None
    return TikaConfig(
        base_url=base_url,
        timeout_s=float(os.environ.get("DRIFTER_TIKA_TIMEOUT_S", "30.0")),
    )


def load_ragas_config() -> RagasConfig | None:
    """Load Ragas config from DRIFTER_RAGAS_* env vars.

    Returns None if DRIFTER_RAGAS_MODEL is not set.
    """
    model_id = os.environ.get("DRIFTER_RAGAS_MODEL")
    if model_id is None:
        return None
    metrics_raw = os.environ.get("DRIFTER_RAGAS_METRICS")
    kwargs: dict[str, object] = {"model_id": model_id}
    if metrics_raw is not None:
        kwargs["metrics"] = [
            m.strip() for m in metrics_raw.split(",") if m.strip()
        ]
    return RagasConfig(**kwargs)  # type: ignore[arg-type]


def load_openrouter_config() -> OpenRouterConfig | None:
    """Load OpenRouter config from DRIFTER_OPENROUTER_* env vars.

    Returns None if DRIFTER_OPENROUTER_API_KEY is not set.
    """
    api_key = os.environ.get("DRIFTER_OPENROUTER_API_KEY")
    if api_key is None:
        return None
    return OpenRouterConfig(
        api_key=api_key,
        model_id=os.environ.get(
            "DRIFTER_OPENROUTER_MODEL", "openai/gpt-4o"
        ),
        embedding_model=os.environ.get(
            "DRIFTER_OPENROUTER_EMBEDDING_MODEL", ""
        ),
        base_url=os.environ.get(
            "DRIFTER_OPENROUTER_BASE_URL", "https://openrouter.ai/api"
        ),
        app_name=os.environ.get("DRIFTER_OPENROUTER_APP_NAME", "drifter"),
        timeout_s=float(
            os.environ.get("DRIFTER_OPENROUTER_TIMEOUT_S", "60.0")
        ),
        max_tokens=int(
            os.environ.get("DRIFTER_OPENROUTER_MAX_TOKENS", "1024")
        ),
        max_batch_size=int(
            os.environ.get("DRIFTER_OPENROUTER_MAX_BATCH_SIZE", "32")
        ),
        temperature=float(
            os.environ.get("DRIFTER_OPENROUTER_TEMPERATURE", "0.1")
        ),
    )


def load_openai_config() -> OpenAIConfig | None:
    """Load OpenAI config from DRIFTER_OPENAI_* env vars.

    Returns None if DRIFTER_OPENAI_API_KEY is not set.
    """
    api_key = os.environ.get("DRIFTER_OPENAI_API_KEY")
    if api_key is None:
        return None
    return OpenAIConfig(
        api_key=api_key,
        model_id=os.environ.get("DRIFTER_OPENAI_MODEL", "gpt-4o"),
        base_url=os.environ.get(
            "DRIFTER_OPENAI_BASE_URL", "https://api.openai.com"
        ),
        timeout_s=float(
            os.environ.get("DRIFTER_OPENAI_TIMEOUT_S", "60.0")
        ),
        max_tokens=int(os.environ.get("DRIFTER_OPENAI_MAX_TOKENS", "1024")),
        temperature=float(
            os.environ.get("DRIFTER_OPENAI_TEMPERATURE", "0.1")
        ),
    )


def load_gemini_config() -> GeminiConfig | None:
    """Load Gemini config from DRIFTER_GEMINI_* env vars.

    Returns None if DRIFTER_GEMINI_API_KEY is not set.
    """
    api_key = os.environ.get("DRIFTER_GEMINI_API_KEY")
    if api_key is None:
        return None
    return GeminiConfig(
        api_key=api_key,
        model_id=os.environ.get("DRIFTER_GEMINI_MODEL", "gemini-2.5-flash"),
        timeout_s=float(os.environ.get("DRIFTER_GEMINI_TIMEOUT_S", "60.0")),
        max_tokens=int(os.environ.get("DRIFTER_GEMINI_MAX_TOKENS", "1024")),
        temperature=float(
            os.environ.get("DRIFTER_GEMINI_TEMPERATURE", "0.1")
        ),
    )


def load_langfuse_config() -> LangfuseConfig | None:
    """Load Langfuse config from DRIFTER_LANGFUSE_* env vars.

    Returns None if DRIFTER_LANGFUSE_PUBLIC_KEY is not set.
    """
    public_key = os.environ.get("DRIFTER_LANGFUSE_PUBLIC_KEY")
    if public_key is None:
        return None
    redis_url = os.environ.get("DRIFTER_LANGFUSE_REDIS_URL")
    buffer_ttl_raw = os.environ.get("DRIFTER_LANGFUSE_BUFFER_TTL_S")
    kwargs: dict[str, object] = {
        "public_key": public_key,
        "secret_key": os.environ.get("DRIFTER_LANGFUSE_SECRET_KEY", ""),
        "host": os.environ.get("DRIFTER_LANGFUSE_HOST", "http://localhost:3000"),
    }
    if redis_url is not None:
        kwargs["redis_url"] = redis_url
    if buffer_ttl_raw is not None:
        kwargs["buffer_ttl_s"] = int(buffer_ttl_raw)
    return LangfuseConfig(**kwargs)  # type: ignore[arg-type]


def load_huggingface_config() -> HuggingFaceConfig | None:
    """Load HuggingFace config from DRIFTER_HF_* env vars.

    Returns None if DRIFTER_HF_TOKEN is not set.
    """
    api_key = os.environ.get("DRIFTER_HF_TOKEN")
    if api_key is None:
        return None
    return HuggingFaceConfig(
        api_key=api_key,
        reranker_model=os.environ.get(
            "DRIFTER_HF_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        ),
        provider=os.environ.get("DRIFTER_HF_PROVIDER", "hf-inference"),
        timeout_s=float(os.environ.get("DRIFTER_HF_TIMEOUT_S", "30.0")),
    )


def load_otel_config() -> OtelConfig | None:
    """Load OpenTelemetry config from DRIFTER_OTEL_* env vars.

    Returns None if DRIFTER_OTEL_ENDPOINT is not set.
    """
    endpoint = os.environ.get("DRIFTER_OTEL_ENDPOINT")
    if endpoint is None:
        return None
    return OtelConfig(
        endpoint=endpoint,
        protocol=os.environ.get("DRIFTER_OTEL_PROTOCOL", "http/protobuf"),
        service_name=os.environ.get("DRIFTER_OTEL_SERVICE_NAME", "drifter"),
        export_interval_ms=int(
            os.environ.get("DRIFTER_OTEL_EXPORT_INTERVAL_MS", "5000")
        ),
        insecure=_parse_bool(
            os.environ.get("DRIFTER_OTEL_INSECURE", "true")
        ),
    )
