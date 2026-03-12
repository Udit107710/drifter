"""YAML configuration loader with env-var secret injection.

Loads ``config.yaml`` (non-secrets) and merges with environment variables
(secrets only).  Falls back to pure env-var loading when no config file
exists, preserving backwards compatibility.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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
    TeiConfig,
    TikaConfig,
    UnstructuredConfig,
    VllmConfig,
)

logger = logging.getLogger(__name__)

# Default config file location (project root)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@dataclass(frozen=True)
class DrifterConfig:
    """Top-level application configuration.

    Explicit ``provider`` fields select which adapter to use for each
    concern.  All adapter sections are loaded simultaneously — the
    provider field decides which one is active.
    """

    # --- Explicit provider selection ---
    generation_provider: str = ""
    embeddings_provider: str = ""
    reranking_provider: str = ""
    observability_provider: str = ""

    # --- Adapter configs (all loaded, provider selects active) ---
    qdrant: QdrantConfig | None = None
    opensearch: OpenSearchConfig | None = None
    tei: TeiConfig | None = None
    ollama: OllamaConfig | None = None
    vllm: VllmConfig | None = None
    openrouter: OpenRouterConfig | None = None
    openai: OpenAIConfig | None = None
    gemini: GeminiConfig | None = None
    huggingface: HuggingFaceConfig | None = None
    otel: OtelConfig | None = None
    langfuse: LangfuseConfig | None = None
    unstructured: UnstructuredConfig | None = None
    tika: TikaConfig | None = None

    # --- Pipeline settings ---
    token_budget: int = 5000
    reranker_top_n: int = 0
    retry: dict[str, Any] = field(default_factory=dict)


def load_config(config_path: Path | str | None = None) -> DrifterConfig:
    """Load configuration from YAML file with env-var secret injection.

    If no config file exists at *config_path*, returns a default
    ``DrifterConfig`` (backwards compatible — bootstrap falls back to
    env-var loaders).
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        logger.debug("No config file at %s, using defaults", path)
        return DrifterConfig()

    logger.info("Loading config from %s", path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return _build_config(raw)


def _build_config(raw: dict[str, Any]) -> DrifterConfig:
    """Build DrifterConfig from parsed YAML dict."""
    return DrifterConfig(
        generation_provider=raw.get("generation", {}).get("provider", ""),
        embeddings_provider=raw.get("embeddings", {}).get("provider", ""),
        reranking_provider=raw.get("reranking", {}).get("provider", ""),
        observability_provider=raw.get("observability", {}).get("provider", ""),
        qdrant=_build_qdrant(raw.get("qdrant")),
        opensearch=_build_opensearch(raw.get("opensearch")),
        tei=_build_tei(raw.get("tei")),
        ollama=_build_ollama(raw.get("ollama")),
        vllm=_build_vllm(raw.get("vllm")),
        openrouter=_build_openrouter(raw.get("openrouter")),
        openai=_build_openai(raw.get("openai")),
        gemini=_build_gemini(raw.get("gemini")),
        huggingface=_build_huggingface(raw.get("huggingface")),
        otel=_build_otel(raw.get("otel")),
        langfuse=_build_langfuse(raw.get("langfuse")),
        unstructured=_build_unstructured(raw.get("unstructured")),
        tika=_build_tika(raw.get("tika")),
        token_budget=raw.get("token_budget", 5000),
        reranker_top_n=raw.get("reranker_top_n", 0),
        retry=raw.get("retry", {}),
    )


def _env(name: str, default: str = "") -> str:
    """Read an environment variable, return default if unset."""
    return os.environ.get(name, default)


def _env_or(name: str, yaml_val: Any, default: Any = None) -> Any:
    """Prefer env var for secrets, fall back to YAML value, then default."""
    env_val = os.environ.get(name)
    if env_val is not None:
        return env_val
    if yaml_val is not None:
        return yaml_val
    return default


# --- Builder functions for each adapter config ---


def _build_qdrant(section: dict[str, Any] | None) -> QdrantConfig | None:
    if section is None:
        return None
    return QdrantConfig(
        host=section.get("host", "localhost"),
        port=int(section.get("port", 6333)),
        grpc_port=int(section.get("grpc_port", 6334)),
        api_key=_env_or("DRIFTER_QDRANT_API_KEY", section.get("api_key")) or None,
        collection_name=section.get("collection_name", "drifter"),
        timeout_s=float(section.get("timeout_s", 10.0)),
        use_tls=bool(section.get("use_tls", False)),
    )


def _build_opensearch(section: dict[str, Any] | None) -> OpenSearchConfig | None:
    if section is None:
        return None
    return OpenSearchConfig(
        hosts=section.get("hosts", ["localhost:9200"]),
        username=section.get("username", "admin"),
        password=_env_or("DRIFTER_OPENSEARCH_PASSWORD", section.get("password"), "admin"),
        index_prefix=section.get("index_prefix", "drifter"),
        use_ssl=bool(section.get("use_ssl", True)),
        timeout_s=float(section.get("timeout_s", 10.0)),
    )


def _build_tei(section: dict[str, Any] | None) -> TeiConfig | None:
    if section is None:
        return None
    return TeiConfig(
        base_url=section.get("base_url", "http://localhost:8080"),
        reranker_url=section.get("reranker_url", ""),
        model_id=section.get("model_id", ""),
        model_version=section.get("model_version", ""),
        reranker_model_id=section.get("reranker_model_id", ""),
        timeout_s=float(section.get("timeout_s", 30.0)),
        max_batch_size=int(section.get("max_batch_size", 32)),
    )


def _build_ollama(section: dict[str, Any] | None) -> OllamaConfig | None:
    if section is None:
        return None
    kwargs: dict[str, Any] = {
        "base_url": section.get("base_url", "http://localhost:11434"),
        "model_id": section.get("model_id", "llama3.2"),
        "timeout_s": float(section.get("timeout_s", 120.0)),
        "num_predict": int(section.get("num_predict", 4096)),
        "num_ctx": int(section.get("num_ctx", 2048)),
        "temperature": float(section.get("temperature", 0.1)),
        "top_k": int(section.get("top_k", 40)),
        "top_p": float(section.get("top_p", 0.9)),
        "min_p": float(section.get("min_p", 0.0)),
        "repeat_penalty": float(section.get("repeat_penalty", 1.1)),
        "repeat_last_n": int(section.get("repeat_last_n", 64)),
        "keep_alive": section.get("keep_alive", "5m"),
    }
    if section.get("seed") is not None:
        kwargs["seed"] = int(section["seed"])
    if section.get("stop"):
        kwargs["stop"] = section["stop"]
    return OllamaConfig(**kwargs)


def _build_vllm(section: dict[str, Any] | None) -> VllmConfig | None:
    if section is None:
        return None
    kwargs: dict[str, Any] = {
        "base_url": section.get("base_url", "http://localhost:8000"),
        "model_id": section.get("model_id", ""),
        "timeout_s": float(section.get("timeout_s", 120.0)),
        "max_tokens": int(section.get("max_tokens", 4096)),
        "temperature": float(section.get("temperature", 0.1)),
        "top_k": int(section.get("top_k", -1)),
        "top_p": float(section.get("top_p", 0.9)),
        "min_p": float(section.get("min_p", 0.0)),
        "repetition_penalty": float(section.get("repetition_penalty", 1.0)),
    }
    if section.get("stop"):
        kwargs["stop"] = section["stop"]
    return VllmConfig(**kwargs)


def _build_openrouter(section: dict[str, Any] | None) -> OpenRouterConfig | None:
    if section is None:
        return None
    api_key = _env_or("DRIFTER_OPENROUTER_API_KEY", section.get("api_key"), "")
    if not api_key:
        return None
    return OpenRouterConfig(
        api_key=api_key,
        model_id=section.get("model_id", "openai/gpt-4o"),
        embedding_model=section.get("embedding_model", ""),
        base_url=section.get("base_url", "https://openrouter.ai/api"),
        app_name=section.get("app_name", "drifter"),
        timeout_s=float(section.get("timeout_s", 60.0)),
        max_tokens=int(section.get("max_tokens", 1024)),
        max_batch_size=int(section.get("max_batch_size", 32)),
        temperature=float(section.get("temperature", 0.1)),
    )


def _build_openai(section: dict[str, Any] | None) -> OpenAIConfig | None:
    if section is None:
        return None
    api_key = _env_or("DRIFTER_OPENAI_API_KEY", section.get("api_key"), "")
    if not api_key:
        return None
    return OpenAIConfig(
        api_key=api_key,
        model_id=section.get("model_id", "gpt-4o"),
        base_url=section.get("base_url", "https://api.openai.com"),
        timeout_s=float(section.get("timeout_s", 60.0)),
        max_tokens=int(section.get("max_tokens", 1024)),
        temperature=float(section.get("temperature", 0.1)),
    )


def _build_gemini(section: dict[str, Any] | None) -> GeminiConfig | None:
    if section is None:
        return None
    api_key = _env_or("DRIFTER_GEMINI_API_KEY", section.get("api_key"), "")
    if not api_key:
        return None
    return GeminiConfig(
        api_key=api_key,
        model_id=section.get("model_id", "gemini-2.5-flash"),
        timeout_s=float(section.get("timeout_s", 60.0)),
        max_tokens=int(section.get("max_tokens", 1024)),
        temperature=float(section.get("temperature", 0.1)),
    )


def _build_huggingface(section: dict[str, Any] | None) -> HuggingFaceConfig | None:
    if section is None:
        return None
    api_key = _env_or("DRIFTER_HF_TOKEN", section.get("api_key"), "")
    if not api_key:
        return None
    return HuggingFaceConfig(
        api_key=api_key,
        reranker_model=section.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
        provider=section.get("provider", "hf-inference"),
        timeout_s=float(section.get("timeout_s", 30.0)),
    )


def _build_otel(section: dict[str, Any] | None) -> OtelConfig | None:
    if section is None:
        return None
    return OtelConfig(
        endpoint=section.get("endpoint", "http://localhost:4318"),
        protocol=section.get("protocol", "http/protobuf"),
        service_name=section.get("service_name", "drifter"),
        export_interval_ms=int(section.get("export_interval_ms", 5000)),
        insecure=bool(section.get("insecure", True)),
    )


def _build_langfuse(section: dict[str, Any] | None) -> LangfuseConfig | None:
    if section is None:
        return None
    public_key = _env_or(
        "DRIFTER_LANGFUSE_PUBLIC_KEY", section.get("public_key"), "",
    )
    secret_key = _env_or(
        "DRIFTER_LANGFUSE_SECRET_KEY", section.get("secret_key"), "",
    )
    if not public_key:
        return None
    kwargs: dict[str, Any] = {
        "public_key": public_key,
        "secret_key": secret_key,
        "host": section.get("host", "http://localhost:3000"),
    }
    redis_url = _env_or("DRIFTER_LANGFUSE_REDIS_URL", section.get("redis_url"))
    if redis_url:
        kwargs["redis_url"] = redis_url
    if section.get("buffer_ttl_s") is not None:
        kwargs["buffer_ttl_s"] = int(section["buffer_ttl_s"])
    return LangfuseConfig(**kwargs)


def _build_unstructured(section: dict[str, Any] | None) -> UnstructuredConfig | None:
    if section is None:
        return None
    return UnstructuredConfig(
        base_url=section.get("base_url", "http://localhost:8000"),
        strategy=section.get("strategy", "auto"),
        timeout_s=float(section.get("timeout_s", 60.0)),
    )


def _build_tika(section: dict[str, Any] | None) -> TikaConfig | None:
    if section is None:
        return None
    return TikaConfig(
        base_url=section.get("base_url", "http://localhost:9998"),
        timeout_s=float(section.get("timeout_s", 30.0)),
    )
