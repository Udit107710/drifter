"""Provider configuration dataclasses with validation."""

from __future__ import annotations

from dataclasses import dataclass, field, fields


def _masked_repr(obj: object, secret_fields: tuple[str, ...]) -> str:
    """Build a repr string that masks secret fields."""
    cls = type(obj)
    parts: list[str] = []
    for f in fields(obj):  # type: ignore[arg-type]
        value = getattr(obj, f.name)
        if f.name in secret_fields and value is not None:
            parts.append(f"{f.name}='***'")
        else:
            parts.append(f"{f.name}={value!r}")
    return f"{cls.__name__}({', '.join(parts)})"


@dataclass(frozen=True)
class QdrantConfig:
    """Configuration for Qdrant vector database."""

    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    api_key: str | None = None
    collection_name: str = "drifter"
    timeout_s: float = 10.0
    use_tls: bool = False

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must not be empty")
        if self.port <= 0:
            raise ValueError("port must be > 0")
        if self.grpc_port <= 0:
            raise ValueError("grpc_port must be > 0")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")

    def __repr__(self) -> str:
        return _masked_repr(self, ("api_key",))


@dataclass(frozen=True)
class OpenSearchConfig:
    """Configuration for OpenSearch lexical search."""

    hosts: list[str] = field(default_factory=lambda: ["localhost:9200"])
    username: str = "admin"
    password: str = "admin"
    index_prefix: str = "drifter"
    use_ssl: bool = True
    timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if not self.hosts:
            raise ValueError("hosts must not be empty")
        for h in self.hosts:
            if not h:
                raise ValueError("each host must not be empty")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")

    def __repr__(self) -> str:
        return _masked_repr(self, ("password",))


@dataclass(frozen=True)
class TeiConfig:
    """Configuration for Text Embeddings Inference."""

    base_url: str = "http://localhost:8080"
    model_id: str = ""
    model_version: str = ""
    timeout_s: float = 30.0
    max_batch_size: int = 32

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")


@dataclass(frozen=True)
class VllmConfig:
    """Configuration for vLLM inference server."""

    base_url: str = "http://localhost:8000"
    model_id: str = ""
    api_key: str | None = None
    timeout_s: float = 60.0
    max_tokens: int = 1024
    temperature: float = 0.1

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if not (0 <= self.temperature <= 2):
            raise ValueError("temperature must be between 0 and 2")

    def __repr__(self) -> str:
        return _masked_repr(self, ("api_key",))


@dataclass(frozen=True)
class UnstructuredConfig:
    """Configuration for Unstructured document parser."""

    base_url: str = "http://localhost:8000"
    strategy: str = "auto"
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        if self.strategy not in {"auto", "fast", "hi_res", "ocr_only"}:
            raise ValueError(
                "strategy must be one of: auto, fast, hi_res, ocr_only"
            )
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")


@dataclass(frozen=True)
class TikaConfig:
    """Configuration for Apache Tika parser."""

    base_url: str = "http://localhost:9998"
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("base_url must not be empty")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")


@dataclass(frozen=True)
class RagasConfig:
    """Configuration for Ragas evaluation framework."""

    model_id: str = ""
    metrics: list[str] = field(
        default_factory=lambda: ["faithfulness", "answer_relevancy"]
    )

    def __post_init__(self) -> None:
        if not self.metrics:
            raise ValueError("metrics must not be empty")


@dataclass(frozen=True)
class GeminiConfig:
    """Configuration for Google Gemini LLM."""

    api_key: str = ""
    model_id: str = "gemini-2.5-flash"
    timeout_s: float = 60.0
    max_tokens: int = 1024
    temperature: float = 0.1

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("api_key must not be empty")
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if not (0 <= self.temperature <= 2):
            raise ValueError("temperature must be between 0 and 2")

    def __repr__(self) -> str:
        return _masked_repr(self, ("api_key",))


@dataclass(frozen=True)
class OtelConfig:
    """Configuration for OpenTelemetry observability."""

    endpoint: str = "http://localhost:4318"
    protocol: str = "http/protobuf"
    service_name: str = "drifter"
    export_interval_ms: int = 5000
    insecure: bool = True

    def __post_init__(self) -> None:
        if not self.endpoint:
            raise ValueError("endpoint must not be empty")
        if self.protocol not in {"http/protobuf", "grpc"}:
            raise ValueError("protocol must be one of: http/protobuf, grpc")
        if self.export_interval_ms <= 0:
            raise ValueError("export_interval_ms must be > 0")


@dataclass(frozen=True)
class LangfuseConfig:
    """Configuration for Langfuse observability."""

    public_key: str = ""
    secret_key: str = ""
    host: str = "http://localhost:3000"
    redis_url: str | None = None
    buffer_ttl_s: int = 300

    def __post_init__(self) -> None:
        if not self.public_key:
            raise ValueError("public_key must not be empty")
        if not self.secret_key:
            raise ValueError("secret_key must not be empty")
        if not self.host:
            raise ValueError("host must not be empty")
        if self.buffer_ttl_s <= 0:
            raise ValueError("buffer_ttl_s must be > 0")

    def __repr__(self) -> str:
        return _masked_repr(self, ("secret_key",))
