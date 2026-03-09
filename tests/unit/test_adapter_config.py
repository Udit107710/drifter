"""Tests for adapter configuration dataclasses and env loaders."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from libs.adapters.config import (
    OpenSearchConfig,
    OtelConfig,
    QdrantConfig,
    RagasConfig,
    TeiConfig,
    TikaConfig,
    UnstructuredConfig,
    VllmConfig,
)
from libs.adapters.env import (
    load_opensearch_config,
    load_otel_config,
    load_qdrant_config,
    load_ragas_config,
    load_tei_config,
    load_tika_config,
    load_unstructured_config,
    load_vllm_config,
)

# ── Config construction ───────────────────────────────────────────────


class TestQdrantConfig:
    def test_defaults(self) -> None:
        cfg = QdrantConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 6333
        assert cfg.grpc_port == 6334
        assert cfg.api_key is None
        assert cfg.collection_name == "drifter"
        assert cfg.timeout_s == 10.0
        assert cfg.use_tls is False

    def test_empty_host_raises(self) -> None:
        with pytest.raises(ValueError, match="host must not be empty"):
            QdrantConfig(host="")

    def test_bad_port_raises(self) -> None:
        with pytest.raises(ValueError, match="port must be > 0"):
            QdrantConfig(port=0)

    def test_bad_grpc_port_raises(self) -> None:
        with pytest.raises(ValueError, match="grpc_port must be > 0"):
            QdrantConfig(grpc_port=-1)

    def test_bad_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_s must be > 0"):
            QdrantConfig(timeout_s=0)

    def test_secret_masked_in_repr(self) -> None:
        cfg = QdrantConfig(api_key="super-secret")
        r = repr(cfg)
        assert "super-secret" not in r
        assert "***" in r

    def test_none_api_key_not_masked(self) -> None:
        cfg = QdrantConfig()
        r = repr(cfg)
        assert "api_key=None" in r


class TestOpenSearchConfig:
    def test_defaults(self) -> None:
        cfg = OpenSearchConfig()
        assert cfg.hosts == ["localhost:9200"]
        assert cfg.username == "admin"
        assert cfg.timeout_s == 10.0

    def test_empty_hosts_raises(self) -> None:
        with pytest.raises(ValueError, match="hosts must not be empty"):
            OpenSearchConfig(hosts=[])

    def test_empty_host_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="each host must not be empty"):
            OpenSearchConfig(hosts=["ok", ""])

    def test_password_masked_in_repr(self) -> None:
        cfg = OpenSearchConfig(password="secret123")
        r = repr(cfg)
        assert "secret123" not in r
        assert "***" in r


class TestTeiConfig:
    def test_defaults(self) -> None:
        cfg = TeiConfig()
        assert cfg.base_url == "http://localhost:8080"
        assert cfg.max_batch_size == 32

    def test_empty_url_raises(self) -> None:
        with pytest.raises(ValueError, match="base_url must not be empty"):
            TeiConfig(base_url="")

    def test_bad_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_batch_size must be > 0"):
            TeiConfig(max_batch_size=0)


class TestVllmConfig:
    def test_defaults(self) -> None:
        cfg = VllmConfig()
        assert cfg.base_url == "http://localhost:8000"
        assert cfg.temperature == 0.1

    def test_bad_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            VllmConfig(temperature=3.0)

    def test_bad_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            VllmConfig(max_tokens=0)

    def test_api_key_masked_in_repr(self) -> None:
        cfg = VllmConfig(api_key="my-key")
        r = repr(cfg)
        assert "my-key" not in r
        assert "***" in r


class TestUnstructuredConfig:
    def test_defaults(self) -> None:
        cfg = UnstructuredConfig()
        assert cfg.strategy == "auto"

    def test_bad_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="strategy must be one of"):
            UnstructuredConfig(strategy="invalid")


class TestTikaConfig:
    def test_defaults(self) -> None:
        cfg = TikaConfig()
        assert cfg.base_url == "http://localhost:9998"

    def test_empty_url_raises(self) -> None:
        with pytest.raises(ValueError, match="base_url must not be empty"):
            TikaConfig(base_url="")


class TestRagasConfig:
    def test_defaults(self) -> None:
        cfg = RagasConfig()
        assert "faithfulness" in cfg.metrics

    def test_empty_metrics_raises(self) -> None:
        with pytest.raises(ValueError, match="metrics must not be empty"):
            RagasConfig(metrics=[])


class TestOtelConfig:
    def test_defaults(self) -> None:
        cfg = OtelConfig()
        assert cfg.protocol == "http/protobuf"

    def test_bad_protocol_raises(self) -> None:
        with pytest.raises(ValueError, match="protocol must be one of"):
            OtelConfig(protocol="invalid")

    def test_bad_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="export_interval_ms must be > 0"):
            OtelConfig(export_interval_ms=0)


# ── Env loaders ───────────────────────────────────────────────────────


class TestEnvLoaders:
    def test_qdrant_returns_none_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert load_qdrant_config() is None

    def test_qdrant_from_env(self) -> None:
        env = {
            "DRIFTER_QDRANT_HOST": "qdrant.local",
            "DRIFTER_QDRANT_PORT": "6380",
            "DRIFTER_QDRANT_API_KEY": "key123",
            "DRIFTER_QDRANT_COLLECTION": "test_col",
            "DRIFTER_QDRANT_USE_TLS": "true",
        }
        with patch.dict("os.environ", env, clear=True):
            cfg = load_qdrant_config()
            assert cfg is not None
            assert cfg.host == "qdrant.local"
            assert cfg.port == 6380
            assert cfg.api_key == "key123"
            assert cfg.collection_name == "test_col"
            assert cfg.use_tls is True

    def test_opensearch_returns_none_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert load_opensearch_config() is None

    def test_opensearch_from_env(self) -> None:
        env = {
            "DRIFTER_OPENSEARCH_HOSTS": "host1:9200,host2:9200",
            "DRIFTER_OPENSEARCH_USERNAME": "user",
            "DRIFTER_OPENSEARCH_PASSWORD": "pass",
        }
        with patch.dict("os.environ", env, clear=True):
            cfg = load_opensearch_config()
            assert cfg is not None
            assert cfg.hosts == ["host1:9200", "host2:9200"]
            assert cfg.username == "user"

    def test_tei_returns_none_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert load_tei_config() is None

    def test_tei_from_env(self) -> None:
        env = {"DRIFTER_TEI_URL": "http://tei:8080", "DRIFTER_TEI_MAX_BATCH_SIZE": "64"}
        with patch.dict("os.environ", env, clear=True):
            cfg = load_tei_config()
            assert cfg is not None
            assert cfg.base_url == "http://tei:8080"
            assert cfg.max_batch_size == 64

    def test_vllm_returns_none_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert load_vllm_config() is None

    def test_vllm_from_env(self) -> None:
        env = {"DRIFTER_VLLM_URL": "http://vllm:8000", "DRIFTER_VLLM_TEMPERATURE": "0.5"}
        with patch.dict("os.environ", env, clear=True):
            cfg = load_vllm_config()
            assert cfg is not None
            assert cfg.temperature == 0.5

    def test_unstructured_returns_none_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert load_unstructured_config() is None

    def test_tika_returns_none_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert load_tika_config() is None

    def test_ragas_returns_none_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert load_ragas_config() is None

    def test_ragas_from_env(self) -> None:
        env = {"DRIFTER_RAGAS_MODEL": "gpt-4", "DRIFTER_RAGAS_METRICS": "faithfulness,relevancy"}
        with patch.dict("os.environ", env, clear=True):
            cfg = load_ragas_config()
            assert cfg is not None
            assert cfg.model_id == "gpt-4"
            assert cfg.metrics == ["faithfulness", "relevancy"]

    def test_otel_returns_none_when_unset(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            assert load_otel_config() is None

    def test_otel_from_env(self) -> None:
        env = {
            "DRIFTER_OTEL_ENDPOINT": "http://jaeger:4318",
            "DRIFTER_OTEL_PROTOCOL": "grpc",
            "DRIFTER_OTEL_INSECURE": "false",
        }
        with patch.dict("os.environ", env, clear=True):
            cfg = load_otel_config()
            assert cfg is not None
            assert cfg.protocol == "grpc"
            assert cfg.insecure is False
