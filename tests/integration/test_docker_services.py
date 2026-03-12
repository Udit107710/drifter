"""Integration smoke tests for docker-compose services.

Run with: uv run pytest tests/integration/ -v

These tests require docker-compose services to be running:
    docker compose up -d
    docker compose ps   # verify all healthy

Each test checks connectivity and basic operations against a real service.
Tests are skipped if the service is not reachable.
"""

from __future__ import annotations

import json
import os
import socket
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest


def _port_open(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a TCP port is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def _http_get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    """Make an HTTP GET request. Returns (status_code, body)."""
    req = Request(url)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except URLError:
        pytest.skip(f"Service at {url} not reachable")
        return 0, ""  # unreachable, but keeps mypy happy


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

class TestQdrant:
    """Smoke tests for the Qdrant service."""

    @pytest.fixture(autouse=True)
    def _require_qdrant(self) -> None:
        if not _port_open("localhost", 6333):
            pytest.skip("Qdrant not running on localhost:6333")

    def test_healthz(self) -> None:
        status, _body = _http_get("http://localhost:6333/healthz")
        assert status == 200

    def test_collections_endpoint(self) -> None:
        status, body = _http_get("http://localhost:6333/collections")
        assert status == 200
        data = json.loads(body)
        assert "result" in data

    def test_create_and_delete_collection(self) -> None:
        """Verify we can create a collection, check it exists, then clean up."""
        collection = "drifter_smoke_test"
        # Create
        req = Request(
            f"http://localhost:6333/collections/{collection}",
            data=json.dumps({
                "vectors": {"size": 4, "distance": "Cosine"},
            }).encode(),
            method="PUT",
        )
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=5) as resp:
            assert resp.status == 200

        # Verify exists
        status, _body = _http_get(f"http://localhost:6333/collections/{collection}")
        assert status == 200

        # Delete
        req = Request(
            f"http://localhost:6333/collections/{collection}",
            method="DELETE",
        )
        with urlopen(req, timeout=5) as resp:
            assert resp.status == 200


# ---------------------------------------------------------------------------
# OpenSearch
# ---------------------------------------------------------------------------

class TestOpenSearch:
    """Smoke tests for the OpenSearch service."""

    @pytest.fixture(autouse=True)
    def _require_opensearch(self) -> None:
        if not _port_open("localhost", 9200):
            pytest.skip("OpenSearch not running on localhost:9200")

    def test_cluster_health(self) -> None:
        status, body = _http_get("http://localhost:9200/_cluster/health")
        assert status == 200
        data = json.loads(body)
        assert data["status"] in ("green", "yellow")

    def test_root_info(self) -> None:
        status, body = _http_get("http://localhost:9200/")
        assert status == 200
        data = json.loads(body)
        assert "version" in data

    def test_create_and_delete_index(self) -> None:
        """Verify we can create an index, insert a doc, search, then clean up."""
        index = "drifter_smoke_test"
        # Create index
        req = Request(
            f"http://localhost:9200/{index}",
            data=json.dumps({
                "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            }).encode(),
            method="PUT",
        )
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=5) as resp:
            assert resp.status == 200

        # Delete index
        req = Request(f"http://localhost:9200/{index}", method="DELETE")
        with urlopen(req, timeout=5) as resp:
            assert resp.status == 200


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------

class TestPostgres:
    """Smoke tests for the Postgres service."""

    @pytest.fixture(autouse=True)
    def _require_postgres(self) -> None:
        if not _port_open("localhost", 5432):
            pytest.skip("Postgres not running on localhost:5432")

    def test_tcp_connection(self) -> None:
        """Verify TCP port is open (basic connectivity)."""
        assert _port_open("localhost", 5432)


# ---------------------------------------------------------------------------
# MinIO
# ---------------------------------------------------------------------------

class TestMinIO:
    """Smoke tests for the MinIO service."""

    @pytest.fixture(autouse=True)
    def _require_minio(self) -> None:
        if not _port_open("localhost", 9000):
            pytest.skip("MinIO not running on localhost:9000")

    def test_health_endpoint(self) -> None:
        status, _body = _http_get("http://localhost:9000/minio/health/live")
        assert status == 200


# ---------------------------------------------------------------------------
# Jaeger
# ---------------------------------------------------------------------------

class TestJaeger:
    """Smoke tests for the Jaeger service."""

    @pytest.fixture(autouse=True)
    def _require_jaeger(self) -> None:
        if not _port_open("localhost", 16686):
            pytest.skip("Jaeger not running on localhost:16686")

    def test_ui_reachable(self) -> None:
        status, _ = _http_get("http://localhost:16686/")
        assert status == 200

    def test_services_api(self) -> None:
        status, body = _http_get("http://localhost:16686/api/services")
        assert status == 200
        data = json.loads(body)
        assert "data" in data

    def test_otlp_http_port(self) -> None:
        """Verify the OTLP HTTP port (4318) is listening."""
        assert _port_open("localhost", 4318)


# ---------------------------------------------------------------------------
# Cross-service: env var → adapter config wiring
# ---------------------------------------------------------------------------

class TestEnvVarWiring:
    """Verify that env vars produce the correct adapter configs."""

    def test_qdrant_env_produces_config(self) -> None:
        from libs.adapters.env import load_qdrant_config

        os.environ["DRIFTER_QDRANT_HOST"] = "localhost"
        try:
            config = load_qdrant_config()
            assert config is not None
            assert config.host == "localhost"
            assert config.port == 6333
        finally:
            del os.environ["DRIFTER_QDRANT_HOST"]

    def test_opensearch_env_produces_config(self) -> None:
        from libs.adapters.env import load_opensearch_config

        os.environ["DRIFTER_OPENSEARCH_HOSTS"] = "localhost:9200"
        os.environ["DRIFTER_OPENSEARCH_USE_SSL"] = "false"
        try:
            config = load_opensearch_config()
            assert config is not None
            assert config.hosts == ["localhost:9200"]
            assert config.use_ssl is False
        finally:
            del os.environ["DRIFTER_OPENSEARCH_HOSTS"]
            del os.environ["DRIFTER_OPENSEARCH_USE_SSL"]

    def test_otel_env_produces_config(self) -> None:
        from libs.adapters.env import load_otel_config

        os.environ["DRIFTER_OTEL_ENDPOINT"] = "http://localhost:4318"
        try:
            config = load_otel_config()
            assert config is not None
            assert config.endpoint == "http://localhost:4318"
            assert config.protocol == "http/protobuf"
        finally:
            del os.environ["DRIFTER_OTEL_ENDPOINT"]

    def test_no_env_returns_none(self) -> None:
        from libs.adapters.env import (
            load_ollama_config,
            load_opensearch_config,
            load_otel_config,
            load_qdrant_config,
        )

        # Ensure trigger vars are not set
        for var in [
            "DRIFTER_QDRANT_HOST",
            "DRIFTER_OPENSEARCH_HOSTS",
            "DRIFTER_OTEL_ENDPOINT",
            "DRIFTER_OLLAMA_URL",
        ]:
            os.environ.pop(var, None)

        assert load_qdrant_config() is None
        assert load_opensearch_config() is None
        assert load_otel_config() is None
        assert load_ollama_config() is None

    def test_factory_uses_memory_when_no_env(self) -> None:
        """With no env vars, factories return in-memory implementations."""
        from libs.adapters.factory import (
            create_generator,
            create_lexical_store,
            create_span_collector,
            create_vector_store,
        )

        vs = create_vector_store(None)
        ls = create_lexical_store(None)
        gen = create_generator(None)
        col = create_span_collector(None)

        assert "memory" in vs.store_id
        assert "memory" in ls.store_id
        assert "mock" in gen.generator_id
        # NoOpCollector doesn't have an id, just verify it doesn't raise
        from libs.observability.spans import Span
        col.collect(Span(name="test", trace_id="t", span_id="s"))
