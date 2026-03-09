"""Tests for orchestrators/bootstrap.py — ServiceRegistry creation."""

from __future__ import annotations

import pytest

from orchestrators.bootstrap import ServiceRegistry, _reject_secret_overrides, create_registry


class TestCreateRegistry:
    """Test registry creation with no external services."""

    def test_creates_registry_with_defaults(self) -> None:
        registry = create_registry()
        assert isinstance(registry, ServiceRegistry)
        assert registry.tracer is not None
        assert registry.retrieval_broker is not None
        assert registry.reranker_service is not None
        assert registry.context_builder_service is not None
        assert registry.generation_service is not None
        assert registry.evaluator is not None
        assert registry.experiment_runner is not None
        assert registry.token_budget == 3000

    def test_token_budget_override(self) -> None:
        registry = create_registry(overrides={"token_budget": "5000"})
        assert registry.token_budget == 5000

    def test_reranker_top_n_override(self) -> None:
        registry = create_registry(overrides={"reranker_top_n": "10"})
        assert registry.reranker_service._top_n == 10

    def test_indexing_service_is_wired(self) -> None:
        registry = create_registry()
        assert registry.indexing_service is not None

    def test_ingestion_orchestrator_is_wired(self) -> None:
        registry = create_registry()
        assert registry.ingestion_orchestrator is not None


class TestSecretRejection:
    """Test that secret fields are rejected from --config overrides."""

    def test_rejects_api_key(self) -> None:
        with pytest.raises(ValueError, match="secret field"):
            _reject_secret_overrides({"api_key": "bad"})

    def test_rejects_password(self) -> None:
        with pytest.raises(ValueError, match="secret field"):
            _reject_secret_overrides({"password": "bad"})

    def test_rejects_nested_secret(self) -> None:
        with pytest.raises(ValueError, match="secret field"):
            _reject_secret_overrides({"qdrant.api_key": "bad"})

    def test_rejects_auth(self) -> None:
        with pytest.raises(ValueError, match="secret field"):
            _reject_secret_overrides({"auth": "bad"})

    def test_allows_non_secret(self) -> None:
        _reject_secret_overrides({"token_budget": "5000"})

    def test_allows_empty(self) -> None:
        _reject_secret_overrides({})
