"""Tests for YAML config loader and VllmConfig."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from libs.adapters.config import VllmConfig
from libs.adapters.config_loader import DrifterConfig, load_config


class TestVllmConfig:
    def test_defaults(self) -> None:
        cfg = VllmConfig(model_id="meta-llama/Llama-3.2-3B")
        assert cfg.base_url == "http://localhost:8000"
        assert cfg.timeout_s == 120.0
        assert cfg.max_tokens == 4096
        assert cfg.temperature == 0.1
        assert cfg.top_k == -1
        assert cfg.top_p == 0.9
        assert cfg.min_p == 0.0
        assert cfg.repetition_penalty == 1.0
        assert cfg.stop == []

    def test_empty_model_id_raises(self) -> None:
        with pytest.raises(ValueError, match="model_id must not be empty"):
            VllmConfig()

    def test_empty_base_url_raises(self) -> None:
        with pytest.raises(ValueError, match="base_url must not be empty"):
            VllmConfig(model_id="model", base_url="")

    def test_bad_temperature_raises(self) -> None:
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            VllmConfig(model_id="model", temperature=3.0)

    def test_bad_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens must be > 0"):
            VllmConfig(model_id="model", max_tokens=0)

    def test_bad_timeout_raises(self) -> None:
        with pytest.raises(ValueError, match="timeout_s must be > 0"):
            VllmConfig(model_id="model", timeout_s=0)


class TestLoadConfig:
    def test_returns_default_when_no_file(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert cfg == DrifterConfig()

    def test_loads_yaml_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            generation:
              provider: ollama
            embeddings:
              provider: tei
            token_budget: 8000
            ollama:
              base_url: http://ollama:11434
              model_id: qwen3.5:9b
              temperature: 0.5
            qdrant:
              host: qdrant-server
              port: 6333
        """))

        cfg = load_config(config_file)
        assert cfg.generation_provider == "ollama"
        assert cfg.embeddings_provider == "tei"
        assert cfg.token_budget == 8000
        assert cfg.ollama is not None
        assert cfg.ollama.base_url == "http://ollama:11434"
        assert cfg.ollama.model_id == "qwen3.5:9b"
        assert cfg.ollama.temperature == 0.5
        assert cfg.qdrant is not None
        assert cfg.qdrant.host == "qdrant-server"

    def test_env_secrets_override_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            openai:
              model_id: gpt-4o
        """))

        with patch.dict("os.environ", {"DRIFTER_OPENAI_API_KEY": "sk-test123"}, clear=False):
            cfg = load_config(config_file)

        assert cfg.openai is not None
        assert cfg.openai.api_key == "sk-test123"
        assert cfg.openai.model_id == "gpt-4o"

    def test_vllm_section(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            generation:
              provider: vllm
            vllm:
              base_url: http://vllm:8000
              model_id: meta-llama/Llama-3.2-3B-Instruct
              max_tokens: 2048
              top_k: 50
              min_p: 0.05
              stop:
                - "\\n"
                - "user:"
        """))

        cfg = load_config(config_file)
        assert cfg.generation_provider == "vllm"
        assert cfg.vllm is not None
        assert cfg.vllm.base_url == "http://vllm:8000"
        assert cfg.vllm.model_id == "meta-llama/Llama-3.2-3B-Instruct"
        assert cfg.vllm.max_tokens == 2048
        assert cfg.vllm.top_k == 50
        assert cfg.vllm.min_p == 0.05
        assert cfg.vllm.stop == ["\n", "user:"]

    def test_sections_not_in_yaml_are_none(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            generation:
              provider: ollama
            ollama:
              model_id: llama3.2
        """))

        cfg = load_config(config_file)
        assert cfg.vllm is None
        assert cfg.openai is None
        assert cfg.gemini is None
        assert cfg.tei is None

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        cfg = load_config(config_file)
        assert cfg == DrifterConfig()

    def test_huggingface_secret_from_env(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            huggingface:
              reranker_model: BAAI/bge-reranker-v2-m3
        """))

        with patch.dict("os.environ", {"DRIFTER_HF_TOKEN": "hf-test"}, clear=False):
            cfg = load_config(config_file)

        assert cfg.huggingface is not None
        assert cfg.huggingface.api_key == "hf-test"
        assert cfg.huggingface.reranker_model == "BAAI/bge-reranker-v2-m3"
