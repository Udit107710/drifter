"""Tests for generation contracts: TokenUsage, Citation, GeneratedAnswer."""

import pytest

from libs.contracts.generation import Citation, GeneratedAnswer, TokenUsage

# ── TokenUsage ──────────────────────────────────────────────────────


class TestTokenUsage:
    def test_create_valid(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.total_tokens == 150

    def test_negative_prompt_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="prompt_tokens"):
            TokenUsage(prompt_tokens=-1, completion_tokens=0, total_tokens=-1)

    def test_negative_completion_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="completion_tokens"):
            TokenUsage(prompt_tokens=0, completion_tokens=-1, total_tokens=-1)

    def test_total_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="total_tokens must equal"):
            TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=20)

    def test_zero_usage(self) -> None:
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        assert usage.total_tokens == 0


# ── Citation ────────────────────────────────────────────────────────


class TestCitation:
    def test_create_valid(self) -> None:
        c = Citation(
            claim="RAG uses retrieval.",
            chunk_id="chk-001",
            chunk_content="Retrieval Augmented Generation uses retrieval.",
            source_id="src-001",
            confidence=0.95,
        )
        assert c.confidence == 0.95

    def test_schema_version_default(self) -> None:
        c = Citation(
            claim="RAG uses retrieval.",
            chunk_id="chk-001",
            chunk_content="Retrieval Augmented Generation uses retrieval.",
            source_id="src-001",
            confidence=0.95,
        )
        assert c.schema_version == 1

    def test_schema_version_custom(self) -> None:
        c = Citation(
            claim="RAG uses retrieval.",
            chunk_id="chk-001",
            chunk_content="Retrieval Augmented Generation uses retrieval.",
            source_id="src-001",
            confidence=0.95,
            schema_version=2,
        )
        assert c.schema_version == 2

    def test_empty_claim_raises(self) -> None:
        with pytest.raises(ValueError, match="claim"):
            Citation(
                claim="",
                chunk_id="chk-001",
                chunk_content="text",
                source_id="src-001",
                confidence=0.5,
            )

    def test_empty_chunk_id_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_id"):
            Citation(
                claim="claim",
                chunk_id="",
                chunk_content="text",
                source_id="src-001",
                confidence=0.5,
            )

    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            Citation(
                claim="claim",
                chunk_id="chk-001",
                chunk_content="text",
                source_id="src-001",
                confidence=1.1,
            )


# ── GeneratedAnswer ─────────────────────────────────────────────────


class TestGeneratedAnswer:
    def _make_answer(self, **overrides: object) -> GeneratedAnswer:
        defaults: dict[str, object] = {
            "answer": "RAG is a retrieval-augmented generation system.",
            "citations": [],
            "model_id": "llama-3-70b",
            "token_usage": TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
            "trace_id": "trace-001",
        }
        defaults.update(overrides)
        return GeneratedAnswer(**defaults)  # type: ignore[arg-type]

    def test_create_valid(self) -> None:
        answer = self._make_answer()
        assert answer.model_id == "llama-3-70b"
        assert answer.token_usage.total_tokens == 120

    def test_with_citations(self) -> None:
        citation = Citation(
            claim="RAG uses retrieval.",
            chunk_id="chk-001",
            chunk_content="text",
            source_id="src-001",
            confidence=0.9,
        )
        answer = self._make_answer(citations=[citation])
        assert len(answer.citations) == 1

    def test_schema_version_default(self) -> None:
        answer = self._make_answer()
        assert answer.schema_version == 1

    def test_schema_version_custom(self) -> None:
        answer = self._make_answer(schema_version=2)
        assert answer.schema_version == 2

    def test_empty_answer_raises(self) -> None:
        with pytest.raises(ValueError, match="answer"):
            self._make_answer(answer="")

    def test_empty_model_id_raises(self) -> None:
        with pytest.raises(ValueError, match="model_id"):
            self._make_answer(model_id="")

    def test_empty_trace_id_raises(self) -> None:
        with pytest.raises(ValueError, match="trace_id"):
            self._make_answer(trace_id="")
