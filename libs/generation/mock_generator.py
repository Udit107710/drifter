"""Deterministic mock generator for testing."""

from __future__ import annotations

from libs.contracts.generation import Citation, GeneratedAnswer, TokenUsage
from libs.generation.models import GenerationRequest


class MockGenerator:
    """Deterministic generator that produces predictable answers from context.

    For each chunk ID in the request, creates a claim and citation.
    Useful for testing the pipeline without an actual LLM.
    """

    def __init__(self, model_id: str = "mock-v1") -> None:
        self._model_id = model_id

    @property
    def generator_id(self) -> str:
        return f"mock:{self._model_id}"

    def generate(self, request: GenerationRequest) -> GeneratedAnswer:
        """Generate a deterministic answer with one citation per context chunk."""
        if not request.context_chunk_ids:
            return GeneratedAnswer(
                answer="I don't have enough context to answer this question.",
                citations=[],
                model_id=self._model_id,
                token_usage=TokenUsage(
                    prompt_tokens=len(request.rendered_prompt.split()),
                    completion_tokens=10,
                    total_tokens=len(request.rendered_prompt.split()) + 10,
                ),
                trace_id=request.trace_id,
            )

        # Build answer with one claim per chunk
        claims: list[str] = []
        citations: list[Citation] = []
        for i, chunk_id in enumerate(request.context_chunk_ids):
            claim = f"Based on chunk {chunk_id}, this is claim {i + 1}."
            claims.append(claim)
            citations.append(
                Citation(
                    claim=claim,
                    chunk_id=chunk_id,
                    chunk_content=f"Content from {chunk_id}",
                    source_id=f"src-{chunk_id}",
                    confidence=0.9,
                )
            )

        answer_text = " ".join(claims)
        prompt_tokens = len(request.rendered_prompt.split())
        completion_tokens = len(answer_text.split())

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            model_id=self._model_id,
            token_usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            trace_id=request.trace_id,
        )
