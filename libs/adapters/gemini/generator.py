"""Google Gemini generator adapter.

Implements the ``Generator`` protocol using the ``google-genai`` SDK.
"""

from __future__ import annotations

import logging
import re

from google import genai
from google.genai import types

from libs.adapters.config import GeminiConfig
from libs.contracts.generation import Citation, GeneratedAnswer, TokenUsage
from libs.generation.models import GenerationRequest

logger = logging.getLogger(__name__)


class GeminiGenerator:
    """Gemini-backed generator implementing the ``Generator`` protocol."""

    def __init__(self, config: GeminiConfig) -> None:
        self._config = config
        self._generator_id = f"gemini:{config.model_id}"
        self._client: genai.Client | None = None

    @property
    def generator_id(self) -> str:
        return self._generator_id

    def connect(self) -> None:
        """Create the Gemini client."""
        self._client = genai.Client(api_key=self._config.api_key)
        logger.info("Gemini client configured for model: %s", self._config.model_id)

    def close(self) -> None:
        """No persistent connection to close."""
        self._client = None

    def health_check(self) -> bool:
        """Check if the client is configured."""
        return self._client is not None

    def generate(self, request: GenerationRequest) -> GeneratedAnswer:
        """Generate an answer using Gemini."""
        if self._client is None:
            raise RuntimeError("GeminiGenerator is not connected. Call connect() first.")

        response = self._client.models.generate_content(
            model=self._config.model_id,
            contents=request.rendered_prompt,
            config=types.GenerateContentConfig(
                system_instruction=request.system_prompt,
                temperature=self._config.temperature,
                max_output_tokens=self._config.max_tokens,
            ),
        )

        answer_text = response.text or ""

        # Extract token usage from response metadata
        usage = response.usage_metadata
        if usage is not None:
            prompt_tokens = usage.prompt_token_count or 0
            completion_tokens = usage.candidates_token_count or 0
        else:
            prompt_tokens = len(request.rendered_prompt.split())
            completion_tokens = len(answer_text.split())

        # Extract citations by matching chunk IDs referenced in the answer
        citations = _extract_citations(answer_text, request.context_chunk_ids)

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            model_id=self._config.model_id,
            token_usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            trace_id=request.trace_id,
        )


def _extract_citations(
    answer_text: str, context_chunk_ids: list[str]
) -> list[Citation]:
    """Extract citations from the answer text by finding referenced chunk IDs."""
    citations: list[Citation] = []
    seen: set[str] = set()

    for chunk_id in context_chunk_ids:
        if chunk_id in seen:
            continue
        if chunk_id not in answer_text:
            continue
        seen.add(chunk_id)

        claim = _find_claim_for_chunk(answer_text, chunk_id)
        if not claim:
            claim = f"Referenced chunk {chunk_id}"

        citations.append(
            Citation(
                claim=claim,
                chunk_id=chunk_id,
                chunk_content=f"Content from {chunk_id}",
                source_id=f"src-{chunk_id}",
                confidence=0.8,
            )
        )

    return citations


def _find_claim_for_chunk(text: str, chunk_id: str) -> str:
    """Find the sentence containing the chunk reference."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        if chunk_id in sentence:
            cleaned = sentence.strip()
            if cleaned:
                return cleaned[:200]
    return ""
