"""OpenAI generator adapter.

Implements the ``Generator`` protocol using the ``openai`` Python SDK.
Supports OpenAI API and any compatible endpoint (Azure, local proxies).
"""

from __future__ import annotations

import logging
import re

import httpx

from libs.adapters.config import OpenAIConfig
from libs.contracts.generation import Citation, GeneratedAnswer, TokenUsage
from libs.generation.models import GenerationRequest

logger = logging.getLogger(__name__)


class OpenAIGenerator:
    """OpenAI-backed generator implementing the ``Generator`` protocol."""

    def __init__(self, config: OpenAIConfig) -> None:
        self._config = config
        self._generator_id = f"openai:{config.model_id}"
        self._client: httpx.Client | None = None

    @property
    def generator_id(self) -> str:
        return self._generator_id

    def connect(self) -> None:
        """Create the HTTP client for OpenAI API."""
        self._client = httpx.Client(
            base_url=self._config.base_url,
            timeout=self._config.timeout_s,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
        )
        logger.info(
            "OpenAI client configured for model: %s (base_url=%s)",
            self._config.model_id,
            self._config.base_url,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def health_check(self) -> bool:
        """Check if the client is configured."""
        return self._client is not None

    def generate(self, request: GenerationRequest) -> GeneratedAnswer:
        """Generate an answer using the OpenAI Chat Completions API."""
        if self._client is None:
            raise RuntimeError(
                "OpenAIGenerator is not connected. Call connect() first."
            )

        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.rendered_prompt},
        ]

        resp = self._client.post(
            "/v1/chat/completions",
            json={
                "model": self._config.model_id,
                "messages": messages,
                "temperature": self._config.temperature,
                "max_completion_tokens": self._config.max_tokens,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        answer_text = data["choices"][0]["message"]["content"] or ""

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        if not prompt_tokens:
            prompt_tokens = len(request.rendered_prompt.split())
        if not completion_tokens:
            completion_tokens = len(answer_text.split())

        citations = _extract_citations(
            answer_text, request.context_chunk_ids,
        )

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
    answer_text: str, context_chunk_ids: list[str],
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
