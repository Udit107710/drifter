"""vLLM generator adapter.

Implements the ``Generator`` protocol using the native vLLM
``/v1/chat/completions`` endpoint.  vLLM is OpenAI-compatible but
exposes extra parameters (``top_k``, ``min_p``, ``repetition_penalty``)
as top-level fields and thinking-model reasoning via
``choices[0].message.reasoning``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

import httpx

from libs.adapters.config import VllmConfig
from libs.contracts.generation import Citation, GeneratedAnswer, TokenUsage
from libs.generation.models import GenerationRequest

logger = logging.getLogger(__name__)


class VllmGenerator:
    """vLLM-backed generator implementing the ``Generator`` protocol."""

    def __init__(self, config: VllmConfig) -> None:
        self._config = config
        self._generator_id = f"vllm:{config.model_id}"
        self._client: httpx.Client | None = None

    @property
    def generator_id(self) -> str:
        return self._generator_id

    def connect(self) -> None:
        """Create the HTTP client for the vLLM server."""
        self._client = httpx.Client(
            base_url=self._config.base_url,
            timeout=self._config.timeout_s,
            headers={"Content-Type": "application/json"},
        )
        logger.info(
            "vLLM client configured for model: %s (base_url=%s)",
            self._config.model_id,
            self._config.base_url,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def health_check(self) -> bool:
        """Check if the vLLM server is reachable."""
        if self._client is None:
            return False
        try:
            resp = self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def generate(
        self,
        request: GenerationRequest,
        *,
        on_token: Callable[[str, bool], None] | None = None,
    ) -> GeneratedAnswer:
        """Generate an answer using the vLLM Chat Completions API.

        Args:
            request: The generation request with prompts and context.
            on_token: Optional callback ``(text, is_thinking)`` for streaming.
                      When provided, enables SSE streaming and calls back for
                      each token as it arrives.
        """
        if self._client is None:
            raise RuntimeError(
                "VllmGenerator is not connected. Call connect() first."
            )

        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.rendered_prompt},
        ]

        payload: dict[str, Any] = {
            "model": self._config.model_id,
            "messages": messages,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "top_k": self._config.top_k,
            "min_p": self._config.min_p,
            "repetition_penalty": self._config.repetition_penalty,
            "stream": on_token is not None,
        }
        if self._config.stop:
            payload["stop"] = self._config.stop

        if on_token is not None:
            return self._generate_streaming(payload, request, on_token)
        return self._generate_sync(payload, request)

    def _generate_sync(
        self, payload: dict[str, Any], request: GenerationRequest,
    ) -> GeneratedAnswer:
        """Non-streaming generation."""
        assert self._client is not None
        resp = self._client.post("/v1/chat/completions", json=payload)
        if resp.status_code != 200:
            logger.error("vLLM error response: %s", resp.text[:500])
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]["message"]
        answer_text = choice.get("content", "")

        # vLLM thinking models put reasoning in `reasoning` field
        reasoning = choice.get("reasoning", "")
        if reasoning:
            logger.debug("vLLM model returned %d chars of reasoning", len(reasoning))

        # Strip residual <think> blocks
        answer_text = re.sub(
            r"<think>.*?</think>", "", answer_text, flags=re.DOTALL,
        ).strip()

        if not answer_text:
            if reasoning:
                logger.warning(
                    "vLLM model %s returned empty content with %d chars "
                    "of reasoning — increase max_tokens (currently %d)",
                    self._config.model_id,
                    len(reasoning),
                    self._config.max_tokens,
                )
            raise RuntimeError(
                f"vLLM model {self._config.model_id} returned an empty "
                f"answer. If using a thinking model, increase max_tokens "
                f"(currently {self._config.max_tokens})."
            )

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", len(request.rendered_prompt.split()))
        completion_tokens = usage.get("completion_tokens", len(answer_text.split()))

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

    def _generate_streaming(
        self,
        payload: dict[str, Any],
        request: GenerationRequest,
        on_token: Callable[[str, bool], None],
    ) -> GeneratedAnswer:
        """SSE streaming generation with token callback."""
        assert self._client is not None

        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        with self._client.stream(
            "POST", "/v1/chat/completions", json=payload,
        ) as resp:
            if resp.status_code != 200:
                resp.read()
                logger.error("vLLM streaming error: %s", resp.text[:500])
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})

                # vLLM thinking: delta.reasoning
                reasoning_text = delta.get("reasoning", "")
                if reasoning_text:
                    reasoning_parts.append(reasoning_text)
                    on_token(reasoning_text, True)

                # Content
                content_text = delta.get("content", "")
                if content_text:
                    answer_parts.append(content_text)
                    on_token(content_text, False)

                # Extract usage from final chunk if available
                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

        answer_text = "".join(answer_parts).strip()
        answer_text = re.sub(
            r"<think>.*?</think>", "", answer_text, flags=re.DOTALL,
        ).strip()

        if not answer_text:
            raise RuntimeError(
                f"vLLM model {self._config.model_id} returned an empty "
                f"answer via streaming."
            )

        if not prompt_tokens:
            prompt_tokens = len(request.rendered_prompt.split())
        if not completion_tokens:
            completion_tokens = len(answer_text.split())

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
