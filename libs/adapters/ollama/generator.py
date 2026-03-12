"""Ollama generator adapter.

Implements the ``Generator`` protocol using the native Ollama
``/api/chat`` endpoint (not the OpenAI-compatible shim).
Supports streaming with ``on_token`` callback for thinking models.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

import httpx

from libs.adapters.config import OllamaConfig
from libs.contracts.generation import Citation, GeneratedAnswer, TokenUsage
from libs.generation.models import GenerationRequest

logger = logging.getLogger(__name__)


class OllamaGenerator:
    """Ollama-backed generator implementing the ``Generator`` protocol.

    Uses the native ``POST /api/chat`` endpoint.  When an ``on_token``
    callback is provided, streams NDJSON responses token-by-token so
    thinking model reasoning can be displayed in real time.
    """

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._generator_id = f"ollama:{config.model_id}"
        self._client: httpx.Client | None = None

    @property
    def generator_id(self) -> str:
        return self._generator_id

    def connect(self) -> None:
        """Create the HTTP client for the Ollama server."""
        self._client = httpx.Client(
            base_url=self._config.base_url,
            timeout=self._config.timeout_s,
            headers={"Content-Type": "application/json"},
        )
        logger.info(
            "Ollama client configured for model: %s (base_url=%s)",
            self._config.model_id,
            self._config.base_url,
        )

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def health_check(self) -> bool:
        """Check if the Ollama server is reachable."""
        if self._client is None:
            return False
        try:
            resp = self._client.get("/api/tags")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def generate(
        self,
        request: GenerationRequest,
        *,
        on_token: Callable[[str, bool], None] | None = None,
    ) -> GeneratedAnswer:
        """Generate an answer using the Ollama Chat API.

        Args:
            request: The generation request with prompts and context.
            on_token: Optional callback ``(text, is_thinking)`` for streaming.
                      When provided, streams NDJSON from Ollama and calls
                      back for each token as it arrives.
        """
        if self._client is None:
            raise RuntimeError(
                "OllamaGenerator is not connected. Call connect() first."
            )

        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.rendered_prompt},
        ]

        options: dict[str, Any] = {
            "temperature": self._config.temperature,
            "num_predict": self._config.num_predict,
            "num_ctx": self._config.num_ctx,
            "top_k": self._config.top_k,
            "top_p": self._config.top_p,
            "min_p": self._config.min_p,
            "repeat_penalty": self._config.repeat_penalty,
            "repeat_last_n": self._config.repeat_last_n,
        }
        if self._config.seed is not None:
            options["seed"] = self._config.seed
        if self._config.stop:
            options["stop"] = self._config.stop

        payload: dict[str, Any] = {
            "model": self._config.model_id,
            "messages": messages,
            "stream": on_token is not None,
            "options": options,
            "keep_alive": self._config.keep_alive,
        }

        if on_token is not None:
            return self._generate_streaming(payload, request, on_token)
        return self._generate_sync(payload, request)

    def _generate_sync(
        self, payload: dict[str, Any], request: GenerationRequest,
    ) -> GeneratedAnswer:
        """Non-streaming generation (stream: false)."""
        assert self._client is not None
        resp = self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        message = data.get("message", {})
        answer_text = message.get("content", "")

        # Thinking models put reasoning in the `thinking` field and
        # the answer in `content`.  Strip residual <think> blocks.
        answer_text = re.sub(
            r"<think>.*?</think>", "", answer_text, flags=re.DOTALL,
        ).strip()

        if not answer_text:
            thinking = message.get("thinking", "")
            if thinking:
                logger.warning(
                    "Ollama model %s returned empty content with %d chars "
                    "of thinking — increase num_predict (currently %d) to "
                    "give the model enough token budget for both reasoning "
                    "and the answer",
                    self._config.model_id,
                    len(thinking),
                    self._config.num_predict,
                )
            raise RuntimeError(
                f"Ollama model {self._config.model_id} returned an empty "
                f"answer. If using a thinking model, increase num_predict "
                f"(currently {self._config.num_predict}) to allow enough "
                f"tokens for both reasoning and the response."
            )

        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
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

    def _generate_streaming(
        self,
        payload: dict[str, Any],
        request: GenerationRequest,
        on_token: Callable[[str, bool], None],
    ) -> GeneratedAnswer:
        """NDJSON streaming generation with token callback.

        Ollama streams one JSON object per line.  Each object has:
        - ``message.content``: answer token (may be empty during thinking)
        - ``message.thinking``: thinking token (thinking models only)
        - ``done``: true on the final chunk (includes token counts)
        """
        assert self._client is not None

        answer_parts: list[str] = []
        thinking_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        with self._client.stream(
            "POST", "/api/chat", json=payload,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.strip():
                    continue

                chunk = json.loads(line)
                message = chunk.get("message", {})

                # Thinking token
                thinking_text = message.get("thinking", "")
                if thinking_text:
                    thinking_parts.append(thinking_text)
                    on_token(thinking_text, True)

                # Content token
                content_text = message.get("content", "")
                if content_text:
                    answer_parts.append(content_text)
                    on_token(content_text, False)

                # Final chunk has token counts
                if chunk.get("done"):
                    prompt_tokens = chunk.get("prompt_eval_count", 0)
                    completion_tokens = chunk.get("eval_count", 0)

        answer_text = "".join(answer_parts)
        answer_text = re.sub(
            r"<think>.*?</think>", "", answer_text, flags=re.DOTALL,
        ).strip()

        if not answer_text:
            raise RuntimeError(
                f"Ollama model {self._config.model_id} returned an empty "
                f"answer via streaming."
            )

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
