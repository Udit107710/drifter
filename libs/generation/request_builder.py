"""Build GenerationRequest from ContextPack."""

from __future__ import annotations

from libs.contracts.context import ContextPack
from libs.generation.models import GenerationRequest
from libs.generation.prompt_templates import (
    DEFAULT_TEMPLATE,
    PromptTemplate,
    format_context_block,
    render_prompt,
)
from libs.generation.sanitizer import sanitize_content


class GenerationRequestBuilder:
    """Builds a GenerationRequest from a ContextPack.

    Handles:
    - Content sanitization (prompt injection prevention)
    - Prompt template rendering
    - Structural separation of system/user/context
    """

    def __init__(
        self,
        template: PromptTemplate | None = None,
        max_completion_tokens: int = 1024,
    ) -> None:
        self._template = template or DEFAULT_TEMPLATE
        self._max_completion_tokens = max_completion_tokens

    def build(self, context_pack: ContextPack, trace_id: str) -> GenerationRequest:
        """Build a generation request from a context pack."""
        # Extract and sanitize chunk contents
        chunk_ids: list[str] = []
        sanitized_contents: list[str] = []
        for item in context_pack.evidence:
            chunk_ids.append(item.chunk.chunk_id)
            sanitized_contents.append(sanitize_content(item.chunk.content))

        # Render the prompt
        context_block = format_context_block(chunk_ids, sanitized_contents)
        system_prompt, user_prompt = render_prompt(
            self._template, context_pack.query, context_block,
        )

        # Combine system + user for rendered_prompt (full text)
        rendered = f"{system_prompt}\n\n{user_prompt}"

        return GenerationRequest(
            rendered_prompt=rendered,
            system_prompt=system_prompt,
            context_chunk_ids=chunk_ids,
            query=context_pack.query,
            trace_id=trace_id,
            token_budget=self._max_completion_tokens,
        )
