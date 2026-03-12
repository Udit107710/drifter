"""Generation service orchestrator."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from libs.contracts.context import ContextPack
from libs.generation.citation_validator import DefaultCitationValidator
from libs.generation.models import GenerationOutcome, GenerationResult, ValidationResult
from libs.generation.protocols import CitationValidator, Generator
from libs.generation.request_builder import GenerationRequestBuilder

logger = logging.getLogger(__name__)


class GenerationService:
    """Orchestrates the generation pipeline.

    Steps:
    1. Validate input (empty context → EMPTY_CONTEXT)
    2. Build GenerationRequest from ContextPack
    3. Call generator.generate()
    4. Validate citations
    5. Return GenerationResult with timing and debug
    """

    def __init__(
        self,
        generator: Generator,
        request_builder: GenerationRequestBuilder | None = None,
        citation_validator: CitationValidator | None = None,
        validate_citations: bool = True,
    ) -> None:
        self._generator = generator
        self._builder = request_builder or GenerationRequestBuilder()
        self._validator = citation_validator or DefaultCitationValidator()
        self._validate = validate_citations

    def run(
        self,
        context_pack: ContextPack,
        trace_id: str,
        on_token: Callable[[str, bool], None] | None = None,
    ) -> GenerationResult:
        start = time.monotonic()
        debug: dict[str, Any] = {
            "generator_id": self._generator.generator_id,
            "trace_id": trace_id,
            "context_chunks": len(context_pack.evidence),
            "context_total_tokens": context_pack.total_tokens,
        }

        logger.debug(
            "generation: entry generator_id=%s context_chunks=%d",
            self._generator.generator_id, len(context_pack.evidence),
        )

        # Empty context check
        if not context_pack.evidence:
            logger.warning("generation: empty context trace_id=%s", trace_id)
            return self._build_result(
                answer=None,
                outcome=GenerationOutcome.EMPTY_CONTEXT,
                start=start,
                debug=debug,
            )

        # Build request
        try:
            request = self._builder.build(context_pack, trace_id)
        except Exception as exc:
            logger.error("generation: request build failed trace_id=%s: %s", trace_id, exc)
            return self._build_result(
                answer=None,
                outcome=GenerationOutcome.GENERATION_FAILED,
                start=start,
                debug=debug,
                errors=[f"Request build failed: {exc} (trace_id={trace_id})"],
            )

        # Generate — pass on_token if the generator supports streaming
        try:
            if on_token is not None and hasattr(self._generator, "generate"):
                import inspect
                sig = inspect.signature(self._generator.generate)
                if "on_token" in sig.parameters:
                    answer = self._generator.generate(request, on_token=on_token)
                else:
                    answer = self._generator.generate(request)
            else:
                answer = self._generator.generate(request)
        except Exception as exc:
            logger.error(
                "generation: failed generator=%s trace_id=%s: %s",
                self._generator.generator_id, trace_id, exc,
            )
            return self._build_result(
                answer=None,
                outcome=GenerationOutcome.GENERATION_FAILED,
                start=start,
                debug=debug,
                errors=[
                    f"Generation failed: {exc} "
                    f"(generator={self._generator.generator_id}, trace_id={trace_id})"
                ],
            )

        # Validate citations
        validation: ValidationResult | None = None
        if self._validate:
            try:
                validation = self._validator.validate(answer, context_pack)
            except Exception as exc:
                return self._build_result(
                    answer=answer,
                    outcome=GenerationOutcome.VALIDATION_FAILED,
                    start=start,
                    debug=debug,
                    errors=[f"Citation validation failed: {exc} (trace_id={trace_id})"],
                )

        debug["citation_count"] = len(answer.citations)
        debug["model_id"] = answer.model_id
        debug["prompt_tokens"] = answer.token_usage.prompt_tokens
        debug["completion_tokens"] = answer.token_usage.completion_tokens

        outcome = GenerationOutcome.SUCCESS
        errors: list[str] = []
        if validation and not validation.is_valid:
            outcome = GenerationOutcome.VALIDATION_FAILED
            errors = validation.errors
            logger.warning("generation: validation failed trace_id=%s errors=%s", trace_id, errors)

        if outcome == GenerationOutcome.SUCCESS:
            logger.info(
                "generation: success model_id=%s citations=%d tokens=%d",
                answer.model_id, len(answer.citations),
                answer.token_usage.prompt_tokens + answer.token_usage.completion_tokens,
            )

        return self._build_result(
            answer=answer,
            outcome=outcome,
            start=start,
            debug=debug,
            validation=validation,
            errors=errors,
        )

    def _build_result(
        self,
        *,
        answer: Any,
        outcome: GenerationOutcome,
        start: float,
        debug: dict[str, Any],
        validation: ValidationResult | None = None,
        errors: list[str] | None = None,
    ) -> GenerationResult:
        elapsed = (time.monotonic() - start) * 1000
        return GenerationResult(
            answer=answer,
            outcome=outcome,
            generator_id=self._generator.generator_id,
            total_latency_ms=elapsed,
            completed_at=datetime.now(UTC),
            validation=validation,
            errors=errors or [],
            debug=debug,
        )
