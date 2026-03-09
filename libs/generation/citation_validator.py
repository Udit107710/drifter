"""Citation alignment and validation."""

from __future__ import annotations

from libs.contracts.context import ContextPack
from libs.contracts.generation import GeneratedAnswer
from libs.generation.models import ValidationResult


class DefaultCitationValidator:
    """Validates that citations align with the provided context.

    Checks:
    - No orphaned citations (referencing chunks not in context)
    - Reports uncited chunks (informational, not an error)
    """

    def validate(
        self, answer: GeneratedAnswer, context: ContextPack,
    ) -> ValidationResult:
        context_chunk_ids = set(context.chunk_ids)
        cited_chunk_ids = {c.chunk_id for c in answer.citations}

        orphaned = sorted(cited_chunk_ids - context_chunk_ids)
        uncited = sorted(context_chunk_ids - cited_chunk_ids)

        errors: list[str] = []
        if orphaned:
            errors.append(
                f"Citations reference chunks not in context: {orphaned}"
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            orphaned_citations=orphaned,
            uncited_chunks=uncited,
        )
