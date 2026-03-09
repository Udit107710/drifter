"""Generation protocols: contracts every generator and validator must satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from libs.contracts.context import ContextPack
from libs.contracts.generation import GeneratedAnswer

if TYPE_CHECKING:
    from libs.generation.models import GenerationRequest, ValidationResult


@runtime_checkable
class Generator(Protocol):
    """Protocol for LLM-backed answer generation."""

    @property
    def generator_id(self) -> str: ...

    def generate(self, request: GenerationRequest) -> GeneratedAnswer: ...


@runtime_checkable
class CitationValidator(Protocol):
    """Protocol for validating citations against the source context."""

    def validate(self, answer: GeneratedAnswer, context: ContextPack) -> ValidationResult: ...
