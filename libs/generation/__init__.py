"""Generation subsystem.

Responsibilities:
- Construct prompts from ContextPack and user query
- Call LLM via provider-agnostic Generator protocol
- Parse response into GeneratedAnswer with Citations
- Never fabricate sources

Boundary: consumes ContextPack + query, produces GeneratedAnswer.
Uses provided evidence only — does not choose retrieval policy.
"""

from libs.generation.citation_validator import DefaultCitationValidator
from libs.generation.mock_generator import MockGenerator
from libs.generation.models import (
    GenerationOutcome,
    GenerationRequest,
    GenerationResult,
    ValidationResult,
)
from libs.generation.prompt_templates import (
    CONCISE_TEMPLATE,
    DEFAULT_TEMPLATE,
    PromptTemplate,
    format_context_block,
    render_prompt,
)
from libs.generation.protocols import CitationValidator, Generator
from libs.generation.request_builder import GenerationRequestBuilder
from libs.generation.sanitizer import sanitize_chunk_contents, sanitize_content
from libs.generation.service import GenerationService

__all__ = [
    "CONCISE_TEMPLATE",
    "DEFAULT_TEMPLATE",
    "CitationValidator",
    "DefaultCitationValidator",
    "GenerationOutcome",
    "GenerationRequest",
    "GenerationRequestBuilder",
    "GenerationResult",
    "GenerationService",
    "Generator",
    "MockGenerator",
    "PromptTemplate",
    "ValidationResult",
    "format_context_block",
    "render_prompt",
    "sanitize_chunk_contents",
    "sanitize_content",
]
