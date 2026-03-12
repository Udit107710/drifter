# libs/generation/

Stage 4 of the query plane. Constructs prompts from evidence, calls the LLM, and validates citations.

## Boundary

- **Consumes:** ContextPack + trace_id
- **Produces:** GenerationResult (with GeneratedAnswer, citations, validation)
- **Rules:** Uses provided evidence only. Never fabricates sources. Retrieved chunk text is untrusted input.

## Service

```python
class GenerationService:
    def __init__(
        self,
        generator: Generator,
        request_builder: GenerationRequestBuilder | None = None,
        citation_validator: CitationValidator | None = None,
        validate_citations: bool = True,
    ) -> None: ...

    def run(self, context_pack: ContextPack, trace_id: str) -> GenerationResult: ...
```

Pipeline: validate context → build request (render prompt) → generate → validate citations → return.

## Protocols

```python
class Generator(Protocol):
    def generator_id(self) -> str: ...
    def generate(self, request: GenerationRequest) -> GeneratedAnswer: ...

class CitationValidator(Protocol):
    def validate(self, answer: GeneratedAnswer, context: ContextPack) -> ValidationResult: ...
```

## Implementations

| Class | Purpose |
|-------|---------|
| `MockGenerator` | Deterministic answers with one citation per chunk. For testing. |
| `OllamaGenerator` | Real generation via Ollama `/api/chat` (NDJSON streaming). In `libs/adapters/ollama/`. |
| `VllmGenerator` | Real generation via vLLM `/v1/chat/completions` (SSE streaming). In `libs/adapters/vllm/`. |
| `OpenAIGenerator` | Real generation via OpenAI/OpenRouter Chat Completions API. In `libs/adapters/openai/`. |
| `GeminiGenerator` | Real generation via Google Gemini API. In `libs/adapters/gemini/`. |
| `DefaultCitationValidator` | Validates citations reference chunks in the context |
| `GenerationRequestBuilder` | Renders ContextPack into a prompt using templates |

## Prompt Templates

`prompt_templates.py` defines `PromptTemplate` and built-in templates:

| Template | Purpose |
|----------|---------|
| `DEFAULT_TEMPLATE` | Standard RAG prompt with evidence blocks and citation instructions |
| `CONCISE_TEMPLATE` | Shorter prompt for concise answers |

```python
class PromptTemplate:
    system_prompt: str
    user_template: str  # Has {context} and {query} placeholders
```

Helper functions:
- `format_context_block(items)` — Renders ContextItems into numbered evidence blocks
- `render_prompt(template, context_pack)` — Produces the final prompt string

## Citation Validation

`DefaultCitationValidator` checks that:
- Every citation references a chunk that exists in the context
- Reports orphaned citations (referencing non-existent chunks)
- Reports uncited chunks (in context but not cited)

## Security

`sanitizer.py` ensures retrieved chunk text is never mixed directly into system prompts:
- `sanitize_content(text)` — Strips potentially dangerous content from chunk text
- `sanitize_chunk_contents(items)` — Applies sanitization to all context items

## Key Types

| Type | Purpose |
|------|---------|
| `GenerationRequest` | Rendered prompt + system prompt + context chunk IDs + query |
| `GenerationResult` | answer + outcome + generator_id + latency + validation + errors |
| `GenerationOutcome` | SUCCESS, EMPTY_CONTEXT, GENERATION_FAILED, VALIDATION_FAILED |
| `ValidationResult` | is_valid + errors + orphaned_citations + uncited_chunks |
| `GeneratedAnswer` | Answer text + citations + model ID + token usage |
| `Citation` | Claim + chunk_id + chunk_content + source_id + confidence |

## Testing

Uses `MockGenerator` which produces deterministic answers with predictable citations. No LLM calls required.
