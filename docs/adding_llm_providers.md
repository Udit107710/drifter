# Adding New LLM Providers

Adding a new LLM (Claude, Gemini, ChatGPT, Ollama, etc.) to Drifter is straightforward. The generator subsystem is protocol-based — any class that implements two things works:

```python
@runtime_checkable
class Generator(Protocol):
    @property
    def generator_id(self) -> str: ...

    def generate(self, request: GenerationRequest) -> GeneratedAnswer: ...
```

That's it. No base class to inherit, no registration needed.

## Step-by-step

### 1. Add a config dataclass to `libs/adapters/config.py`

```python
@dataclass(frozen=True)
class AnthropicConfig:
    """Configuration for Anthropic Claude API."""

    api_key: str                          # required — no default for secrets
    model_id: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    temperature: float = 0.1
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("api_key must not be empty")
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if not (0 <= self.temperature <= 1):
            raise ValueError("temperature must be between 0 and 1")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")

    def __repr__(self) -> str:
        return _masked_repr(self, ("api_key",))
```

Follow the same pattern for OpenAI / Google — the fields will differ slightly but the structure is identical.

### 2. Add an env loader to `libs/adapters/env.py`

```python
def load_anthropic_config() -> AnthropicConfig | None:
    """Load Anthropic config from DRIFTER_ANTHROPIC_* env vars."""
    api_key = os.environ.get("DRIFTER_ANTHROPIC_API_KEY")
    if api_key is None:
        return None
    return AnthropicConfig(
        api_key=api_key,
        model_id=os.environ.get("DRIFTER_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        max_tokens=int(os.environ.get("DRIFTER_ANTHROPIC_MAX_TOKENS", "1024")),
        temperature=float(os.environ.get("DRIFTER_ANTHROPIC_TEMPERATURE", "0.1")),
        timeout_s=float(os.environ.get("DRIFTER_ANTHROPIC_TIMEOUT_S", "60.0")),
    )
```

### 3. Create the adapter under `libs/adapters/<provider>/`

```
libs/adapters/anthropic/
├── __init__.py
└── generator.py
```

**`libs/adapters/anthropic/__init__.py`**:
```python
"""Anthropic Claude generator adapter."""
from libs.adapters.anthropic.generator import AnthropicGenerator

__all__ = ["AnthropicGenerator"]
```

**`libs/adapters/anthropic/generator.py`**:
```python
"""Anthropic Claude generator adapter."""

from __future__ import annotations

from libs.adapters.config import AnthropicConfig
from libs.contracts.generation import Citation, GeneratedAnswer, TokenUsage
from libs.generation.models import GenerationRequest


class AnthropicGenerator:
    """Generator backed by the Anthropic Claude API.

    Satisfies the ``Generator`` protocol.
    """

    def __init__(self, config: AnthropicConfig) -> None:
        self._config = config
        self._client = None  # lazy init — no import at module level

    @property
    def generator_id(self) -> str:
        return f"anthropic:{self._config.model_id}"

    # -- Lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Initialise the Anthropic client."""
        import anthropic  # import inside method, not at module level

        self._client = anthropic.Anthropic(
            api_key=self._config.api_key,
            timeout=self._config.timeout_s,
        )

    def close(self) -> None:
        self._client = None

    def health_check(self) -> bool:
        return self._client is not None

    # -- Generator protocol --------------------------------------------------

    def generate(self, request: GenerationRequest) -> GeneratedAnswer:
        if self._client is None:
            raise RuntimeError("Call connect() before generate()")

        response = self._client.messages.create(
            model=self._config.model_id,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            system=request.system_prompt,
            messages=[{"role": "user", "content": request.rendered_prompt}],
        )

        answer_text = response.content[0].text
        usage = response.usage

        # Parse citations from the answer (your citation extraction logic here)
        citations = self._extract_citations(answer_text, request)

        return GeneratedAnswer(
            answer=answer_text,
            citations=citations,
            model_id=self._config.model_id,
            token_usage=TokenUsage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
            ),
            trace_id=request.trace_id,
        )

    def _extract_citations(
        self, answer: str, request: GenerationRequest
    ) -> list[Citation]:
        """Extract citations from the generated answer.

        Implement your citation parsing logic here. This depends on how
        you instruct the LLM to format citations in the system prompt.
        """
        return []  # TODO: implement citation extraction
```

Key rules to follow:
- **Never import the client library at module level** — use lazy imports inside `connect()` or `generate()` so the adapter file is importable without `pip install anthropic`.
- **Never log or repr the API key** — use `_masked_repr` in the config's `__repr__`.
- **Map token usage** — every provider reports tokens differently; map them to `TokenUsage(prompt_tokens, completion_tokens, total_tokens)`.

### 4. Wire it into the factory (`libs/adapters/factory.py`)

Update `create_generator` to accept your new config type:

```python
def create_generator(config: OllamaConfig | AnthropicConfig | None = None) -> Generator:
    if config is None:
        from libs.generation.mock_generator import MockGenerator
        return MockGenerator()

    if isinstance(config, OllamaConfig):
        from libs.adapters.ollama import OllamaGenerator
        return OllamaGenerator(config)

    if isinstance(config, AnthropicConfig):
        from libs.adapters.anthropic import AnthropicGenerator
        return AnthropicGenerator(config)

    raise TypeError(f"Unsupported generator config type: {type(config)}")
```

### 5. Add tests

Add to `tests/unit/test_adapter_stubs.py`:

```python
def test_anthropic_satisfies_generator_protocol(self) -> None:
    from libs.generation.protocols import Generator
    gen = AnthropicGenerator(AnthropicConfig(api_key="test"))
    assert isinstance(gen, Generator)

def test_anthropic_generate_before_connect_raises(self) -> None:
    gen = AnthropicGenerator(AnthropicConfig(api_key="test"))
    with pytest.raises(RuntimeError):
        gen.generate(...)  # not connected
```

### 6. Use it

```python
from libs.adapters.env import load_anthropic_config
from libs.adapters.factory import create_generator

generator = create_generator(load_anthropic_config())
# Returns MockGenerator if DRIFTER_ANTHROPIC_API_KEY is not set
# Returns AnthropicGenerator if it is
```

## Quick reference for other providers

The pattern is identical — only the client library and field mappings change:

| Provider | Config class | Client library | Token fields |
|----------|-------------|---------------|-------------|
| Anthropic Claude | `AnthropicConfig` | `anthropic` | `usage.input_tokens`, `usage.output_tokens` |
| OpenAI / ChatGPT | `OpenAIConfig` | `openai` | `usage.prompt_tokens`, `usage.completion_tokens` |
| Google Gemini | `GeminiConfig` | `google-genai` | `usage_metadata.prompt_token_count`, `candidates_token_count` |
| Ollama | `OllamaConfig` | `httpx` (native `/api/chat`) | `prompt_eval_count`, `eval_count` |

The `GenerationRequest` your `generate()` receives already contains:
- `rendered_prompt` — the full user prompt with context baked in
- `system_prompt` — the system instruction
- `context_chunk_ids` — which chunks were used (for citation tracking)
- `query` — the original user question
- `trace_id` — for observability
- `token_budget` — max tokens to respect

You just need to map these to your provider's API and map the response back to `GeneratedAnswer`.

## What NOT to do

- Don't import client libraries at module level — breaks imports when the library isn't installed
- Don't hardcode API keys — always load from config / env vars
- Don't skip `TokenUsage` — token accounting is required for cost tracking and context budgeting
- Don't fabricate citations — if you can't extract them from the response, return an empty list
- Don't add provider-specific logic to the pipeline — the pipeline only talks to `Generator`, never to a specific provider
