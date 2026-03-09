"""Token counter implementations."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class WhitespaceTokenCounter:
    """Counts tokens by splitting on whitespace.

    A simple, zero-dependency token counter suitable for testing and
    approximate budgeting. For production use, swap in a model-specific
    tokenizer behind the TokenCounter protocol.
    """

    def count(self, text: str) -> int:
        """Return the number of whitespace-delimited tokens in *text*.

        Returns 0 for empty or whitespace-only strings.
        """
        return len(text.split())


class TiktokenTokenCounter:
    """Counts tokens using tiktoken (OpenAI's BPE tokenizer).

    Provides accurate token counts for GPT-family models. Falls back
    to cl100k_base encoding when a model-specific encoding is not found.

    Args:
        model: Model name for encoding lookup (e.g. "gpt-4o", "gpt-3.5-turbo").
               Defaults to "gpt-4o".
    """

    def __init__(self, model: str = "gpt-4o") -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "tiktoken is required for TiktokenTokenCounter. "
                "Install it with: pip install tiktoken"
            ) from exc

        try:
            self._encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.warning(
                "No tiktoken encoding for model %r, falling back to cl100k_base",
                model,
            )
            self._encoding = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        """Return the number of BPE tokens in *text*."""
        return len(self._encoding.encode(text))
