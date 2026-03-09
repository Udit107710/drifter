"""Token counter implementations."""

from __future__ import annotations


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
