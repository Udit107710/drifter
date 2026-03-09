"""Content sanitization for prompt injection prevention."""

from __future__ import annotations

import re

# Patterns that could be prompt injection attempts
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\|?(system|im_start|im_end)\|?>", re.IGNORECASE),
    re.compile(r"```\s*(system|instruction)", re.IGNORECASE),
]


def sanitize_content(text: str) -> str:
    """Sanitize retrieved content to prevent prompt injection.

    Replaces known injection patterns with [REDACTED].
    Does NOT modify the original content — returns a new string.
    """
    result = text
    for pattern in _INJECTION_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def sanitize_chunk_contents(chunks_content: list[str]) -> list[str]:
    """Sanitize a batch of chunk contents."""
    return [sanitize_content(c) for c in chunks_content]
