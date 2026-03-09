"""Tests for token counter implementations."""

from __future__ import annotations

import pytest

from libs.chunking.protocols import TokenCounter
from libs.chunking.token_counter import WhitespaceTokenCounter


class TestWhitespaceTokenCounter:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(WhitespaceTokenCounter(), TokenCounter)

    def test_counts_words(self) -> None:
        counter = WhitespaceTokenCounter()
        assert counter.count("hello world foo") == 3

    def test_empty_string(self) -> None:
        counter = WhitespaceTokenCounter()
        assert counter.count("") == 0

    def test_whitespace_only(self) -> None:
        counter = WhitespaceTokenCounter()
        assert counter.count("   ") == 0


def _make_tiktoken_counter():
    """Create a TiktokenTokenCounter, skipping the test if tiktoken is unavailable."""
    from libs.chunking.token_counter import TiktokenTokenCounter

    try:
        return TiktokenTokenCounter()
    except ImportError:
        pytest.skip("tiktoken not installed")


class TestTiktokenTokenCounter:
    def test_counts_subword_tokens(self) -> None:
        counter = _make_tiktoken_counter()
        # "hello world" is 2 tokens in most BPE encodings
        count = counter.count("hello world")
        assert count >= 2
        # Subword tokenization: word count != token count for complex text
        complex_text = "uncharacteristically"
        assert counter.count(complex_text) >= 1

    def test_empty_string(self) -> None:
        counter = _make_tiktoken_counter()
        assert counter.count("") == 0

    def test_satisfies_protocol(self) -> None:
        counter = _make_tiktoken_counter()
        assert isinstance(counter, TokenCounter)
