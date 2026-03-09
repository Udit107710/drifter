"""Retrieval broker protocols: query embedding and normalization."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class QueryEmbedder(Protocol):
    """Embeds a query string into a dense vector for retrieval."""

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class QueryNormalizer(Protocol):
    """Preprocesses query text before retrieval."""

    def normalize(self, raw_query: str) -> str: ...


class PassthroughNormalizer:
    """Default normalizer that returns query text unchanged."""

    def normalize(self, raw_query: str) -> str:
        return raw_query
