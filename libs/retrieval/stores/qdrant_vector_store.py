"""Qdrant vector store adapter — re-exports from libs.adapters.qdrant.

This module is kept for backward compatibility.  New code should import
directly from :mod:`libs.adapters.qdrant`.
"""

from libs.adapters.qdrant.vector_store import QdrantVectorStore

__all__ = ["QdrantVectorStore"]
