"""OpenSearch vector store adapter — re-exports from libs.adapters.opensearch.

This module is kept for backward compatibility.  New code should import
directly from :mod:`libs.adapters.opensearch`.
"""

from libs.adapters.opensearch.vector_store import OpenSearchVectorStore

__all__ = ["OpenSearchVectorStore"]
