"""OpenSearch lexical store adapter — re-exports from libs.adapters.opensearch.

This module is kept for backward compatibility.  New code should import
directly from :mod:`libs.adapters.opensearch`.
"""

from libs.adapters.opensearch.lexical_store import OpenSearchLexicalStore

__all__ = ["OpenSearchLexicalStore"]
