"""Retrieval subsystem.

Contains two sub-packages:
- stores: VectorStore and LexicalStore protocols
- broker: RetrievalBroker that orchestrates dense, lexical, and hybrid retrieval

Boundary: consumes a query, produces list[RetrievalCandidate].
Returns candidates, never answers.
"""
