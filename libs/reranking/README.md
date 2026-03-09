# libs/reranking/

Stage 2 of the query plane. Converts high-recall retrieval candidates into a precision-ranked shortlist.

## Boundary

- **Consumes:** list[RetrievalCandidate] + RetrievalQuery
- **Produces:** list[RankedCandidate] (with rerank scores)
- **Rule:** Orders candidates, not prompts.

## Service

```python
class RerankerService:
    def __init__(self, reranker: Reranker, top_n: int = 0) -> None: ...
    def run(self, candidates: list[RetrievalCandidate], query: RetrievalQuery) -> RerankerResult: ...
```

Pipeline: validate inputs → rerank → truncate to top_n → build result.

## Protocol

```python
class Reranker(Protocol):
    def reranker_id(self) -> str: ...
    def rerank(self, candidates: list[RetrievalCandidate], query: RetrievalQuery) -> list[RankedCandidate]: ...
```

## Implementations

| Reranker | ID | Description |
|----------|-----|-------------|
| `FeatureBasedReranker` | `feature-based-v1` | Multi-signal scoring (retrieval score, authority, freshness, diversity). No external model needed. |
| `PassthroughReranker` | `passthrough` | Returns candidates in original order (for testing/baselines) |
| `CrossEncoderReranker` | `cross-encoder:{model}` | Stub for cross-encoder models |
| `TeiCrossEncoderReranker` | TEI-based | Real cross-encoder via TEI (in `libs/adapters/tei/`) |

## Feature-Based Reranker

`FeatureBasedReranker` combines multiple signals with configurable `FeatureWeights`:
- Retrieval score (from the store)
- Authority score (from source metadata)
- Freshness (time decay from source freshness hint)
- Source diversity bonus

Default weights work without any external model, making it the primary reranker for local development.

## Converters

`converters.py` bridges the broker and reranker:

```python
def fused_list_to_retrieval_candidates(fused: list[FusedCandidate]) -> list[RetrievalCandidate]
```

Converts `FusedCandidate` objects (from RRF fusion) into `RetrievalCandidate` objects (for reranking).

## Result Types

| Type | Fields |
|------|--------|
| `RerankerResult` | query, ranked_candidates, count, outcome, reranker_id, latency, errors, debug |
| `RerankerOutcome` | SUCCESS, NO_CANDIDATES, FAILED |

## Testing

Uses `FeatureBasedReranker` and `PassthroughReranker`. No external model dependencies.
