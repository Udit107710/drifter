# libs/context_builder/

Stage 3 of the query plane. Selects evidence from ranked candidates under a token budget.

## Boundary

- **Consumes:** list[RankedCandidate] + query + token_budget
- **Produces:** ContextPack (evidence within budget, diversity score, exclusion records)
- **Rule:** Decides final evidence. Does not call the LLM.

## Service

```python
class ContextBuilderService:
    def __init__(self, builder: ContextBuilder) -> None: ...
    def run(self, candidates: list[RankedCandidate], query: str, token_budget: int) -> BuilderResult: ...
```

## Protocol

```python
class ContextBuilder(Protocol):
    def build(self, candidates: list[RankedCandidate], query: str, token_budget: int) -> BuilderResult: ...
```

## Implementations

| Builder | Description |
|---------|-------------|
| `GreedyContextBuilder` | Greedy selection: take highest-ranked candidates until budget is exhausted. Deduplicates, tracks exclusions. |
| `DiversityAwareBuilder` | Diversity-weighted selection: penalizes candidates from already-represented sources. |

## GreedyContextBuilder Pipeline

1. Deduplicate candidates by chunk content hash
2. For each candidate (in rank order):
   - Count tokens
   - If fits within remaining budget → include
   - If doesn't fit → record exclusion reason
3. Compute diversity score (unique sources / total evidence)
4. Build ContextPack

## Key Types

| Type | Purpose |
|------|---------|
| `ContextPack` | Evidence list + total tokens + budget + diversity score |
| `ContextItem` | Chunk + rank + token count + selection reason |
| `BuilderResult` | context_pack + outcome + exclusions + dedup_removed + latency |
| `BuilderConfig` | token_budget, diversity_weight, max_chunks, deduplicate |
| `ExclusionRecord` | chunk_id + reason + token_count |
| `BuilderOutcome` | SUCCESS, EMPTY_CANDIDATES, BUDGET_EXHAUSTED, FAILED |

## Deduplication

`dedup.py` removes candidates with duplicate chunk content hashes. This prevents the same passage from appearing multiple times when retrieved by both dense and lexical stores.

## Diversity Score

Measures source diversity of selected evidence: `unique_source_count / total_evidence_count`. Higher is better. The `DiversityAwareBuilder` uses this to actively promote diverse sources during selection.

## Testing

Uses `WhitespaceTokenCounter` for token counting. Deterministic behavior with predictable chunk token counts.
