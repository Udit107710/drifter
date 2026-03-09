# EVALUATION_STRATEGY.md

## Evaluation layers

### A. Ingestion / indexing correctness
Validate:
- source discovery
- version handling
- tombstones
- parser outputs
- chunk lineage
- metadata propagation
- index write completeness

### B. Retrieval quality
Minimum metrics:
- Recall@k
- Precision@k
- MRR
- NDCG (when graded relevance exists)

Evaluate separately for:
- lexical only
- dense only
- hybrid pre-rerank
- reranked output

### C. Answer quality
Track:
- faithfulness / groundedness
- citation accuracy
- completeness
- unsupported claims

### D. System quality
Track:
- latency per stage
- failure rate
- stale data lag
- index freshness
- deterministic local replay

## Recommended datasets
- BEIR for retrieval benchmarking
- HotpotQA for multi-hop evidence retrieval
- Natural Questions for QA
- custom local gold sets for subsystem debugging
