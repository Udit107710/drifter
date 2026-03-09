# AGENTS_ARCHITECTURE.md

## High-level system structure

This repository builds a RAG platform with three planes:

### Control Plane
Owns configuration and lifecycle:
- source registry
- ingestion scheduling/crawl state
- experiment definitions
- evaluation datasets
- model/provider config
- index lifecycle management

### Data Plane
Owns persisted knowledge and indexes:
- raw document store (production: MinIO/S3; local: in-memory/filesystem)
- canonical document store (production: Postgres; local: in-memory)
- chunk store (production: Postgres; local: in-memory)
- crawl state store (production: Postgres; local: in-memory)
- vector index (production: Qdrant; local: in-memory)
- lexical index (production: OpenSearch; local: in-memory)
- metadata / lineage store (production: Postgres; local: in-memory)
- experiment config + evaluation results (production: Postgres; local: in-memory)

### Query Plane
Owns online request handling:
- query normalization
- retrieval broker
- reranking
- context builder
- generation
- citation/verification
- tracing / diagnostics

## Canonical subsystem list

1. ingestion
2. parsing
3. chunking
4. embeddings
5. indexing
6. retrieval stores
7. retrieval broker
8. reranking
9. context builder
10. generation
11. observability
12. evaluation
13. experiments
14. integration adapters

## Non-negotiable boundary rules

- retrieval returns candidates, not answers
- reranking orders candidates, not prompts
- context builder decides final evidence under token limits
- generation uses provided evidence; it does not choose retrieval policy
- ingestion/indexing logic must not live in API handlers
- storage-specific code must remain behind adapters
- error classification logic lives in `libs/resilience.py` — subsystems must not duplicate transient/permanent detection
- adapter lifecycle checks use `Connectable`/`HealthCheckable` protocols (`libs/adapters/protocols.py`), not `hasattr`
