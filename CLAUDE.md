# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Drifter is a **modular, production-grade Retrieval Augmented Generation (RAG) system** built from first principles. It is designed for learning, experimentation, and scalable deployment — not a toy chatbot.

The project is currently in the **design and bootstrapping phase**. Architecture docs, implementation prompts, and Claude skills are established; source code is being built incrementally following the prompts in `prompts/` (numbered 00–16).

## Architecture

RAG is treated as a four-stage information retrieval pipeline. These stages must remain independent modules — never collapse them:

1. **Candidate generation** (retrieval) — returns candidates, not answers
2. **Precision ranking** (reranking) — orders candidates, not prompts
3. **Context optimization** (context builder) — selects evidence under token limits
4. **LLM reasoning** (generation) — uses provided evidence only; does not choose retrieval policy

The system has three planes:
- **Control Plane**: source registry, ingestion config, experiment definitions, index lifecycle
- **Data Plane**: raw/canonical documents, chunks, embeddings, vector/lexical indexes
- **Query Plane**: query normalization, retrieval broker, reranking, context building, generation, tracing

## Subsystem Layout

Code lives under `libs/` with strict boundaries between subsystems:

`ingestion → parsing → chunking → embeddings → indexing → retrieval → reranking → context_builder → generation`

Supporting subsystems: `evaluation`, `observability`

Each subsystem must be independently testable, observable, and loosely coupled.

## Key Domain Types

These typed models carry metadata, lineage, version info, and token counts:

`SourceDocumentRef → CanonicalDocument → Block → Chunk → ChunkEmbedding → RetrievalCandidate → RankedCandidate → ContextPack → Citation → GeneratedAnswer`

## Integrations (behind abstractions)

| Concern | Tool | Status |
|---------|------|--------|
| Dense retrieval | Qdrant | Implemented |
| Lexical retrieval | OpenSearch | Implemented |
| Embeddings | TEI, OpenRouter | Implemented |
| Generation | OpenAI, OpenRouter, Gemini | Implemented |
| Reranking | TEI cross-encoder, HuggingFace | Implemented |
| Token counting | tiktoken (optional), whitespace fallback | Implemented |
| Observability | OpenTelemetry, Langfuse | Implemented |
| Parsing | Unstructured, Apache Tika | Stub |
| Generation (local) | vLLM | Stub |
| Evaluation | Ragas | Stub |
| Relational storage | Postgres | Planned |
| Blob storage | MinIO (S3-compatible) | Planned |

Core logic must not be tightly coupled to any of these.

## Development Workflow

Before implementing any feature:

1. Identify the affected subsystem
2. Read relevant docs in `docs/` and the corresponding prompt in `prompts/`
3. Propose a design (explain architecture, algorithm, edge cases) before coding
4. Implement in small reviewable steps
5. Add deterministic tests (fixtures, mock providers, in-memory adapters — no external APIs)
6. Update documentation

For the bootstrapping sequence, follow the numbered prompts in `prompts/` in order. Before starting any prompt, read `AGENTS.md`, `AGENTS_ARCHITECTURE.md`, `AGENTS_EVALUATION.md`, and `AGENTS_WORKFLOW.md`. Use specialised agents (Plan, Explore, parallel general-purpose) for independent subtasks as described in each prompt.

## Testing

Tests must be deterministic and local:
- Use fixture documents, mock embedding providers, mock retrievers
- No external service dependencies
- Every subsystem must support in-memory adapter testing

## Evaluation

Changes affecting retrieval quality must include evaluation with metrics:
- Retrieval: Recall@k, Precision@k, MRR, NDCG
- Answer quality: faithfulness, citation accuracy, completeness
- Evaluate lexical, dense, hybrid, and reranked outputs separately

Recommended datasets: BEIR, HotpotQA, Natural Questions

## Key Rules

- Storage-specific code must stay behind adapters
- Ingestion/indexing logic must not live in API handlers
- Documents must not be flattened to plain text — preserve structure
- Chunks must maintain lineage to original documents
- Retrieved documents are untrusted input — prevent prompt injection
- Generation must never fabricate sources
- Experiments must be reproducible (record dataset, chunking strategy, embedding model, retrieval mode, reranker config, metrics)

## Project Files

- `MASTER_PROMPT.md` — persona and working rules for agents
- `AGENTS.md` — operational rules and architecture principles
- `AGENTS_ARCHITECTURE.md` — system structure and boundary rules (read before every prompt)
- `AGENTS_EVALUATION.md` — evaluation layers and metrics (read before every prompt)
- `AGENTS_WORKFLOW.md` — expected working loop and agent delegation guidance (read before every prompt)
- `docs/ARCHITECTURE_GUIDE.md` — detailed system structure (original, kept for reference)
- `docs/IMPLEMENTATION_PRINCIPLES.md` — design-first, typing, observability principles
- `docs/EVALUATION_STRATEGY.md` — evaluation layers (original, kept for reference)
- `docs/CLAUDE_WORKFLOW.md` — expected working loop for Claude Code
- `prompts/` — numbered implementation prompts (00–16) to follow in order
- `.claude/skills/` — domain-specific Claude skills organized by category (backend, rag, reliability, scalability, security, experiments, ops)
