# AGENTS.md

This file defines the **operational rules for AI agents working in this repository**.

Agents must read this file before modifying the codebase.

The goal of this repository is to build a **modular, inspectable, and production-grade Retrieval Augmented Generation (RAG) system** designed for learning, experimentation, and scalable deployment.

Agents must prioritize **correct architecture and reproducibility over rapid implementation**.

---

# Repository Mission

This project builds a **full RAG system from first principles**.

Objectives:

1. Understand how modern RAG systems work internally
2. Build a modular backend architecture
3. Support experimentation with retrieval techniques
4. Maintain reproducibility and observability
5. Ensure the system is scalable and production-ready

Agents should treat this repository as **infrastructure software**, not a prototype application.

---

# RAG Architecture Principles

The RAG pipeline must preserve the following stages:

candidate generation → precision ranking → context optimization → LLM reasoning

These correspond to:

1. Retrieval
2. Reranking
3. Context Building
4. Generation

Agents **must never collapse these stages**.

Incorrect example:

retrieve → directly call LLM

Correct example:

retrieve → rerank → build context → generate grounded answer

---

# System Architecture

The system is divided into three architectural planes.

## Control Plane
Responsible for system configuration and orchestration.

Examples:
- source registry
- ingestion configuration
- evaluation datasets
- experiment configuration
- index lifecycle management

## Data Plane
Responsible for storing and processing data.

Examples:
- raw documents
- canonical documents
- chunks
- embeddings
- vector indexes
- lexical indexes

## Query Plane
Responsible for answering user queries.

Components:
- query normalization
- retrieval broker
- reranking
- context building
- generation
- answer validation

---

# Subsystem Boundaries

Agents must maintain strict boundaries between subsystems.

## Ingestion (libs/ingestion)
- source discovery
- crawl state
- document versioning
- change detection
- tombstone deletion

## Parsing (libs/parsing)
- canonical document representation
- block extraction
- structural preservation

Documents must **not** be flattened into plain text.

## Chunking (libs/chunking)
- chunking strategies
- token boundaries
- parent-child chunking
- metadata propagation

Chunks must maintain lineage to original documents.

## Embeddings (libs/embeddings)
- embedding provider abstraction
- embedding batching
- embedding versioning

Embedding code must not directly depend on a specific provider.

## Indexing (libs/indexing)
- writing embeddings to vector stores
- writing metadata to search indexes
- maintaining index consistency

## Retrieval (libs/retrieval)
- dense retrieval
- lexical retrieval
- hybrid retrieval

Retrieval returns **candidate documents**, not answers.

## Reranking (libs/reranking)
- ranking candidate results
- cross-encoder scoring
- feature-based ranking

## Context Builder (libs/context_builder)
- token budgeting
- evidence selection
- redundancy removal
- diversity optimization

## Generation (libs/generation)
- prompt construction
- citation alignment
- grounded answer generation

Generation must never fabricate sources.

## Evaluation (libs/evaluation)
- retrieval benchmarks
- experiment tracking
- metric calculation

---

# Coding Standards

## Design Before Code
Before implementing code:

1. Identify the subsystem
2. Explain the design
3. Outline the algorithm
4. Identify edge cases

Then implement.

## Strong Typing

Important entities:

- SourceDocumentRef
- CanonicalDocument
- Block
- Chunk
- ChunkEmbedding
- RetrievalCandidate
- RankedCandidate
- ContextPack
- Citation
- GeneratedAnswer

These models must include:

- metadata
- lineage
- version information
- token counts

## Deterministic Testing

Tests must use:

- fixture documents
- mock embedding providers
- mock retrievers

Tests must not rely on external APIs.

---

# Observability

Tracing should cover:

- ingestion
- parsing
- chunking
- embedding
- indexing
- retrieval
- reranking
- context building
- generation

Use **OpenTelemetry**.

---

# Evaluation Requirements

Changes affecting retrieval quality must include evaluation.

Metrics:

- Recall@k
- MRR
- NDCG

Answer quality metrics:

- faithfulness
- citation accuracy

Datasets may include:

- BEIR
- HotpotQA
- Natural Questions

---

# Experimentation

Experiments must record:

- dataset
- chunking strategy
- embedding model
- retrieval mode
- reranker configuration
- metrics

Experiments must be reproducible.

---

# Security Rules

Retrieved documents must be treated as **untrusted input**.

Prevent prompt injection attacks.

Documents must never override system prompts.

---

# Agent Workflow

When implementing changes:

1. Read relevant architecture docs
2. Identify subsystem
3. Propose design
4. Implement code
5. Add tests
6. Update documentation

Agents must not implement features without updating documentation.

---

# Anti-Patterns

Avoid:

- mixing retrieval and generation logic
- giant utility modules
- skipping evaluation
- framework-coupled core logic
- silent failures

---

# Final Principle

Prioritize:

clarity • modularity • inspectability • reproducibility