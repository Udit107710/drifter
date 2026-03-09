# IMPLEMENTATION_PRINCIPLES.md

## Principles for this repository

### 1. Design before implementation
Before writing code for a subsystem:
- state the problem
- define inputs and outputs
- explain the architecture
- propose algorithm(s)
- identify edge cases

### 2. Strong internal models
Use explicit typed models for:
- documents
- blocks
- chunks
- embeddings
- retrieval candidates
- ranked candidates
- citations
- evaluation cases

### 3. Deterministic local mode
Every subsystem must be testable without external services by using:
- local fixtures
- mock providers
- in-memory adapters

### 4. Observability by default
Emit structured events and traces for:
- ingest
- parse
- chunk
- embed
- index
- retrieve
- rerank
- pack context
- generate

### 5. Evaluation as a first-class feature
Do not defer evaluation until the end.
Build retrieval and answer evaluation into the project early.

### 6. Avoid framework lock-in
Frameworks may help at the edges, but core logic must remain understandable from this repository's own code.
