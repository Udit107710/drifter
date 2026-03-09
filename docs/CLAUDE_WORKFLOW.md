# CLAUDE_WORKFLOW.md

## Expected working loop for Claude Code

For each prompt in `prompts/`:

1. Read the prompt fully.
2. **Before any work**, read these foundational documents:
   - `AGENTS.md` — operational rules and architecture principles
   - `AGENTS_ARCHITECTURE.md` — system structure and boundary rules
   - `AGENTS_EVALUATION.md` — evaluation layers and metrics
   - `AGENTS_WORKFLOW.md` — expected working loop and agent delegation guidance
   - `MASTER_PROMPT.md` — persona and working rules
3. Restate:
   - subsystem being built
   - architectural role
   - constraints
4. **Use specialised agents** for independent subtasks (see `AGENTS_WORKFLOW.md` for delegation guidance).
5. Show a design plan before coding.
6. Implement in small reviewable steps.
7. Add tests.
8. Update docs named in the prompt.
9. Summarize tradeoffs and remaining gaps.

## Important rule

If the repository is empty, create structure and docs first.
Do not jump into arbitrary implementation without establishing contracts and subsystem boundaries.
