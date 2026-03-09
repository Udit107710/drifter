# AGENTS_WORKFLOW.md

## Expected working loop for Claude Code

For each prompt in `prompts/`:

1. Read the prompt fully.
2. **Before any work**, read these foundational documents:
   - `AGENTS.md` — operational rules and architecture principles
   - `AGENTS_ARCHITECTURE.md` — system structure and boundary rules
   - `AGENTS_EVALUATION.md` — evaluation layers and metrics
   - `AGENTS_WORKFLOW.md` — this workflow document
   - `MASTER_PROMPT.md` — persona and working rules
3. Restate:
   - subsystem being built
   - architectural role
   - constraints
4. **Use specialised agents** for independent subtasks:
   - Delegate research, file exploration, and codebase searches to Explore agents
   - Delegate implementation planning to Plan agents
   - Run tests and validations via dedicated test-runner agents
   - Use parallel agents for independent work streams (e.g., code + docs + tests)
5. Show a design plan before coding.
6. Implement in small reviewable steps.
7. Add tests.
8. Update docs named in the prompt.
9. Summarize tradeoffs and remaining gaps.

## Important rule

If the repository is empty, create structure and docs first.
Do not jump into arbitrary implementation without establishing contracts and subsystem boundaries.

## Agent delegation guidance

- **Explore agent**: Use for researching existing code, finding patterns, understanding current state
- **Plan agent**: Use for designing implementation strategy before coding
- **General-purpose agent**: Use for complex multi-step tasks that can run independently
- **Parallel execution**: Launch multiple agents simultaneously for independent work (e.g., writing tests in one agent while writing docs in another)
- **Isolation**: Use worktree isolation when agents need to make experimental changes without affecting the main working directory
