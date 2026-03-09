"""CLI exit codes."""

EXIT_SUCCESS = 0
EXIT_PARTIAL = 1
EXIT_FAILED = 2
EXIT_CONFIG_ERROR = 3
EXIT_INPUT_ERROR = 4


def outcome_to_exit_code(outcome: str) -> int:
    """Map a pipeline outcome string to an exit code."""
    return {
        "success": EXIT_SUCCESS,
        "partial": EXIT_PARTIAL,
        "no_results": EXIT_SUCCESS,
        "failed": EXIT_FAILED,
    }.get(outcome, EXIT_FAILED)
