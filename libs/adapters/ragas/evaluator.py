"""Ragas evaluation — stub adapter."""

from __future__ import annotations

from libs.adapters.config import RagasConfig


class RagasAnswerEvaluator:
    """Answer evaluator backed by Ragas.

    This is a stub adapter.  Install the ``ragas`` package and implement
    the evaluation logic to use this evaluator in production.
    """

    def __init__(self, config: RagasConfig) -> None:
        self._config = config

    # -- lifecycle -------------------------------------------------------------

    def connect(self) -> None:
        """Initialise the Ragas evaluation pipeline."""

    def close(self) -> None:
        """Release resources."""

    def health_check(self) -> bool:
        """Return ``False`` — stub is not connected to a real service."""
        return False

    # -- evaluation ------------------------------------------------------------

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> dict[str, float]:
        """Evaluate an answer against provided contexts.

        Returns a dict mapping metric names to scores.
        """
        raise NotImplementedError(
            "Install ragas and implement RagasAnswerEvaluator"
        )
