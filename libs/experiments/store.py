"""Experiment store: persistence protocol and in-memory implementation."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from libs.experiments.models import ExperimentRun


@runtime_checkable
class ExperimentStore(Protocol):
    """Protocol for persisting and querying experiment runs."""

    def save(self, run: ExperimentRun) -> None:
        """Persist an experiment run."""
        ...

    def get(self, run_id: str) -> ExperimentRun | None:
        """Retrieve a run by ID, or None if not found."""
        ...

    def list_all(self) -> list[ExperimentRun]:
        """Return all runs, ordered by started_at descending."""
        ...

    def list_by_tag(self, tag: str) -> list[ExperimentRun]:
        """Return runs whose config contains the given tag."""
        ...

    def list_by_name(self, name: str) -> list[ExperimentRun]:
        """Return runs whose config name matches exactly."""
        ...


class InMemoryExperimentStore:
    """In-memory experiment store for testing and local development."""

    def __init__(self) -> None:
        self._runs: dict[str, ExperimentRun] = {}

    def save(self, run: ExperimentRun) -> None:
        self._runs[run.run_id] = run

    def get(self, run_id: str) -> ExperimentRun | None:
        return self._runs.get(run_id)

    def list_all(self) -> list[ExperimentRun]:
        return sorted(
            self._runs.values(),
            key=lambda r: r.started_at,
            reverse=True,
        )

    def list_by_tag(self, tag: str) -> list[ExperimentRun]:
        return [
            r for r in self.list_all()
            if tag in r.config.tags
        ]

    def list_by_name(self, name: str) -> list[ExperimentRun]:
        return [
            r for r in self.list_all()
            if r.config.name == name
        ]
