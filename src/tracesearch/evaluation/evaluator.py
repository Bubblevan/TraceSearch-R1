"""Small evaluator façade for callers that prefer an object API."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tracesearch.data.schema import Task, Trajectory
from tracesearch.evaluation.metrics import evaluate_metrics


class Evaluator:
    def __init__(self, *, top_k: int = 5, expected_rollouts: int | None = None, pass_k: int | None = None):
        self.top_k = top_k
        self.expected_rollouts = expected_rollouts
        self.pass_k = pass_k

    def evaluate(self, tasks: Iterable[Task], trajectories: Iterable[Trajectory]) -> dict[str, Any]:
        return evaluate_metrics(
            tasks,
            trajectories,
            top_k=self.top_k,
            expected_rollouts=self.expected_rollouts,
            pass_k=self.pass_k,
        )
