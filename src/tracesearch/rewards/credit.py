from __future__ import annotations

from tracesearch.agent.types import Step


def fatal_step_index(steps: list[Step], threshold: int = 3) -> int | None:
    """Return the first step in a consecutive tool-failure cascade."""
    failures = 0
    for index, step in enumerate(steps):
        failures = failures + 1 if step.failed else 0
        if failures >= threshold:
            return index - threshold + 1
    return None


def weighted_advantages(advantage: float, steps: list[Step]) -> list[float]:
    """Map trajectory advantage to steps using normalized contribution scores.

    A missing contribution is neutral (1.0). Training integrations may replace
    contributions with judge-produced retrieval and reasoning scores.
    """
    raw = [max(step.contribution, 0.0) if step.contribution is not None else 1.0 for step in steps]
    total = sum(raw)
    if not raw or total == 0:
        return [0.0] * len(raw)
    return [advantage * weight * len(raw) / total for weight in raw]
