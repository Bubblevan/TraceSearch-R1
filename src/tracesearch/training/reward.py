"""M1's intentionally simple rule-based outcome reward."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tracesearch.data.schema import Task, TerminationReason, Trajectory
from tracesearch.evaluation.metrics import normalized_exact_match


@dataclass(frozen=True)
class RewardBreakdown:
    answer_correct: float
    format_valid: float
    total: float
    components: dict[str, float] = field(default_factory=dict)
    status: str = "completed"

    def __post_init__(self) -> None:
        for name, value in (("answer_correct", self.answer_correct), ("format_valid", self.format_valid), ("total", self.total)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_correct": self.answer_correct,
            "format_valid": self.format_valid,
            "total": self.total,
            "components": dict(self.components),
            "status": self.status,
        }


def compute_outcome_reward(task: Task, trajectory: Trajectory | None) -> RewardBreakdown:
    """Return zero for missing/malformed trajectories and EM for valid answers."""

    if trajectory is None:
        return RewardBreakdown(0.0, 0.0, 0.0, {"answer_correct": 0.0, "format_valid": 0.0}, "missing")
    format_valid = float(
        trajectory.termination_reason is TerminationReason.ANSWER
        and isinstance(trajectory.answer, str)
        and bool(trajectory.answer.strip())
    )
    answer_correct = float(format_valid and normalized_exact_match(trajectory.answer, task.answers))
    total = answer_correct
    return RewardBreakdown(
        answer_correct=answer_correct,
        format_valid=format_valid,
        total=total,
        components={"answer_correct": answer_correct, "format_valid": format_valid, "total": total},
        status="completed" if format_valid else "invalid_or_no_answer",
    )


reward_from_trajectory = compute_outcome_reward
