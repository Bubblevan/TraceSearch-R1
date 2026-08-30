from __future__ import annotations

from tracesearch.agent.types import Trajectory


def trajectory_metrics(trajectory: Trajectory) -> dict[str, float]:
    """Metrics shared by offline fixtures and later online benchmark runners."""
    tool_steps = [step for step in trajectory.steps if step.action.kind != "answer"]
    failures = sum(step.failed for step in tool_steps)
    cited = bool(trajectory.answer and "http" in trajectory.answer)
    return {
        "tool_turns": float(len(tool_steps)),
        "tool_failure_rate": failures / len(tool_steps) if tool_steps else 0.0,
        "citation_present": float(cited),
        "answered": float(trajectory.answer is not None),
    }
