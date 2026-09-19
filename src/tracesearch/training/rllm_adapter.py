"""Thin optional rLLM conversion boundary; rLLM internals stay external."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tracesearch.data.schema import Task, Trajectory
from tracesearch.training.reward import RewardBreakdown


@dataclass(frozen=True)
class RLLMTask:
    task_id: str
    prompt: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RLLMEpisode:
    task_id: str
    rollout_id: str
    sample_index: int
    steps: tuple[dict[str, Any], ...]
    answer: str | None
    metadata: dict[str, Any]


def task_to_rllm(task: Task) -> RLLMTask:
    policy_task = task.policy_view()
    return RLLMTask(task.task_id, policy_task.question, dict(policy_task.metadata))


def trajectory_to_rllm(trajectory: Trajectory) -> RLLMEpisode:
    return RLLMEpisode(
        task_id=trajectory.task_id or "",
        rollout_id=trajectory.rollout_id or "",
        sample_index=trajectory.sample_index,
        steps=tuple(step.to_dict() for step in trajectory.steps),
        answer=trajectory.answer,
        metadata=dict(trajectory.metadata),
    )


def reward_to_rllm(reward: RewardBreakdown) -> dict[str, Any]:
    return {"reward": reward.total, "reward_components": reward.to_dict()}


class RLLMAdapter:
    """Dependency-free adapter object for the selected external rLLM version."""

    def task(self, task: Task) -> RLLMTask:
        return task_to_rllm(task)

    def episode(self, trajectory: Trajectory) -> RLLMEpisode:
        return trajectory_to_rllm(trajectory)

    def reward(self, reward: RewardBreakdown) -> dict[str, Any]:
        return reward_to_rllm(reward)
