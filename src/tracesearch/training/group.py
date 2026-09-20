"""Concurrent group rollouts with explicit missing/cancelled outcomes."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

from tracesearch.agent import SearchAgent
from tracesearch.data.schema import Task, Trajectory
from tracesearch.training.reward import RewardBreakdown, compute_outcome_reward


@dataclass(frozen=True)
class RolloutRecord:
    sample_index: int
    rollout_id: str
    trajectory: Trajectory | None
    reward: RewardBreakdown
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "rollout_id": self.rollout_id,
            "trajectory": self.trajectory.to_dict() if self.trajectory is not None else None,
            "reward": self.reward.to_dict(),
            "status": self.status,
            "error": self.error,
        }


@dataclass(frozen=True)
class GroupDiagnostics:
    group_reward_mean: float
    group_reward_std: float
    zero_variance_group: bool
    all_zero_group: bool
    all_one_group: bool
    unique_action_sequence_count: int
    unique_final_answer_count: int
    mean_trajectory_length: float
    mean_generated_token_count: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_reward_mean": self.group_reward_mean,
            "group_reward_std": self.group_reward_std,
            "zero_variance_group": self.zero_variance_group,
            "all_zero_group": self.all_zero_group,
            "all_one_group": self.all_one_group,
            "unique_action_sequence_count": self.unique_action_sequence_count,
            "unique_final_answer_count": self.unique_final_answer_count,
            "mean_trajectory_length": self.mean_trajectory_length,
            "mean_generated_token_count": self.mean_generated_token_count,
        }


@dataclass(frozen=True)
class GroupRolloutResult:
    task_id: str
    requested_group_size: int
    records: tuple[RolloutRecord, ...]
    group_reward_mean: float
    group_reward_std: float
    zero_variance_group: bool
    diagnostics: GroupDiagnostics = field(default_factory=lambda: GroupDiagnostics(0.0, 0.0, True, True, False, 0, 0, 0.0, 0.0))

    @property
    def completed_rollouts(self) -> int:
        return sum(record.trajectory is not None for record in self.records)

    @property
    def missing_rollouts(self) -> int:
        return self.requested_group_size - self.completed_rollouts

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "requested_group_size": self.requested_group_size,
            "completed_rollouts": self.completed_rollouts,
            "missing_rollouts": self.missing_rollouts,
            "records": [record.to_dict() for record in self.records],
            "group_reward_mean": self.group_reward_mean,
            "group_reward_std": self.group_reward_std,
            "zero_variance_group": self.zero_variance_group,
            "diagnostics": self.diagnostics.to_dict(),
        }


async def run_group_rollouts(
    task: Task,
    policy_factory: Callable[[int], Any] | Any,
    search: Any,
    visit: Any,
    *,
    group_size: int,
    seed: int | None = None,
    max_turns: int = 6,
    max_searches: int | None = None,
    max_visits: int | None = None,
    tool_budget: int | None = None,
    concurrency: int | None = None,
) -> GroupRolloutResult:
    """Run exactly ``group_size`` requested slots and retain failures as records."""

    if group_size < 1:
        raise ValueError("group_size must be positive")
    if concurrency is not None and concurrency < 1:
        raise ValueError("concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency) if concurrency else None

    async def one(sample_index: int) -> RolloutRecord:
        rollout_id = _rollout_id(task.task_id, seed, sample_index)
        try:
            if semaphore is not None:
                await semaphore.acquire()
            try:
                policy = _make_policy(policy_factory, sample_index)
                agent = SearchAgent(
                    policy,
                    search,
                    visit,
                    max_turns=max_turns,
                    max_searches=max_searches,
                    max_visits=max_visits,
                    tool_budget=tool_budget,
                    seed=seed,
                    rollout_id=rollout_id,
                    sample_index=sample_index,
                )
                trajectory = await agent.run_async(task)
            finally:
                if semaphore is not None:
                    semaphore.release()
            reward = compute_outcome_reward(task, trajectory)
            return RolloutRecord(sample_index, trajectory.rollout_id or rollout_id, trajectory, reward, "completed")
        except asyncio.CancelledError:
            reward = compute_outcome_reward(task, None)
            return RolloutRecord(sample_index, rollout_id, None, reward, "cancelled", "rollout task cancelled")
        except Exception as exc:
            reward = compute_outcome_reward(task, None)
            return RolloutRecord(sample_index, rollout_id, None, reward, "failed", f"{type(exc).__name__}: {exc}")

    records = tuple(sorted(await asyncio.gather(*(one(index) for index in range(group_size))), key=lambda item: item.sample_index))
    rewards = [record.reward.total for record in records]
    mean = float(statistics.mean(rewards))
    std = float(statistics.pstdev(rewards))
    trajectories = [record.trajectory for record in records if record.trajectory is not None]
    action_sequences = {
        tuple((step.action.kind.value, step.action.value) for step in trajectory.steps)
        for trajectory in trajectories
    }
    final_answers = {trajectory.answer for trajectory in trajectories if trajectory.answer is not None}
    trajectory_lengths = [len(trajectory.steps) for trajectory in trajectories]
    generated_token_counts = [
        sum(
            len(step.metadata.get("response_token_ids", ()))
            for step in trajectory.steps
            if isinstance(step.metadata.get("response_token_ids", ()), (list, tuple))
        )
        for trajectory in trajectories
    ]
    diagnostics = GroupDiagnostics(
        group_reward_mean=mean,
        group_reward_std=std,
        zero_variance_group=std == 0.0,
        all_zero_group=all(reward == 0.0 for reward in rewards),
        all_one_group=all(reward == 1.0 for reward in rewards),
        unique_action_sequence_count=len(action_sequences),
        unique_final_answer_count=len(final_answers),
        mean_trajectory_length=float(statistics.mean(trajectory_lengths)) if trajectory_lengths else 0.0,
        mean_generated_token_count=float(statistics.mean(generated_token_counts)) if generated_token_counts else 0.0,
    )
    return GroupRolloutResult(
        task_id=task.task_id,
        requested_group_size=group_size,
        records=records,
        group_reward_mean=mean,
        group_reward_std=std,
        zero_variance_group=std == 0.0,
        diagnostics=diagnostics,
    )


def aggregate_group_diagnostics(results: list[GroupRolloutResult] | tuple[GroupRolloutResult, ...]) -> dict[str, Any]:
    """Aggregate group diversity without changing any reward values."""

    if not results:
        return {
            "group_count": 0,
            "group_reward_mean": 0.0,
            "group_reward_std": 0.0,
            "zero_variance_group_rate": 0.0,
            "all_zero_group_rate": 0.0,
            "all_one_group_rate": 0.0,
            "unique_action_sequence_count": 0.0,
            "unique_final_answer_count": 0.0,
            "mean_trajectory_length": 0.0,
            "mean_generated_token_count": 0.0,
        }
    diagnostics = [result.diagnostics for result in results]
    count = len(diagnostics)
    return {
        "group_count": count,
        "group_reward_mean": float(statistics.mean(item.group_reward_mean for item in diagnostics)),
        "group_reward_std": float(statistics.mean(item.group_reward_std for item in diagnostics)),
        "zero_variance_group_rate": sum(item.zero_variance_group for item in diagnostics) / count,
        "all_zero_group_rate": sum(item.all_zero_group for item in diagnostics) / count,
        "all_one_group_rate": sum(item.all_one_group for item in diagnostics) / count,
        "unique_action_sequence_count": float(statistics.mean(item.unique_action_sequence_count for item in diagnostics)),
        "unique_final_answer_count": float(statistics.mean(item.unique_final_answer_count for item in diagnostics)),
        "mean_trajectory_length": float(statistics.mean(item.mean_trajectory_length for item in diagnostics)),
        "mean_generated_token_count": float(statistics.mean(item.mean_generated_token_count for item in diagnostics)),
    }


def _make_policy(factory: Callable[[int], Any] | Any, sample_index: int) -> Any:
    if not callable(factory) or hasattr(factory, "act"):
        return factory
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        parameters = {}
    return factory(sample_index) if parameters else factory()


def _rollout_id(task_id: str, seed: int | None, sample_index: int) -> str:
    material = f"{task_id}|{seed if seed is not None else 0}|{sample_index}".encode("utf-8")
    return "rollout-" + hashlib.sha256(material).hexdigest()[:24]
