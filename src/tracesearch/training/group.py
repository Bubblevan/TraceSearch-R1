"""Concurrent group rollouts with explicit missing/cancelled outcomes."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import statistics
from dataclasses import dataclass
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
class GroupRolloutResult:
    task_id: str
    requested_group_size: int
    records: tuple[RolloutRecord, ...]
    group_reward_mean: float
    group_reward_std: float
    zero_variance_group: bool

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
    return GroupRolloutResult(
        task_id=task.task_id,
        requested_group_size=group_size,
        records=records,
        group_reward_mean=mean,
        group_reward_std=std,
        zero_variance_group=std == 0.0,
    )


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
