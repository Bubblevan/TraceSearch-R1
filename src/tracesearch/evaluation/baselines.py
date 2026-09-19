"""Controlled M1 baseline runners sharing the TraceSearch environment."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterable
from typing import Any

from tracesearch.agent import DirectAnswerPolicy, LLMPolicy, RetrieveOncePolicy, SearchAgent
from tracesearch.data.schema import Task, Trajectory
from tracesearch.evaluation.metrics import evaluate_metrics


async def run_baseline_rollouts(
    tasks: Iterable[Task],
    policy_factory: Callable[[Task], Any] | Any,
    search: Any,
    visit: Any,
    *,
    seed: int | None = None,
    max_turns: int = 6,
    rollout_count: int = 1,
) -> list[Trajectory]:
    if rollout_count < 1:
        raise ValueError("rollout_count must be positive")
    trajectories: list[Trajectory] = []
    for task in tasks:
        for sample_index in range(rollout_count):
            policy = _make_policy(policy_factory, task)
            agent = SearchAgent(
                policy,
                search,
                visit,
                max_turns=max_turns,
                seed=seed,
                sample_index=sample_index,
            )
            trajectories.append(await agent.run_async(task))
    return trajectories


def evaluate_baseline(
    tasks: Iterable[Task],
    trajectories: Iterable[Trajectory],
    *,
    top_k: int = 5,
    rollout_count: int = 1,
    pass_k: int | None = None,
) -> dict[str, Any]:
    return evaluate_metrics(
        tasks,
        trajectories,
        top_k=top_k,
        expected_rollouts=rollout_count,
        pass_k=pass_k,
    )


def run_baseline_rollouts_sync(*args: Any, **kwargs: Any) -> list[Trajectory]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_baseline_rollouts(*args, **kwargs))
    raise RuntimeError("run_baseline_rollouts_sync cannot run inside an active event loop")


def make_direct_answer_policy(client: Any, *, model: str, **kwargs: Any) -> DirectAnswerPolicy:
    return DirectAnswerPolicy(client, model=model, **kwargs)


def make_retrieve_once_policy(client: Any, *, model: str, **kwargs: Any) -> RetrieveOncePolicy:
    return RetrieveOncePolicy(client, model=model, **kwargs)


def make_prompted_agent_policy(client: Any, *, model: str, **kwargs: Any) -> LLMPolicy:
    return LLMPolicy(client, model=model, **kwargs)


def _make_policy(factory: Callable[[Task], Any] | Any, task: Task) -> Any:
    if not callable(factory) or hasattr(factory, "act"):
        return factory
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        parameters = {}
    return factory(task) if parameters else factory()
