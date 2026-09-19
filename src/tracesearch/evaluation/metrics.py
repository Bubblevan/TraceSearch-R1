"""Deterministic single- and multi-rollout metrics.

Missing rollout slots are explicit failures for answer/pass metrics. Tool
rates only use observed tool calls because a missing trajectory has no tool
call to classify.
"""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from tracesearch.agent.types import ActionKind, Trajectory
from tracesearch.data.schema import Task


def normalize_answer(value: str) -> str:
    """Casefold, trim/collapse whitespace, and remove basic punctuation."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = "".join(" " if unicodedata.category(char).startswith(("P", "S")) else char for char in normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def normalized_exact_match(answer: str | None, aliases: Iterable[str]) -> bool:
    if answer is None:
        return False
    candidate = normalize_answer(answer)
    return any(candidate == normalize_answer(alias) for alias in aliases)


def trajectory_metrics(trajectory: Trajectory) -> dict[str, float]:
    """Backward-compatible per-trajectory metrics from the original scaffold."""

    tool_steps = [step for step in trajectory.steps if step.is_tool_step]
    failures = sum(step.failed for step in tool_steps)
    cited = bool(trajectory.answer and ("http" in trajectory.answer or "[" in trajectory.answer))
    return {
        "tool_turns": float(len(tool_steps)),
        "tool_failure_rate": failures / len(tool_steps) if tool_steps else 0.0,
        "citation_present": float(cited),
        "answered": float(trajectory.answer is not None),
    }


def evaluate_metrics(
    tasks: Iterable[Task],
    trajectories: Iterable[Trajectory] | Mapping[str, Iterable[Trajectory]],
    *,
    top_k: int = 5,
    expected_rollouts: int | None = None,
    pass_k: int | None = None,
) -> dict[str, Any]:
    """Evaluate task groups without consulting environment or policy state.

    ``trajectories`` may be a flat iterable or a ``task_id -> iterable``
    mapping. Groups are ordered by ``sample_index`` and then rollout identity.
    The evaluator pads each task to ``expected_rollouts`` with missing slots;
    those slots score zero for answer, exact-match, pass, and exact-match
    variance calculations. If omitted, the largest observed sample index plus
    one is used (or one when the input is empty), preserving sparse slots.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if expected_rollouts is not None and expected_rollouts < 1:
        raise ValueError("expected_rollouts must be positive")
    if pass_k is not None and pass_k < 1:
        raise ValueError("pass_k must be positive")

    task_values = list(tasks)
    groups = _group_trajectories(trajectories)
    observed_max_index = max(
        (
            trajectory.sample_index + 1
            for task in task_values
            for trajectory in groups.get(task.task_id, [])
        ),
        default=0,
    )
    rollout_width = expected_rollouts or max(1, observed_max_index)
    pass_width = min(pass_k or rollout_width, rollout_width)
    slots_by_task = {
        task.task_id: _slots(groups.get(task.task_id, []), rollout_width) for task in task_values
    }
    observed_rollouts = sum(len(groups.get(task.task_id, [])) for task in task_values)
    missing_rollouts = sum(
        max(0, rollout_width - len(groups.get(task.task_id, []))) for task in task_values
    )
    observed_tasks = sum(bool(groups.get(task.task_id)) for task in task_values)

    if not task_values:
        return _empty_metrics(0, rollout_width)

    rewards: dict[str, list[float]] = {}
    for task in task_values:
        rewards[task.task_id] = [
            float(trajectory is not None and normalized_exact_match(trajectory.answer, task.answers))
            for trajectory in slots_by_task[task.task_id]
        ]

    all_slots = [trajectory for task in task_values for trajectory in slots_by_task[task.task_id]]
    observed_slot_values = [trajectory for trajectory in all_slots if trajectory is not None]
    all_tool_steps = [
        step for trajectory in observed_slot_values for step in trajectory.steps if step.is_tool_step
    ]
    successful = sum(not step.failed for step in all_tool_steps)
    failed = sum(step.failed for step in all_tool_steps)
    answered = [float(trajectory is not None and bool(trajectory.answer and trajectory.answer.strip())) for trajectory in all_slots]
    exact = [reward for reward_values in rewards.values() for reward in reward_values]
    tool_turns = [trajectory.tool_turns if trajectory else 0 for trajectory in all_slots]
    search_calls = [trajectory.search_calls if trajectory else 0 for trajectory in all_slots]
    visit_calls = [trajectory.visit_calls if trajectory else 0 for trajectory in all_slots]

    search_recalls: list[float] = []
    visited_recalls: list[float] = []
    failure_by_type: Counter[str] = Counter()
    latencies: list[float] = []
    injected_failure_count = 0
    for task in task_values:
        gold = set(task.gold_evidence_ids)
        for trajectory in slots_by_task[task.task_id]:
            retrieved: set[str] = set()
            visited: set[str] = set()
            if trajectory is not None:
                for step in trajectory.steps:
                    if not step.is_tool_step or step.tool_result is None:
                        continue
                    result = step.tool_result
                    latencies.append(result.latency_ms)
                    if result.metadata.get("injected_fault"):
                        injected_failure_count += 1
                    if not result.ok and result.error_type is not None:
                        failure_by_type[result.error_type.value] += 1
                    if step.action.kind is ActionKind.SEARCH:
                        retrieved.update(item.doc_id for item in result.evidence if item.rank <= top_k)
                    elif step.action.kind is ActionKind.VISIT:
                        visited.update(item.doc_id for item in result.evidence)
                        document = result.metadata.get("document")
                        if isinstance(document, dict) and document.get("doc_id"):
                            visited.add(str(document["doc_id"]))
            if gold:
                search_recalls.append(len(retrieved & gold) / len(gold))
                visited_recalls.append(len(visited & gold) / len(gold))

    pass_at_1 = statistics.mean(reward_values[0] for reward_values in rewards.values())
    pass_at_k = statistics.mean(
        float(any(reward_values[:pass_width])) for reward_values in rewards.values()
    )
    group_variances = [statistics.pvariance(reward_values) for reward_values in rewards.values()]
    termination_histogram = Counter(
        trajectory.termination_reason.value
        for trajectory in observed_slot_values
    )
    if missing_rollouts:
        termination_histogram["missing"] += missing_rollouts

    return {
        "task_count": len(task_values),
        "evaluated_task_count": observed_tasks,
        "observed_task_count": observed_tasks,
        "observed_rollout_count": observed_rollouts,
        "expected_rollout_count": rollout_width,
        "missing_task_count": len(task_values) - observed_tasks,
        "missing_trajectory_count": missing_rollouts,
        "answer_rate": _mean(answered),
        "normalized_exact_match": _mean(exact),
        "exact_match_rate": _mean(exact),
        "pass@1": pass_at_1,
        "pass@k": pass_at_k,
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k,
        "group_exact_match_variance": _mean(group_variances),
        "avg_tool_turns": _mean(tool_turns),
        "avg_search_calls": _mean(search_calls),
        "avg_visit_calls": _mean(visit_calls),
        "tool_success_rate": successful / len(all_tool_steps) if all_tool_steps else 0.0,
        "tool_failure_rate": failed / len(all_tool_steps) if all_tool_steps else 0.0,
        "search_recall_at_k": _mean(search_recalls),
        "visited_gold_evidence_recall": _mean(visited_recalls),
        "termination_histogram": dict(sorted(termination_histogram.items())),
        "injected_failure_count": injected_failure_count,
        "failure_count_by_type": dict(sorted(failure_by_type.items())),
        "tool_latency_p50_ms": _percentile(latencies, 0.50),
        "tool_latency_p95_ms": _percentile(latencies, 0.95),
    }


def _group_trajectories(
    trajectories: Iterable[Trajectory] | Mapping[str, Iterable[Trajectory]],
) -> dict[str, list[Trajectory]]:
    if isinstance(trajectories, Mapping):
        grouped: dict[str, list[Trajectory]] = {}
        for task_id, values in trajectories.items():
            if isinstance(values, Trajectory):
                grouped[str(task_id)] = [values]
            else:
                grouped[str(task_id)] = list(values)
    else:
        grouped = defaultdict(list)
        for trajectory in trajectories:
            if trajectory.task_id is not None:
                grouped[trajectory.task_id].append(trajectory)
    for task_id in grouped:
        grouped[task_id].sort(key=lambda item: (item.sample_index, item.rollout_id, item.trajectory_id))
    return dict(grouped)


def _slots(values: list[Trajectory], width: int) -> list[Trajectory | None]:
    slots: list[Trajectory | None] = [None] * width
    seen: set[int] = set()
    for trajectory in values:
        sample_index = trajectory.sample_index
        if sample_index < 0 or sample_index >= width:
            raise ValueError(
                f"sample_index {sample_index} for task {trajectory.task_id!r} "
                f"is outside rollout width {width}"
            )
        if sample_index in seen:
            raise ValueError(
                f"duplicate sample_index {sample_index} for task {trajectory.task_id!r}"
            )
        seen.add(sample_index)
        slots[sample_index] = trajectory
    return slots


def _empty_metrics(task_count: int, rollout_width: int) -> dict[str, Any]:
    missing = task_count * rollout_width
    return {
        "task_count": task_count,
        "evaluated_task_count": 0,
        "observed_task_count": 0,
        "observed_rollout_count": 0,
        "expected_rollout_count": rollout_width,
        "missing_task_count": task_count,
        "missing_trajectory_count": missing,
        "answer_rate": 0.0,
        "normalized_exact_match": 0.0,
        "exact_match_rate": 0.0,
        "pass@1": 0.0,
        "pass@k": 0.0,
        "pass_at_1": 0.0,
        "pass_at_k": 0.0,
        "group_exact_match_variance": 0.0,
        "avg_tool_turns": 0.0,
        "avg_search_calls": 0.0,
        "avg_visit_calls": 0.0,
        "tool_success_rate": 0.0,
        "tool_failure_rate": 0.0,
        "search_recall_at_k": 0.0,
        "visited_gold_evidence_recall": 0.0,
        "termination_histogram": {"missing": missing} if missing else {},
        "injected_failure_count": 0,
        "failure_count_by_type": {},
        "tool_latency_p50_ms": 0.0,
        "tool_latency_p95_ms": 0.0,
    }


def _mean(values: Iterable[bool | int | float]) -> float:
    values = list(values)
    return float(statistics.mean(values)) if values else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * fraction))))
    return float(values[index])
