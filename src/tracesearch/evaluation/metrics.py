"""Deterministic metrics computed only from tasks and saved trajectories."""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Iterable
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


def evaluate_metrics(tasks: Iterable[Task], trajectories: Iterable[Trajectory], *, top_k: int = 5) -> dict[str, Any]:
    """Evaluate a saved run without consulting environment or policy state."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    task_values = list(tasks)
    trajectory_values = list(trajectories)
    by_id = {trajectory.task_id: trajectory for trajectory in trajectory_values if trajectory.task_id}
    pairs = [(task, by_id.get(task.task_id)) for task in task_values]
    pairs = [(task, trajectory) for task, trajectory in pairs if trajectory is not None]
    if not pairs:
        return _empty_metrics(len(task_values))

    answered = [trajectory.answer is not None and bool(trajectory.answer.strip()) for _, trajectory in pairs]
    exact = [normalized_exact_match(trajectory.answer, task.answers) for task, trajectory in pairs]
    tool_turns = [trajectory.tool_turns for _, trajectory in pairs]
    search_calls = [trajectory.search_calls for _, trajectory in pairs]
    visit_calls = [trajectory.visit_calls for _, trajectory in pairs]
    all_tool_steps = [step for _, trajectory in pairs for step in trajectory.steps if step.is_tool_step]
    successful = sum(not step.failed for step in all_tool_steps)
    failed = sum(step.failed for step in all_tool_steps)
    search_recalls: list[float] = []
    visited_recalls: list[float] = []
    failure_by_type: Counter[str] = Counter()
    latencies: list[float] = []
    injected_failure_count = 0
    for task, trajectory in pairs:
        gold = set(task.gold_evidence_ids)
        retrieved: set[str] = set()
        visited: set[str] = set()
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
        # No-search calibration tasks are intentionally not applicable to
        # evidence recall, so they do not dilute retrieval metrics.
        if gold:
            search_recalls.append(len(retrieved & gold) / len(gold))
            visited_recalls.append(len(visited & gold) / len(gold))

    termination_histogram = Counter(trajectory.termination_reason.value for _, trajectory in pairs)
    return {
        "task_count": len(task_values),
        "evaluated_task_count": len(pairs),
        "answer_rate": _mean(answered),
        "normalized_exact_match": _mean(exact),
        "exact_match_rate": _mean(exact),
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


def _empty_metrics(task_count: int) -> dict[str, Any]:
    return {
        "task_count": task_count,
        "evaluated_task_count": 0,
        "answer_rate": 0.0,
        "normalized_exact_match": 0.0,
        "exact_match_rate": 0.0,
        "avg_tool_turns": 0.0,
        "avg_search_calls": 0.0,
        "avg_visit_calls": 0.0,
        "tool_success_rate": 0.0,
        "tool_failure_rate": 0.0,
        "search_recall_at_k": 0.0,
        "visited_gold_evidence_recall": 0.0,
        "termination_histogram": {},
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
