"""M1 evaluation/training artifact writer."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tracesearch.data.io import write_json, write_jsonl, write_trajectories
from tracesearch.data.schema import Task, Trajectory
from tracesearch.evaluation.metrics import evaluate_metrics
from tracesearch.experiment.manifest import ExperimentManifest
from tracesearch.training.reward import compute_outcome_reward


def write_m1_artifacts(
    output_dir: str | Path,
    manifest: ExperimentManifest,
    tasks: Iterable[Task],
    trajectories: Iterable[Trajectory],
    *,
    training_metrics: Iterable[dict[str, Any]] = (),
    reward_breakdowns: Iterable[dict[str, Any]] | None = None,
    top_k: int = 5,
    expected_rollouts: int | None = None,
    pass_k: int | None = None,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    task_values = list(tasks)
    trajectory_values = list(trajectories)
    metrics = evaluate_metrics(
        task_values,
        trajectory_values,
        top_k=top_k,
        expected_rollouts=expected_rollouts,
        pass_k=pass_k,
    )
    write_json(destination / "manifest.json", manifest)
    write_trajectories(destination / "trajectories.jsonl", trajectory_values)
    write_json(destination / "metrics.json", metrics)
    if reward_breakdowns is None:
        by_task = {task.task_id: task for task in task_values}
        reward_values = [
            {
                "task_id": trajectory.task_id,
                "rollout_id": trajectory.rollout_id,
                "sample_index": trajectory.sample_index,
                "reward": compute_outcome_reward(by_task[trajectory.task_id or ""], trajectory).to_dict(),
            }
            for trajectory in trajectory_values
            if trajectory.task_id in by_task
        ]
    else:
        reward_values = list(reward_breakdowns)
    write_jsonl(destination / "reward_breakdown.jsonl", reward_values)
    write_jsonl(destination / "training_metrics.jsonl", list(training_metrics))
    (destination / "summary.md").write_text(_summary(manifest, metrics), encoding="utf-8")
    return destination


def _summary(manifest: ExperimentManifest, metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"# TraceSearch M1 run `{manifest.run_id}`",
            "",
            f"- Stage: `{manifest.stage}`",
            f"- Seed: `{manifest.seed}`",
            f"- Git commit: `{manifest.git_commit}`",
            f"- Git dirty: `{manifest.git_dirty}`",
            "",
            "## Evaluation metrics",
            "",
            f"- Tasks: {metrics['task_count']}",
            f"- Answer rate: {metrics['answer_rate']:.3f}",
            f"- Normalized exact match: {metrics['normalized_exact_match']:.3f}",
            f"- Pass@1 / pass@k: {metrics['pass@1']:.3f} / {metrics['pass@k']:.3f}",
            f"- Missing trajectory slots: {metrics['missing_trajectory_count']}",
            f"- Tool failure rate: {metrics['tool_failure_rate']:.3f}",
            "",
            "Training metrics are separate from evaluator exact-match variance.",
            "",
        ]
    )
