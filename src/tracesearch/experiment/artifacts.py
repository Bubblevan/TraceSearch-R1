"""Write and summarize the immutable files produced by a run."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterable

from tracesearch.data.io import write_json, write_trajectories
from tracesearch.data.schema import Task, Trajectory
from tracesearch.evaluation.metrics import evaluate_metrics
from tracesearch.experiment.manifest import ExperimentManifest


def write_run_artifacts(
    output_dir: str | Path,
    manifest: ExperimentManifest,
    tasks: Iterable[Task],
    trajectories: Iterable[Trajectory],
    *,
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
    summary = _summary(manifest, metrics, destination)
    (destination / "summary.md").write_text(summary, encoding="utf-8")
    return destination


def _summary(manifest: ExperimentManifest, metrics: dict[str, object], destination: Path) -> str:
    return "\n".join(
        [
            f"# TraceSearch M0 run `{manifest.run_id}`",
            "",
            f"- Stage: `{manifest.stage}`",
            f"- Seed: `{manifest.seed}`",
            f"- Artifacts: `{destination}`",
            "",
            "## Metrics",
            "",
            f"- Tasks: {metrics['task_count']} (evaluated: {metrics['evaluated_task_count']})",
            f"- Answer rate: {metrics['answer_rate']:.3f}",
            f"- Normalized exact match: {metrics['normalized_exact_match']:.3f}",
            f"- Pass@1 / pass@k: {metrics['pass@1']:.3f} / {metrics['pass@k']:.3f}",
            f"- Missing trajectory slots: {metrics['missing_trajectory_count']}",
            f"- Group reward variance: {metrics['group_reward_variance']:.3f}",
            f"- Search recall@k: {metrics['search_recall_at_k']:.3f}",
            f"- Visited gold evidence recall: {metrics['visited_gold_evidence_recall']:.3f}",
            f"- Average tool turns: {metrics['avg_tool_turns']:.3f}",
            f"- Tool success rate: {metrics['tool_success_rate']:.3f}",
            f"- Injected failure count: {metrics['injected_failure_count']}",
            "",
            "Metrics are generated from `tasks + trajectories`; this fixture is not a benchmark result.",
            "",
        ]
    )
