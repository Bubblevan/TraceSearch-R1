"""Run the offline M0 fixture from a clean Python environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from tracesearch.environment import FaultSchedule, FailureInjector
from tracesearch.experiment.runner import run_from_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TraceSearch-R1 M0 offline.")
    parser.add_argument("--tasks", type=Path, default=Path("data/m0/tasks.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("data/m0/corpus.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--pass-k", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("runs/m0-smoke"))
    parser.add_argument("--fault-profile", choices=("none", "reproducible"), default="none")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    schedule = None
    injector = None
    if args.fault_profile == "reproducible":
        schedule = FaultSchedule({(1, "visit"): "timeout"})
        injector = FailureInjector(seed=args.seed, irrelevant_result_rate=0.0)
    manifest, trajectories = run_from_files(
        args.tasks,
        args.corpus,
        args.output,
        seed=args.seed,
        top_k=args.top_k,
        max_turns=args.max_turns,
        rollout_count=args.rollouts,
        pass_k=args.pass_k,
        fault_schedule=schedule,
        failure_injector=injector,
    )
    import json

    metrics = json.loads((args.output / "metrics.json").read_text(encoding="utf-8"))
    print(f"artifacts: {args.output}")
    print(
        "metrics: "
        f"tasks={metrics['task_count']} "
        f"answer_rate={metrics['answer_rate']:.3f} "
        f"exact_match={metrics['normalized_exact_match']:.3f} "
        f"pass@k={metrics['pass@k']:.3f} "
        f"missing={metrics['missing_trajectory_count']} "
        f"tool_failure_rate={metrics['tool_failure_rate']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
