"""Run a local Level-B rollout dry run without performing an optimizer step."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from tracesearch.agent import LLMGenerationConfig, LLMPolicy, TransformersModelClient
from tracesearch.data.adapters import NQTaskAdapter
from tracesearch.data.io import load_corpus, write_json, write_jsonl, write_trajectories
from tracesearch.environment import Corpus, LocalSearchEnvironment, LocalSearchTool, LocalVisitTool
from tracesearch.evaluation.metrics import evaluate_metrics
from tracesearch.experiment.manifest import (
    ExperimentManifest,
    current_git_commit,
    current_git_dirty,
    source_tree_hash,
)
from tracesearch.training import (
    TrainingTrace,
    aggregate_group_diagnostics,
    compute_group_advantages,
    compute_outcome_reward,
    run_group_rollouts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/m1/dev.jsonl")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--corpus", default="data/m0/corpus.jsonl")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="runs/m1-level-b-rollout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--gen-top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    return parser


async def _run_groups(args: argparse.Namespace, dataset: Any, policy: LLMPolicy, search: Any, visit: Any) -> list[Any]:
    results = []
    for task in dataset.tasks:
        results.append(
            await run_group_rollouts(
                task,
                policy,
                search,
                visit,
                group_size=args.group_size,
                seed=args.seed,
                max_turns=args.max_turns,
                concurrency=args.concurrency,
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = NQTaskAdapter().load_jsonl(args.dataset, split=args.split)
    client = TransformersModelClient(args.model_path)
    generation = LLMGenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.gen_top_k,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )
    policy = LLMPolicy(
        client,
        model=Path(args.model_path).name,
        generation=generation,
        require_token_ids=True,
    )
    environment = LocalSearchEnvironment(Corpus(load_corpus(args.corpus)), top_k=args.top_k)
    results = asyncio.run(_run_groups(args, dataset, policy, LocalSearchTool(environment), LocalVisitTool(environment)))
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    trajectories = [record.trajectory for result in results for record in result.records if record.trajectory is not None]
    metrics = evaluate_metrics(
        dataset.tasks,
        trajectories,
        top_k=args.top_k,
        expected_rollouts=args.group_size,
        pass_k=args.group_size,
    )
    all_reward_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    for result in results:
        advantages = compute_group_advantages([record.reward.total for record in result.records])
        for record, advantage in zip(result.records, advantages):
            all_reward_rows.append(
                {
                    "task_id": result.task_id,
                    "sample_index": record.sample_index,
                    "rollout_id": record.rollout_id,
                    "advantage": advantage,
                    "reward": record.reward.to_dict(),
                    "status": record.status,
                    "error": record.error,
                }
            )
            if record.trajectory is None:
                trace_rows.append(
                    {
                        "task_id": result.task_id,
                        "sample_index": record.sample_index,
                        "rollout_id": record.rollout_id,
                        "status": record.status,
                        "trace": None,
                    }
                )
                continue
            for step in record.trajectory.steps:
                generation_record = step.metadata.get("generation_record")
                if generation_record is not None:
                    generation_rows.append(
                        {
                            "task_id": result.task_id,
                            "rollout_id": record.rollout_id,
                            "sample_index": record.sample_index,
                            **generation_record,
                        }
                    )
            observation_ids = {
                step.step_index: client.tokenize_text(step.observation)
                for step in record.trajectory.steps
                if step.is_tool_step
            }
            trace = TrainingTrace.from_trajectory(
                record.trajectory,
                observation_token_ids=observation_ids,
                policy_version=Path(args.model_path).name,
                reward_components=record.reward.components,
                total_reward=record.reward.total,
                group_advantage=advantage,
                sampling_metadata={"group_size": args.group_size, "seed": args.seed},
            )
            trace_rows.append(
                {
                    "task_id": result.task_id,
                    "sample_index": record.sample_index,
                    "rollout_id": record.rollout_id,
                    "status": record.status,
                    "trace": trace.to_dict(),
                }
            )
    manifest = ExperimentManifest(
        run_id=destination.name,
        stage="m1-level-b-rollout",
        seed=args.seed,
        git_commit=current_git_commit(),
        git_dirty=current_git_dirty(),
        source_tree_hash=source_tree_hash(),
        dataset=dataset.manifest_fields(),
        environment={"backend": "local_bm25", "corpus": args.corpus, "top_k": args.top_k},
        agent={"baseline": "prompted_agent", "max_turns": args.max_turns, "group_size": args.group_size},
        model={
            "model": Path(args.model_path).name,
            "checkpoint": args.model_path,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.gen_top_k,
            "repetition_penalty": args.repetition_penalty,
            "max_generation_tokens": args.max_tokens,
        },
    )
    write_json(destination / "manifest.json", manifest)
    write_trajectories(destination / "trajectories.jsonl", trajectories)
    write_jsonl(destination / "generation_records.jsonl", generation_rows)
    write_jsonl(destination / "training_traces.jsonl", trace_rows)
    write_jsonl(destination / "reward_breakdown.jsonl", all_reward_rows)
    write_json(destination / "group_diagnostics.json", {
        "aggregate": aggregate_group_diagnostics(results),
        "groups": [result.to_dict() for result in results],
    })
    write_json(destination / "metrics.json", metrics)
    (destination / "summary.md").write_text(_summary(manifest, metrics, results), encoding="utf-8")
    print(f"artifacts: {destination}")
    print(f"trajectories: {len(trajectories)}")
    print("optimizer_steps: 0")
    return 0


def _summary(manifest: ExperimentManifest, metrics: dict[str, Any], results: list[Any]) -> str:
    diagnostics = aggregate_group_diagnostics(results)
    return "\n".join(
        [
            f"# TraceSearch M1 Level-B rollout `{manifest.run_id}`",
            "",
            f"- Git commit: `{manifest.git_commit}`",
            f"- Git dirty: `{manifest.git_dirty}`",
            f"- Optimizer steps: `0`",
            "",
            "## Metrics",
            "",
            f"- Answer rate: {metrics['answer_rate']:.3f}",
            f"- Pass@1 / pass@k: {metrics['pass@1']:.3f} / {metrics['pass@k']:.3f}",
            f"- Missing trajectory slots: {metrics['missing_trajectory_count']}",
            "",
            "## Group diagnostics",
            "",
            f"- Zero-variance group rate: {diagnostics['zero_variance_group_rate']:.3f}",
            f"- All-zero / all-one group rate: {diagnostics['all_zero_group_rate']:.3f} / {diagnostics['all_one_group_rate']:.3f}",
            f"- Unique action sequences (mean): {diagnostics['unique_action_sequence_count']:.3f}",
            f"- Mean generated tokens: {diagnostics['mean_generated_token_count']:.3f}",
            "",
            "This is a training-ready dry run; it does not claim an optimizer or GRPO training result.",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
