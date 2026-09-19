"""Offline M0 integration runner."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tracesearch.agent import OracleFixturePolicy, SearchAgent
from tracesearch.data.hashing import hash_corpus
from tracesearch.data.io import load_corpus, load_tasks
from tracesearch.data.schema import Task, Trajectory
from tracesearch.environment import Corpus, FaultSchedule, FailureInjector, LocalSearchEnvironment, LocalSearchTool, LocalVisitTool
from tracesearch.experiment.artifacts import write_run_artifacts
from tracesearch.experiment.manifest import (
    ExperimentManifest,
    current_git_commit,
    current_git_dirty,
    source_tree_hash,
)


def run_experiment(
    tasks: list[Task],
    corpus: Corpus,
    output_dir: str | Path,
    *,
    seed: int = 42,
    top_k: int = 5,
    max_turns: int = 8,
    fault_schedule: FaultSchedule | None = None,
    failure_injector: FailureInjector | None = None,
    policy_factory: Any = None,
    run_id: str | None = None,
    rollout_count: int = 1,
    pass_k: int | None = None,
    repo_dir: str | Path | None = None,
) -> tuple[ExperimentManifest, list[Trajectory]]:
    if rollout_count < 1:
        raise ValueError("rollout_count must be positive")
    run_id = run_id or f"m0-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    repository = Path(repo_dir).resolve() if repo_dir else Path(__file__).resolve().parents[3]
    environment = LocalSearchEnvironment(
        corpus,
        top_k=top_k,
        fault_schedule=fault_schedule,
        failure_injector=failure_injector,
    )
    trajectories: list[Trajectory] = []

    async def execute() -> None:
        for task in tasks:
            for sample_index in range(rollout_count):
                policy = policy_factory(task) if policy_factory else OracleFixturePolicy()
                agent = SearchAgent(
                    policy,
                    LocalSearchTool(environment),
                    LocalVisitTool(environment),
                    max_turns=max_turns,
                    seed=seed,
                    sample_index=sample_index,
                )
                trajectories.append(await agent.run_async(task))

    asyncio.run(execute())
    fault_config: dict[str, Any] = {}
    if failure_injector:
        fault_config["probabilistic"] = {
            "seed": failure_injector.seed,
            "timeout_rate": failure_injector.timeout_rate,
            "exception_rate": failure_injector.exception_rate,
            "empty_result_rate": failure_injector.empty_result_rate,
            "malformed_result_rate": failure_injector.malformed_result_rate,
            "irrelevant_result_rate": failure_injector.irrelevant_result_rate,
        }
    if fault_schedule:
        fault_config["schedule"] = {str(key): str(value) for key, value in fault_schedule.events.items()}
    manifest = ExperimentManifest(
        run_id=run_id,
        stage="m0",
        seed=seed,
        git_commit=current_git_commit(str(repository)),
        git_dirty=current_git_dirty(str(repository)),
        source_tree_hash=source_tree_hash(str(repository)),
        dataset={
            "name": "tracesearch-m0-fixture",
            "version": "1.0",
            "split": tasks[0].split if tasks else "empty",
            "task_count": len(tasks),
            "corpus_hash": hash_corpus(corpus.documents),
        },
        environment={"backend": "local_bm25", "top_k": top_k, "fault": fault_config},
        agent={"policy": "OracleFixturePolicy" if policy_factory is None else getattr(policy_factory, "__name__", "custom"), "max_turns": max_turns, "tool_budget": None, "rollout_count": rollout_count},
        model=None,
        training=None,
        budget=None,
        cost={"currency": None, "total": 0},
    )
    write_run_artifacts(
        output_dir,
        manifest,
        tasks,
        trajectories,
        top_k=top_k,
        expected_rollouts=rollout_count,
        pass_k=pass_k,
    )
    return manifest, trajectories


def run_from_files(
    tasks_path: str | Path,
    corpus_path: str | Path,
    output_dir: str | Path,
    **kwargs: Any,
) -> tuple[ExperimentManifest, list[Trajectory]]:
    return run_experiment(load_tasks(tasks_path), Corpus(load_corpus(corpus_path)), output_dir, **kwargs)
