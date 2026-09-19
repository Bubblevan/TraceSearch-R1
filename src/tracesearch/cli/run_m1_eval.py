"""Run a real-model M1 baseline evaluation against external services."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from tracesearch.agent import (
    DirectAnswerPolicy,
    LLMGenerationConfig,
    LLMPolicy,
    OpenAICompatibleClient,
    RetrieveOncePolicy,
)
from tracesearch.data.adapters import NQTaskAdapter
from tracesearch.environment import HTTPRetriever
from tracesearch.evaluation.baselines import run_baseline_rollouts
from tracesearch.experiment.m1_artifacts import write_m1_artifacts
from tracesearch.experiment.manifest import ExperimentManifest, current_git_commit, current_git_dirty, source_tree_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/m1/dev.jsonl")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--retriever-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible model gateway base URL")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--mode", choices=("direct", "retrieve-once", "agent"), default="agent")
    parser.add_argument("--output", default="runs/m1-eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = NQTaskAdapter().load_jsonl(args.dataset, split=args.split)
    client = OpenAICompatibleClient(args.base_url, api_key=args.api_key or os.environ.get("OPENAI_API_KEY"))
    generation = LLMGenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    if args.mode == "direct":
        policy = DirectAnswerPolicy(client, model=args.model, generation=generation)
    elif args.mode == "retrieve-once":
        policy = RetrieveOncePolicy(client, model=args.model, generation=generation)
    else:
        policy = LLMPolicy(client, model=args.model, generation=generation, require_token_ids=True)
    retriever = HTTPRetriever(args.retriever_url, top_k=args.top_k)
    trajectories = asyncio.run(
        run_baseline_rollouts(
            dataset.tasks,
            policy,
            retriever,
            retriever,
            seed=args.seed,
            max_turns=args.max_turns,
        )
    )
    manifest = ExperimentManifest(
        run_id=Path(args.output).name,
        stage="m1-eval",
        seed=args.seed,
        git_commit=current_git_commit(),
        git_dirty=current_git_dirty(),
        source_tree_hash=source_tree_hash(),
        dataset=dataset.manifest_fields(),
        environment={
            "backend": "http_retriever",
            "retriever_url": args.retriever_url,
            "top_k": args.top_k,
        },
        agent={"baseline": args.mode, "max_turns": args.max_turns},
        model={
            "model": args.model,
            "base_url": args.base_url,
            "prompt_template_version": "tracesearch-m1-v1",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_generation_tokens": args.max_tokens,
        },
    )
    output = write_m1_artifacts(args.output, manifest, dataset.tasks, trajectories, top_k=args.top_k)
    print(f"artifacts: {output}")
    print(f"trajectories: {len(trajectories)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
