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
    TransformersModelClient,
)
from tracesearch.data.adapters import NQTaskAdapter
from tracesearch.data.io import load_corpus
from tracesearch.environment import (
    HTTPRetriever,
    Corpus,
    LocalSearchEnvironment,
    LocalSearchTool,
    LocalVisitTool,
)
from tracesearch.evaluation.baselines import run_baseline_rollouts
from tracesearch.experiment.m1_artifacts import write_m1_artifacts
from tracesearch.experiment.manifest import ExperimentManifest, current_git_commit, current_git_dirty, source_tree_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/m1/dev.jsonl")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--retriever-url", default=None)
    parser.add_argument("--corpus", default=None, help="Explicit local BM25 corpus for a Level-B smoke")
    parser.add_argument("--model", default=None, help="Model name for an OpenAI-compatible endpoint")
    parser.add_argument("--model-path", default=None, help="Local Transformers model directory")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible model gateway base URL")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--mode", choices=("direct", "retrieve-once", "agent"), default="agent")
    parser.add_argument("--output", default="runs/m1-eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gen-top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.model_path) == bool(args.base_url):
        raise SystemExit("choose exactly one model source: --model-path or --base-url")
    if args.base_url and not args.model:
        raise SystemExit("--model is required with --base-url")
    if not args.retriever_url and not args.corpus:
        raise SystemExit("choose an explicit retriever source: --retriever-url or --corpus")
    dataset = NQTaskAdapter().load_jsonl(args.dataset, split=args.split)
    if args.model_path:
        client = TransformersModelClient(args.model_path)
        model_name = Path(args.model_path).name
    else:
        client = OpenAICompatibleClient(args.base_url, api_key=args.api_key or os.environ.get("OPENAI_API_KEY"))
        model_name = args.model
    generation = LLMGenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.gen_top_k,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        enable_thinking=args.enable_thinking,
        max_tokens=args.max_tokens,
    )
    if args.mode == "direct":
        policy = DirectAnswerPolicy(client, model=model_name, generation=generation)
    elif args.mode == "retrieve-once":
        policy = RetrieveOncePolicy(client, model=model_name, generation=generation)
    else:
        policy = LLMPolicy(client, model=model_name, generation=generation, require_token_ids=True)
    if args.corpus:
        environment = LocalSearchEnvironment(Corpus(load_corpus(args.corpus)), top_k=args.top_k)
        search = LocalSearchTool(environment)
        visit = LocalVisitTool(environment)
        retriever_backend = "local_bm25"
    else:
        retriever = HTTPRetriever(args.retriever_url, top_k=args.top_k)
        search = retriever
        visit = retriever
        retriever_backend = "http_retriever"
    trajectories = asyncio.run(
        run_baseline_rollouts(
            dataset.tasks,
            policy,
            search,
            visit,
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
            "backend": retriever_backend,
            "retriever_url": args.retriever_url,
            "corpus": args.corpus,
            "top_k": args.top_k,
        },
        agent={"baseline": args.mode, "max_turns": args.max_turns},
        model={
            "model": model_name,
            "checkpoint": args.model_path,
            "base_url": args.base_url,
            "prompt_template_version": "tracesearch-m1-v1",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.gen_top_k,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
            "enable_thinking": args.enable_thinking,
            "max_generation_tokens": args.max_tokens,
        },
    )
    output = write_m1_artifacts(args.output, manifest, dataset.tasks, trajectories, top_k=args.top_k)
    print(f"artifacts: {output}")
    print(f"trajectories: {len(trajectories)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
