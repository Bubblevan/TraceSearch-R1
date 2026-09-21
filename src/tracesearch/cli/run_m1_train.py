"""Run the M1-C backend entry point after validating optional dependencies."""

from __future__ import annotations

import argparse
import importlib.util


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m1/train.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--backend", choices=("rllm-verl",), default="rllm-verl")
    parser.add_argument("--phase", choices=("c0", "c1", "c2"), default="c0")
    parser.add_argument("--dataset", default="data/m1/dev.jsonl")
    parser.add_argument("--corpus", default="data/m0/corpus.jsonl")
    parser.add_argument("--output", default="runs/m1-c-backend")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--task-count", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    parser.add_argument("--cpu-offload-gb", type=float, default=4.0)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    missing = [name for name in ("rllm", "verl", "vllm") if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(
            "M1 Level C is blocked: missing optional training dependencies "
            + ", ".join(missing)
            + ". Install a compatibility-tested backend before running training."
        )
    from tracesearch.cli.run_m1c_backend import run

    checkpoint = args.checkpoint or args.model
    backend_args = argparse.Namespace(
        phase=args.phase,
        model=checkpoint,
        dataset=args.dataset,
        corpus=args.corpus,
        output=args.output,
        seed=args.seed,
        group_size=args.group_size,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        top_k=args.top_k,
        task_count=args.task_count,
        gpu_memory_utilization=args.gpu_memory_utilization,
        cpu_offload_gb=args.cpu_offload_gb,
        enforce_eager=args.enforce_eager,
    )
    return run(backend_args)


if __name__ == "__main__":
    main()
