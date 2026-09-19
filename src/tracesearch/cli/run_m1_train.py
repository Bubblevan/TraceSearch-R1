"""Validate M1 training prerequisites without claiming a training run."""

from __future__ import annotations

import argparse
import importlib.util


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/m1/train.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", choices=("rllm-verl",), default="rllm-verl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    missing = [name for name in ("rllm", "verl") if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(
            "M1 Level C is blocked: missing optional training dependencies "
            + ", ".join(missing)
            + ". Install a compatibility-tested backend before running training."
        )
    raise SystemExit(
        "The external rLLM/verl integration is not enabled in this checkout; "
        "run the adapter smoke tests first and record the selected backend versions."
    )


if __name__ == "__main__":
    main()
