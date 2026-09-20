"""Prepare a deterministic, bounded NQ-style dataset split without bundling data."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from tracesearch.data.adapters import NQTaskAdapter
from tracesearch.data.io import write_json, write_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="local NQ-style JSONL source")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--upstream-dataset", default="natural_questions")
    parser.add_argument("--upstream-revision", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=0)
    parser.add_argument("--dev-size", type=int, default=0)
    parser.add_argument("--test-size", type=int, default=0)
    return parser


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"dataset row at {path}:{line_number} must be an object")
            rows.append(value)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sizes = {"train": args.train_size, "dev": args.dev_size, "test": args.test_size}
    if any(size < 0 for size in sizes.values()) or not any(sizes.values()):
        raise SystemExit("at least one non-negative split size must be positive")
    rows = _read_rows(args.input)
    adapter = NQTaskAdapter(
        upstream_dataset=args.upstream_dataset,
        upstream_revision=args.upstream_revision,
    )
    # Validate IDs and sort before sampling so source file order cannot alter a split.
    validated = sorted(rows, key=adapter._source_id)
    source_ids = [adapter._source_id(row) for row in validated]
    if len(source_ids) != len(set(source_ids)):
        raise SystemExit("input dataset contains duplicate stable example IDs")
    rng = random.Random(args.seed)
    rng.shuffle(validated)
    requested = sum(sizes.values())
    if requested > len(validated):
        raise SystemExit(f"requested {requested} rows but input has only {len(validated)}")
    selected = validated[:requested]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    manifest: dict[str, Any] = {
        "source": str(Path(args.input)),
        "upstream_dataset": args.upstream_dataset,
        "upstream_revision": args.upstream_revision,
        "selection_seed": args.seed,
        "preprocessing_version": adapter.preprocessing_version,
        "splits": {},
    }
    for split, size in sizes.items():
        split_rows = selected[cursor : cursor + size]
        cursor += size
        if not split_rows:
            continue
        dataset = adapter.adapt_rows(split_rows, split=split)
        path = output_dir / f"{split}.jsonl"
        write_tasks(path, dataset.tasks)
        manifest["splits"][split] = dataset.manifest_fields()
    write_json(output_dir / "manifest.json", manifest)
    print(f"prepared: {output_dir}")
    print(f"rows: {requested}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
