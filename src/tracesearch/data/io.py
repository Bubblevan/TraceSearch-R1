"""JSON and JSONL persistence helpers for canonical data objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, TypeVar

from tracesearch.data.schema import Document, Task, Trajectory


T = TypeVar("T")


def write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, values: Iterable[Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            payload = value.to_dict() if hasattr(value, "to_dict") else value
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return destination


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_tasks(path: str | Path, tasks: Iterable[Task]) -> Path:
    return write_jsonl(path, tasks)


def load_tasks(path: str | Path) -> list[Task]:
    return [Task.from_dict(row) for row in read_jsonl(path)]


def write_corpus(path: str | Path, documents: Iterable[Document]) -> Path:
    return write_jsonl(path, documents)


def load_corpus(path: str | Path) -> list[Document]:
    return [Document.from_dict(row) for row in read_jsonl(path)]


def write_trajectories(path: str | Path, trajectories: Iterable[Trajectory]) -> Path:
    return write_jsonl(path, trajectories)


def load_trajectories(path: str | Path) -> list[Trajectory]:
    return [Trajectory.from_dict(row) for row in read_jsonl(path)]
