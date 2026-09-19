"""Serializable experiment manifest for every reproducible run."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tracesearch.data.schema import SCHEMA_VERSION


class Section(dict[str, Any]):
    """A JSON object that is convenient to use as either mapping or attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass
class ExperimentManifest:
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    stage: str = "m0"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    git_dirty: bool | None = None
    source_tree_hash: str | None = None
    seed: int = 0
    dataset: Section | dict[str, Any] = field(default_factory=Section)
    environment: Section | dict[str, Any] = field(default_factory=Section)
    agent: Section | dict[str, Any] = field(default_factory=Section)
    model: Section | dict[str, Any] | None = None
    training: Section | dict[str, Any] | None = None
    budget: Section | dict[str, Any] | None = None
    cost: Section | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id is required")
        self.dataset = Section(self.dataset or {})
        self.environment = Section(self.environment or {})
        self.agent = Section(self.agent or {})
        self.model = Section(self.model or {}) if self.model is not None else None
        self.training = Section(self.training or {}) if self.training is not None else None
        self.budget = Section(self.budget or {}) if self.budget is not None else None
        self.cost = Section(self.cost or {}) if self.cost is not None else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "stage": self.stage,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "source_tree_hash": self.source_tree_hash,
            "seed": self.seed,
            "dataset": dict(self.dataset),
            "environment": dict(self.environment),
            "agent": dict(self.agent),
            "model": dict(self.model) if self.model is not None else None,
            "training": dict(self.training) if self.training is not None else None,
            "budget": dict(self.budget) if self.budget is not None else None,
            "cost": dict(self.cost) if self.cost is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentManifest":
        return cls(**{key: data[key] for key in data if key in cls.__dataclass_fields__})


def current_git_commit(repo_dir: str | None = None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def current_git_dirty(repo_dir: str | None = None) -> bool | None:
    """Return whether tracked or non-ignored untracked files are present."""

    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(completed.stdout.strip())


def source_tree_hash(repo_dir: str | None = None) -> str | None:
    """Hash tracked plus non-ignored untracked files with their relative paths."""

    root = Path(repo_dir or Path.cwd()).resolve()
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    digest = hashlib.sha256()
    paths = sorted(path for path in completed.stdout.splitlines() if path)
    for relative in paths:
        path = root / relative
        if not path.is_file():
            continue
        digest.update(relative.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()
