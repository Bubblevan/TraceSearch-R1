"""Deterministic Natural Questions-style adapter without benchmark data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tracesearch.data.schema import Task


@dataclass(frozen=True)
class DatasetProvenance:
    upstream_dataset: str
    upstream_revision: str | None
    split: str
    preprocessing_version: str
    selected_ids: tuple[str, ...]
    data_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "upstream_dataset": self.upstream_dataset,
            "upstream_revision": self.upstream_revision,
            "split": self.split,
            "preprocessing_version": self.preprocessing_version,
            "selected_ids": list(self.selected_ids),
            "data_hash": self.data_hash,
        }


@dataclass(frozen=True)
class NQDataset:
    tasks: tuple[Task, ...]
    provenance: DatasetProvenance

    def manifest_fields(self) -> dict[str, Any]:
        return {"name": "natural_questions_style", **self.provenance.to_dict(), "task_count": len(self.tasks)}


class NQTaskAdapter:
    """Convert common NQ/FlashRAG-like JSON rows to evaluator-side Tasks.

    The adapter never stores the full benchmark. Each returned Task has an
    explicit ``policy_metadata`` surface; answer aliases, evidence IDs, and
    provenance labels remain evaluator-side fields.
    """

    def __init__(
        self,
        *,
        upstream_dataset: str = "natural_questions",
        upstream_revision: str | None = None,
        preprocessing_version: str = "tracesearch-m1-nq-v1",
    ) -> None:
        if not upstream_dataset.strip():
            raise ValueError("upstream_dataset must be non-empty")
        if not preprocessing_version.strip():
            raise ValueError("preprocessing_version must be non-empty")
        self.upstream_dataset = upstream_dataset
        self.upstream_revision = upstream_revision
        self.preprocessing_version = preprocessing_version

    def adapt_rows(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        split: str,
        selected_ids: Iterable[str] | None = None,
    ) -> NQDataset:
        if not split.strip():
            raise ValueError("split must be non-empty")
        normalized_rows = [dict(row) for row in rows]
        wanted = {str(item) for item in selected_ids} if selected_ids is not None else None
        selected_rows: list[dict[str, Any]] = []
        for row in normalized_rows:
            source_id = self._source_id(row)
            if wanted is None or source_id in wanted:
                selected_rows.append(row)
        if wanted is not None:
            available = {self._source_id(row) for row in selected_rows}
            missing = sorted(wanted - available)
            if missing:
                raise ValueError(f"selected_ids not found in dataset: {missing}")
        selected_rows.sort(key=lambda row: self._source_id(row))
        tasks = tuple(self._task_from_row(row, split=split) for row in selected_rows)
        selected = tuple(task.task_id for task in tasks)
        payload = [self._canonical_row(row) for row in selected_rows]
        data_hash = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return NQDataset(
            tasks=tasks,
            provenance=DatasetProvenance(
                upstream_dataset=self.upstream_dataset,
                upstream_revision=self.upstream_revision,
                split=split,
                preprocessing_version=self.preprocessing_version,
                selected_ids=selected,
                data_hash=data_hash,
            ),
        )

    def load_jsonl(
        self,
        path: str | Path,
        *,
        split: str,
        selected_ids: Iterable[str] | None = None,
    ) -> NQDataset:
        rows: list[dict[str, Any]] = []
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid dataset JSON at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"dataset row at {path}:{line_number} must be an object")
                rows.append(row)
        return self.adapt_rows(rows, split=split, selected_ids=selected_ids)

    def _task_from_row(self, row: dict[str, Any], *, split: str) -> Task:
        task_id = self._source_id(row)
        question = self._question(row)
        answers = self._answers(row)
        evidence_ids = self._evidence_ids(row)
        provenance = {
            "upstream_dataset": self.upstream_dataset,
            "upstream_revision": self.upstream_revision,
            "preprocessing_version": self.preprocessing_version,
            "source_id": task_id,
        }
        return Task(
            task_id=task_id,
            question=question,
            answers=answers,
            split=split,
            gold_evidence_ids=evidence_ids,
            metadata={"evaluation_provenance": provenance},
            policy_metadata={"dataset": self.upstream_dataset, "split": split},
        )

    @staticmethod
    def _source_id(row: dict[str, Any]) -> str:
        for key in ("task_id", "example_id", "id", "question_id"):
            if row.get(key) is not None and str(row[key]).strip():
                return str(row[key])
        raise ValueError("NQ row must contain task_id, example_id, id, or question_id")

    @staticmethod
    def _question(row: dict[str, Any]) -> str:
        question = row.get("question")
        if isinstance(question, dict):
            question = question.get("text") or question.get("question_text")
        question = question or row.get("question_text")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("NQ row must contain a non-empty question")
        return question.strip()

    @staticmethod
    def _answers(row: dict[str, Any]) -> list[str]:
        raw: list[Any] = []
        if isinstance(row.get("answers"), list):
            raw.extend(row["answers"])
        elif row.get("answer") is not None:
            raw.append(row["answer"])
        elif isinstance(row.get("short_answers"), list):
            raw.extend(row["short_answers"])
        annotations = row.get("annotations")
        if isinstance(annotations, list):
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                short = annotation.get("short_answers") or annotation.get("answers") or []
                raw.extend(short if isinstance(short, list) else [short])
                if annotation.get("yes_no_answer"):
                    raw.append(annotation["yes_no_answer"])
        answers: list[str] = []
        for answer in raw:
            if isinstance(answer, dict):
                answer = answer.get("text") or answer.get("answer")
            if isinstance(answer, str) and answer.strip() and answer.strip() not in answers:
                answers.append(answer.strip())
        if not answers:
            raise ValueError("NQ row must contain at least one answer alias")
        return answers

    @staticmethod
    def _evidence_ids(row: dict[str, Any]) -> list[str]:
        raw = row.get("gold_evidence_ids") or row.get("document_ids") or row.get("evidence_ids") or []
        if not isinstance(raw, list):
            raw = [raw]
        return [str(item) for item in raw if str(item).strip()]

    @staticmethod
    def _canonical_row(row: dict[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
