"""Typed, JSON-friendly schemas used by the M0 research substrate.

The classes in this module intentionally use only standard-library types.  They
are the stable boundary between policies, environments, evaluators, and future
training integrations.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar


SCHEMA_VERSION = "m0.v1"


class ActionKind(StrEnum):
    SEARCH = "search"
    VISIT = "visit"
    ANSWER = "answer"


class ToolErrorType(StrEnum):
    TIMEOUT = "timeout"
    TRANSIENT = "transient"
    INVALID_ARGUMENT = "invalid_argument"
    NOT_FOUND = "not_found"
    EMPTY_RESULT = "empty_result"
    MALFORMED_RESULT = "malformed_result"
    INJECTED_FAILURE = "injected_failure"
    INTERNAL = "internal"


class TerminationReason(StrEnum):
    ANSWER = "answer"
    MAX_TURNS = "max_turns"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
    FATAL_TOOL_FAILURE = "fatal_tool_failure"
    POLICY_ERROR = "policy_error"
    ENVIRONMENT_ERROR = "environment_error"


def _enum_value(value: Any) -> Any:
    return value.value if isinstance(value, StrEnum) else value


def _jsonable(value: Any) -> Any:
    """Convert nested schema values without leaking Python enum syntax."""

    if isinstance(value, StrEnum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _require_text(name: str, value: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    answers: list[str] = field(default_factory=list)
    split: str = "unspecified"
    gold_evidence_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("task_id", self.task_id)
        _require_text("question", self.question)
        _require_text("split", self.split)
        if not all(isinstance(answer, str) and answer.strip() for answer in self.answers):
            raise ValueError("answers must contain non-empty strings")
        if not all(isinstance(doc_id, str) and doc_id.strip() for doc_id in self.gold_evidence_ids):
            raise ValueError("gold_evidence_ids must contain non-empty strings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "question": self.question,
            "answers": list(self.answers),
            "split": self.split,
            "gold_evidence_ids": list(self.gold_evidence_ids),
            "metadata": _jsonable(self.metadata),
        }

    def policy_view(self) -> "Task":
        """Return the task surface exposed to ordinary policies."""

        return Task(
            task_id=self.task_id,
            question=self.question,
            answers=[],
            split=self.split,
            gold_evidence_ids=[],
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Task":
        return cls(
            task_id=str(data["task_id"]),
            question=str(data["question"]),
            answers=list(data.get("answers", [])),
            split=str(data.get("split", "unspecified")),
            gold_evidence_ids=list(data.get("gold_evidence_ids", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("doc_id", self.doc_id)
        _require_text("title", self.title)
        _require_text("text", self.text)
        if self.url is not None and not isinstance(self.url, str):
            raise ValueError("url must be a string or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(
            doc_id=str(data["doc_id"]),
            title=str(data["title"]),
            text=str(data["text"]),
            url=data.get("url"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class Evidence:
    doc_id: str
    title: str
    snippet: str
    score: float
    rank: int
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("doc_id", self.doc_id)
        _require_text("title", self.title)
        if not isinstance(self.snippet, str):
            raise ValueError("snippet must be a string")
        if self.rank < 1:
            raise ValueError("rank must be positive")
        if self.url is not None and not isinstance(self.url, str):
            raise ValueError("url must be a string or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "snippet": self.snippet,
            "score": float(self.score),
            "rank": int(self.rank),
            "url": self.url,
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        return cls(
            doc_id=str(data["doc_id"]),
            title=str(data["title"]),
            snippet=str(data.get("snippet", "")),
            score=float(data.get("score", 0.0)),
            rank=int(data.get("rank", 1)),
            url=data.get("url"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    value: str

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "kind", ActionKind(self.kind))
        except ValueError as exc:
            raise ValueError(f"unsupported action kind: {self.kind!r}") from exc
        if not isinstance(self.value, str):
            raise ValueError("action value must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Action":
        return cls(kind=ActionKind(data["kind"]), value=str(data.get("value", "")))


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    ok: bool
    request: Any
    text: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    error_type: ToolErrorType | None = None
    error_message: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("tool_name", self.tool_name)
        if self.ok and self.error_type is not None:
            raise ValueError("successful ToolResult cannot have error_type")
        if not self.ok and self.error_type is None:
            raise ValueError("failed ToolResult must have error_type")
        if self.error_type is not None:
            try:
                object.__setattr__(self, "error_type", ToolErrorType(self.error_type))
            except ValueError as exc:
                raise ValueError(f"unsupported tool error type: {self.error_type!r}") from exc
        if self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        if not all(isinstance(item, Evidence) for item in self.evidence):
            raise ValueError("evidence must contain Evidence objects")

    @property
    def failed(self) -> bool:
        return not self.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "ok": self.ok,
            "request": _jsonable(self.request),
            "text": self.text,
            "evidence": [item.to_dict() for item in self.evidence],
            "error_type": _enum_value(self.error_type),
            "error_message": self.error_message,
            "latency_ms": float(self.latency_ms),
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolResult":
        error_type = data.get("error_type")
        return cls(
            tool_name=str(data["tool_name"]),
            ok=bool(data["ok"]),
            request=data.get("request"),
            text=str(data.get("text", "")),
            evidence=[Evidence.from_dict(item) for item in data.get("evidence", [])],
            error_type=ToolErrorType(error_type) if error_type else None,
            error_message=data.get("error_message"),
            latency_ms=float(data.get("latency_ms", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Step:
    """One policy decision and its structured environment response.

    The first five fields retain the original M0 scaffold's positional API.
    New callers should prefer keyword arguments and ``tool_result``.
    """

    thought: str
    action: Action
    observation: str = ""
    error: str | None = None
    contribution: float | None = None
    step_index: int = 0
    tool_result: ToolResult | None = None
    timestamp: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action, Action):
            self.action = Action.from_dict(self.action) if isinstance(self.action, dict) else Action(*self.action)
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if self.tool_result is not None and not isinstance(self.tool_result, ToolResult):
            self.tool_result = ToolResult.from_dict(self.tool_result)
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")

    @property
    def failed(self) -> bool:
        return bool(self.error) or (self.tool_result is not None and self.tool_result.failed)

    @property
    def is_tool_step(self) -> bool:
        return self.action.kind is not ActionKind.ANSWER

    @property
    def reasoning(self) -> str:
        return self.thought

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "thought": self.thought,
            "reasoning": self.thought,
            "action": self.action.to_dict(),
            "observation": self.observation,
            "error": self.error,
            "tool_result": self.tool_result.to_dict() if self.tool_result else None,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "contribution": self.contribution,
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Step":
        return cls(
            thought=str(data.get("thought", data.get("reasoning", ""))),
            action=Action.from_dict(data["action"]),
            observation=str(data.get("observation", "")),
            error=data.get("error"),
            contribution=data.get("contribution"),
            step_index=int(data.get("step_index", 0)),
            tool_result=ToolResult.from_dict(data["tool_result"]) if data.get("tool_result") else None,
            timestamp=data.get("timestamp"),
            latency_ms=data.get("latency_ms"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Trajectory:
    question: str
    steps: list[Step] = field(default_factory=list)
    answer: str | None = None
    termination: str = TerminationReason.MAX_TURNS.value
    schema_version: str = SCHEMA_VERSION
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str | None = None
    termination_reason: TerminationReason | None = None
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("question", self.question)
        if self.termination_reason is None:
            self.termination_reason = TerminationReason(self.termination)
        else:
            self.termination_reason = TerminationReason(self.termination_reason)
        self.termination = self.termination_reason.value
        if not all(isinstance(step, Step) for step in self.steps):
            self.steps = [Step.from_dict(step) if isinstance(step, dict) else step for step in self.steps]

    @property
    def tool_turns(self) -> int:
        return sum(step.is_tool_step for step in self.steps)

    @property
    def search_calls(self) -> int:
        return sum(step.action.kind is ActionKind.SEARCH for step in self.steps)

    @property
    def visit_calls(self) -> int:
        return sum(step.action.kind is ActionKind.VISIT for step in self.steps)

    @property
    def failed_tool_calls(self) -> int:
        return sum(step.is_tool_step and step.failed for step in self.steps)

    @property
    def successful_tool_calls(self) -> int:
        return sum(step.is_tool_step and not step.failed for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "question": self.question,
            "steps": [step.to_dict() for step in self.steps],
            "answer": self.answer,
            "termination_reason": self.termination_reason.value,
            "termination": self.termination,
            "seed": self.seed,
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Trajectory":
        reason = data.get("termination_reason", data.get("termination", TerminationReason.MAX_TURNS.value))
        return cls(
            question=str(data["question"]),
            steps=[Step.from_dict(item) for item in data.get("steps", [])],
            answer=data.get("answer"),
            termination=str(reason),
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            trajectory_id=str(data.get("trajectory_id", uuid.uuid4())),
            task_id=data.get("task_id"),
            termination_reason=TerminationReason(reason),
            seed=data.get("seed"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
