from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ActionKind(StrEnum):
    SEARCH = "search"
    VISIT = "visit"
    ANSWER = "answer"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    value: str


@dataclass
class Step:
    thought: str
    action: Action
    observation: str = ""
    error: str | None = None
    contribution: float | None = None

    @property
    def failed(self) -> bool:
        return self.error is not None


@dataclass
class Trajectory:
    question: str
    steps: list[Step] = field(default_factory=list)
    answer: str | None = None
    termination: str = "max_turns"

    @property
    def tool_turns(self) -> int:
        return sum(step.action.kind is not ActionKind.ANSWER for step in self.steps)
