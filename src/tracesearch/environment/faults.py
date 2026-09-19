"""Deterministic scheduled and seeded probabilistic fault injection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from tracesearch.data.schema import ToolErrorType


class FaultType(StrEnum):
    TIMEOUT = "timeout"
    TRANSIENT = "transient"
    EMPTY_RESULT = "empty_result"
    MALFORMED_RESULT = "malformed_result"
    EXCEPTION = "exception"
    IRRELEVANT_RESULT = "irrelevant_result"
    INJECTED_FAILURE = "injected_failure"


FaultKind = FaultType


def _fault_type(value: FaultType | ToolErrorType | str) -> FaultType:
    raw = value.value if isinstance(value, (FaultType, ToolErrorType)) else value
    if raw == ToolErrorType.INVALID_ARGUMENT.value:
        return FaultType.INJECTED_FAILURE
    if raw == ToolErrorType.INTERNAL.value:
        return FaultType.EXCEPTION
    try:
        return FaultType(raw)
    except ValueError as exc:
        raise ValueError(f"unsupported fault type: {value!r}") from exc


@dataclass
class FaultSchedule:
    """Map ``(step_index, tool_name)`` or ``(task_id, step, tool)`` to a fault."""

    events: Mapping[Any, FaultType | ToolErrorType | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.events = dict(self.events)

    def add(
        self,
        step_index: int,
        tool_name: str,
        fault: FaultType | ToolErrorType | str,
        *,
        task_id: str | None = None,
    ) -> None:
        key = (task_id, step_index, tool_name) if task_id is not None else (step_index, tool_name)
        self.events[key] = fault

    def get_fault(self, *, step_index: int, tool_name: str, task_id: str | None = None) -> FaultType | None:
        candidates = (
            (task_id, step_index, tool_name),
            (step_index, tool_name),
            (step_index, tool_name.casefold()),
            step_index,
            f"{step_index}:{tool_name}",
        )
        for key in candidates:
            if key in self.events:
                return _fault_type(self.events[key])
        return None

    def fault_for(self, task_id: str | None, step_index: int, tool_name: str) -> FaultType | None:
        return self.get_fault(task_id=task_id, step_index=step_index, tool_name=tool_name)


@dataclass
class FailureInjector:
    """Seeded independent fault decisions for a fixed action sequence."""

    seed: int = 0
    timeout_rate: float = 0.0
    exception_rate: float = 0.0
    empty_result_rate: float = 0.0
    malformed_result_rate: float = 0.0
    irrelevant_result_rate: float = 0.0
    def __post_init__(self) -> None:
        rates = (
            self.timeout_rate,
            self.exception_rate,
            self.empty_result_rate,
            self.malformed_result_rate,
            self.irrelevant_result_rate,
        )
        if any(rate < 0 or rate > 1 for rate in rates):
            raise ValueError("fault rates must be between 0 and 1")

    def next_fault(
        self,
        tool_name: str = "",
        step_index: int | None = None,
        *,
        task_id: str = "",
        rollout_id: str = "",
    ) -> FaultType | None:
        """Return a keyed decision with no mutable RNG state.

        The key includes every part of the failure surface that must remain
        stable when rollouts are reordered or executed concurrently.
        """

        step = 0 if step_index is None else step_index
        rates = (
            (self.timeout_rate, FaultType.TIMEOUT),
            (self.exception_rate, FaultType.EXCEPTION),
            (self.empty_result_rate, FaultType.EMPTY_RESULT),
            (self.malformed_result_rate, FaultType.MALFORMED_RESULT),
            (self.irrelevant_result_rate, FaultType.IRRELEVANT_RESULT),
        )
        for draw_index, (rate, fault) in enumerate(rates):
            value = self._uniform(task_id, rollout_id, step, tool_name, draw_index)
            if value < rate:
                return fault
        return None

    def decide(
        self,
        tool_name: str = "",
        step_index: int | None = None,
        *,
        task_id: str = "",
        rollout_id: str = "",
    ) -> FaultType | None:
        return self.next_fault(
            tool_name,
            step_index,
            task_id=task_id,
            rollout_id=rollout_id,
        )

    def _uniform(self, task_id: str, rollout_id: str, step_index: int, tool_name: str, draw_index: int) -> float:
        material = "|".join(
            (str(self.seed), task_id, rollout_id, str(step_index), tool_name, str(draw_index))
        ).encode("utf-8")
        digest = hashlib.blake2b(material, digest_size=8).digest()
        return int.from_bytes(digest, "big") / 2**64
