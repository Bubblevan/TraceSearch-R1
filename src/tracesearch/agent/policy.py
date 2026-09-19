"""Model-vendor-neutral policy protocols and deterministic fixture policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from tracesearch.data.schema import Action, ActionKind, Task, Trajectory


@dataclass(frozen=True)
class PolicyOutput:
    reasoning: str
    action: Action

    @property
    def thought(self) -> str:
        return self.reasoning


class Policy(Protocol):
    async def act(self, task: Task, trajectory: Trajectory) -> PolicyOutput: ...


class ScriptedPolicy:
    """Replay a fixed list of actions for deterministic tests and fixtures."""

    def __init__(self, outputs: Sequence[PolicyOutput | Action | tuple[str, Action] | tuple[Action, str]]):
        self.outputs = list(outputs)

    async def act(self, task: Task, trajectory: Trajectory) -> PolicyOutput:
        index = len(trajectory.steps)
        if index >= len(self.outputs):
            raise RuntimeError("scripted policy exhausted")
        output = self.outputs[index]
        if isinstance(output, PolicyOutput):
            return output
        if isinstance(output, Action):
            return PolicyOutput(reasoning="scripted action", action=output)
        first, second = output
        if isinstance(first, str):
            return PolicyOutput(reasoning=first, action=second)
        return PolicyOutput(reasoning=second, action=first)


class OracleFixturePolicy:
    """Explicit fixture-only policy that uses gold evidence to smoke-test M0.

    This policy is intentionally named and never used by ``SearchAgent``
    automatically.  It is not a model and makes no claim about learned search.
    """

    allow_gold_evidence = True

    def __init__(self, *, search_first: bool = True):
        self.search_first = search_first

    async def act(self, task: Task, trajectory: Trajectory) -> PolicyOutput:
        if self.search_first and not trajectory.steps and task.gold_evidence_ids:
            return PolicyOutput("Fixture policy searches the task question.", Action(ActionKind.SEARCH, task.question))
        visited = {
            step.action.value
            for step in trajectory.steps
            if step.action.kind is ActionKind.VISIT and not step.failed
        }
        for doc_id in task.gold_evidence_ids:
            if doc_id not in visited:
                return PolicyOutput("Fixture policy visits aligned evidence.", Action(ActionKind.VISIT, doc_id))
        answer = task.answers[0] if task.answers else "No fixture answer provided."
        return PolicyOutput("Fixture policy returns the declared fixture answer.", Action(ActionKind.ANSWER, answer))
