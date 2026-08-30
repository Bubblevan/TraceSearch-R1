from __future__ import annotations

from collections.abc import Callable

from tracesearch.agent.types import Action, ActionKind, Step, Trajectory
from tracesearch.environment.tools import SearchTool, VisitTool

Policy = Callable[[str, list[Step]], tuple[str, Action]]


class SearchAgent:
    """Minimal synchronous control loop; training backends can reuse its trajectory schema."""

    def __init__(self, policy: Policy, search: SearchTool, visit: VisitTool, max_turns: int = 6):
        self.policy = policy
        self.search = search
        self.visit = visit
        self.max_turns = max_turns

    def run(self, question: str) -> Trajectory:
        trajectory = Trajectory(question=question)
        for _ in range(self.max_turns):
            thought, action = self.policy(question, trajectory.steps)
            if action.kind is ActionKind.ANSWER:
                trajectory.steps.append(Step(thought=thought, action=action))
                trajectory.answer = action.value
                trajectory.termination = "answer"
                return trajectory
            try:
                observation = self._execute(action)
                trajectory.steps.append(Step(thought=thought, action=action, observation=observation))
            except Exception as exc:  # adapters surface a failed environment step
                trajectory.steps.append(Step(thought=thought, action=action, error=str(exc)))
        return trajectory

    def _execute(self, action: Action) -> str:
        if action.kind is ActionKind.SEARCH:
            return self.search.search(action.value)
        if action.kind is ActionKind.VISIT:
            return self.visit.visit(action.value)
        raise ValueError(f"Unsupported action: {action.kind}")
