"""Asynchronous canonical agent loop with a synchronous compatibility wrapper."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Callable
from typing import Any

from tracesearch.agent.policy import PolicyOutput
from tracesearch.agent.types import (
    Action,
    ActionKind,
    Step,
    TerminationReason,
    ToolErrorType,
    ToolResult,
    Trajectory,
)
from tracesearch.data.schema import Task
from tracesearch.environment.render import render_tool_result


Policy = Callable[[str, list[Step]], tuple[str, Action]]


def exception_to_tool_error_type(exc: BaseException) -> ToolErrorType:
    """Classify adapter exceptions without collapsing known failures to INTERNAL."""

    if isinstance(exc, TimeoutError):
        return ToolErrorType.TIMEOUT
    if isinstance(exc, (FileNotFoundError, KeyError, LookupError)):
        return ToolErrorType.NOT_FOUND
    if isinstance(exc, (ConnectionError, OSError)):
        return ToolErrorType.TRANSIENT
    if isinstance(exc, ValueError):
        return ToolErrorType.INVALID_ARGUMENT
    if isinstance(exc, TypeError):
        return ToolErrorType.MALFORMED_RESULT
    return ToolErrorType.INTERNAL


class SearchAgent:
    """Execute search/visit/answer policies against sync or async tools."""

    def __init__(
        self,
        policy: Any,
        search: Any,
        visit: Any,
        max_turns: int = 6,
        *,
        max_searches: int | None = None,
        max_visits: int | None = None,
        tool_budget: int | None = None,
        seed: int | None = None,
        rollout_id: str | None = None,
        sample_index: int = 0,
    ):
        if max_turns < 1:
            raise ValueError("max_turns must be positive")
        for name, value in (("max_searches", max_searches), ("max_visits", max_visits), ("tool_budget", tool_budget)):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        self.policy = policy
        self.search = search
        self.visit = visit
        self.max_turns = max_turns
        self.max_searches = max_searches
        self.max_visits = max_visits
        self.tool_budget = tool_budget
        self.seed = seed
        self.rollout_id = rollout_id
        self.sample_index = sample_index

    async def run_async(
        self,
        task: Task,
        *,
        rollout_id: str | None = None,
        sample_index: int | None = None,
    ) -> Trajectory:
        """Run one task and return the canonical structured trajectory."""

        if not isinstance(task, Task):
            raise TypeError("run_async expects a Task")
        resolved_sample_index = self.sample_index if sample_index is None else sample_index
        if resolved_sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        resolved_rollout_id = rollout_id or self.rollout_id or self._rollout_id(task, resolved_sample_index)
        trajectory = Trajectory(
            question=task.question,
            task_id=task.task_id,
            seed=self.seed,
            rollout_id=resolved_rollout_id,
            sample_index=resolved_sample_index,
        )
        for step_index in range(self.max_turns):
            if self._budget_exhausted(trajectory):
                trajectory.termination_reason = TerminationReason.TOOL_BUDGET_EXHAUSTED
                trajectory.termination = trajectory.termination_reason.value
                return trajectory
            try:
                output = await self._ask_policy(task, trajectory)
            except Exception as exc:
                trajectory.steps.append(
                    Step(
                        step_index=step_index,
                        thought="",
                        action=Action(ActionKind.ANSWER, ""),
                        error=f"{type(exc).__name__}: {exc}",
                        metadata={"failure_class": "policy_error"},
                    )
                )
                trajectory.termination_reason = TerminationReason.POLICY_ERROR
                trajectory.termination = trajectory.termination_reason.value
                return trajectory

            if output.action.kind is ActionKind.ANSWER:
                trajectory.steps.append(
                    Step(
                        step_index=step_index,
                        thought=output.reasoning,
                        action=output.action,
                        metadata=self._policy_output_metadata(output),
                    )
                )
                trajectory.answer = output.action.value
                trajectory.termination_reason = TerminationReason.ANSWER
                trajectory.termination = trajectory.termination_reason.value
                return trajectory

            try:
                result = await self._execute_result(
                    output.action,
                    step_index=step_index,
                    task_id=task.task_id,
                    rollout_id=trajectory.rollout_id,
                )
            except Exception as exc:
                result = ToolResult(
                    tool_name=output.action.kind.value,
                    ok=False,
                    request=output.action.value,
                    error_type=exception_to_tool_error_type(exc),
                    error_message=f"{type(exc).__name__}: {exc}",
                    metadata={
                        "failure_class": "environment_error",
                        "exception_class": type(exc).__name__,
                    },
                )
            trajectory.steps.append(
                Step(
                    step_index=step_index,
                    thought=output.reasoning,
                    action=output.action,
                    observation=render_tool_result(result),
                    error=result.error_message if not result.ok else None,
                    tool_result=result,
                    latency_ms=result.latency_ms,
                    metadata={"tool_name": result.tool_name, **self._policy_output_metadata(output)},
                )
            )

        trajectory.termination_reason = TerminationReason.MAX_TURNS
        trajectory.termination = trajectory.termination_reason.value
        return trajectory

    def run(self, question: str) -> Trajectory:
        """Compatibility wrapper for the original question-string API."""

        task_id = "compat-" + hashlib.sha256(question.encode("utf-8")).hexdigest()[:16]
        task = Task(task_id=task_id, question=question)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(task))
        raise RuntimeError("SearchAgent.run cannot be called from an active event loop; use run_async")

    async def _ask_policy(self, task: Task, trajectory: Trajectory) -> PolicyOutput:
        policy = self.policy
        if hasattr(policy, "act"):
            policy_task = task if getattr(policy, "allow_gold_evidence", False) else task.policy_view()
            raw = policy.act(policy_task, trajectory)
        else:
            raw = policy(task.question, trajectory.steps)
        if inspect.isawaitable(raw):
            raw = await raw
        if isinstance(raw, PolicyOutput):
            return raw
        if isinstance(raw, Action):
            return PolicyOutput("", raw)
        if isinstance(raw, tuple) and len(raw) == 2:
            thought, action = raw
            if isinstance(thought, Action):
                thought, action = action, thought
            return PolicyOutput(str(thought), action)
        raise TypeError("policy must return PolicyOutput, Action, or (reasoning, Action)")

    async def _execute_result(
        self,
        action: Action,
        *,
        step_index: int,
        task_id: str,
        rollout_id: str,
    ) -> ToolResult:
        if action.kind is ActionKind.SEARCH:
            raw = await self._call_tool(self.search, "search", action.value, step_index, task_id, rollout_id)
        elif action.kind is ActionKind.VISIT:
            raw = await self._call_tool(self.visit, "visit", action.value, step_index, task_id, rollout_id)
        else:
            raise ValueError(f"unsupported tool action: {action.kind}")
        if isinstance(raw, ToolResult):
            return raw
        if isinstance(raw, str):
            return ToolResult(tool_name=action.kind.value, ok=True, request=action.value, text=raw)
        if isinstance(raw, dict):
            return ToolResult.from_dict(raw)
        raise TypeError(f"tool returned unsupported result type: {type(raw).__name__}")

    async def _call_tool(
        self,
        tool: Any,
        method_name: str,
        value: str,
        step_index: int,
        task_id: str,
        rollout_id: str,
    ) -> Any:
        method = getattr(tool, method_name)
        kwargs: dict[str, object] = {}
        try:
            parameters = inspect.signature(method).parameters
            accepts_context = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
            if "step_index" in parameters or accepts_context:
                kwargs["step_index"] = step_index
            if "task_id" in parameters or accepts_context:
                kwargs["task_id"] = task_id
            if "rollout_id" in parameters or accepts_context:
                kwargs["rollout_id"] = rollout_id
        except (TypeError, ValueError):
            parameters = {}
        raw = method(value, **kwargs)
        return await raw if inspect.isawaitable(raw) else raw

    def _budget_exhausted(self, trajectory: Trajectory) -> bool:
        if self.tool_budget is not None and trajectory.tool_turns >= self.tool_budget:
            return True
        if self.max_searches is not None and trajectory.search_calls >= self.max_searches:
            return True
        if self.max_visits is not None and trajectory.visit_calls >= self.max_visits:
            return True
        return False

    @staticmethod
    def _policy_output_metadata(output: PolicyOutput) -> dict[str, Any]:
        metadata = dict(output.metadata)
        if output.raw_text is not None:
            metadata["policy_raw_output"] = output.raw_text
        if output.response_token_ids is not None:
            metadata["response_token_ids"] = list(output.response_token_ids)
        if output.response_logprobs is not None:
            metadata["response_logprobs"] = list(output.response_logprobs)
        return metadata

    def _rollout_id(self, task: Task, sample_index: int) -> str:
        material = f"{task.task_id}|{self.seed if self.seed is not None else 0}|{sample_index}".encode("utf-8")
        return "rollout-" + hashlib.sha256(material).hexdigest()[:24]

    def _trajectory_id(self, task: Task, sample_index: int | None = None) -> str:
        index = self.sample_index if sample_index is None else sample_index
        rollout_id = self.rollout_id or self._rollout_id(task, index)
        material = f"{rollout_id}|{task.question}".encode("utf-8")
        return "traj-" + hashlib.sha256(material).hexdigest()[:24]

    def _execute(self, action: Action) -> str:
        """Legacy helper retained for callers of the original scaffold."""

        async def execute() -> ToolResult:
            return await self._execute_result(
                action,
                step_index=0,
                task_id="compat",
                rollout_id="compat",
            )

        result = asyncio.run(execute())
        return render_tool_result(result)
