"""Model-vendor-neutral policy protocols and deterministic fixture policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from tracesearch.agent.llm import LLMGenerationConfig, ModelClient
from tracesearch.agent.parser import parse_policy_output
from tracesearch.data.schema import Action, ActionKind, Task, Trajectory


@dataclass(frozen=True)
class PolicyOutput:
    reasoning: str
    action: Action
    raw_text: str | None = None
    response_token_ids: tuple[int, ...] | None = None
    response_logprobs: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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


class LLMPolicy:
    """Async learned policy backed by an OpenAI-compatible model client.

    ``SearchAgent`` supplies a policy-visible task surface. Model-generated
    token IDs are carried through ``PolicyOutput`` unchanged; this class never
    decodes and re-tokenizes a response for training.
    """

    allow_gold_evidence = False

    def __init__(
        self,
        client: ModelClient,
        *,
        model: str,
        generation: LLMGenerationConfig | None = None,
        prompt_template_version: str = "tracesearch-m1-v1",
        require_token_ids: bool = False,
        mode: str = "agent",
    ) -> None:
        if not model.strip():
            raise ValueError("model must be non-empty")
        self.client = client
        self.model = model
        self.generation = generation or LLMGenerationConfig()
        self.prompt_template_version = prompt_template_version
        self.require_token_ids = require_token_ids
        self.mode = mode

    async def act(self, task: Task, trajectory: Trajectory) -> PolicyOutput:
        policy_task = task.policy_view()
        messages = self._messages(policy_task, trajectory)
        generation = await self.client.generate(
            messages,
            model=self.model,
            config=self.generation,
        )
        if self.require_token_ids and generation.token_ids is None:
            raise RuntimeError("model gateway did not return actual response token IDs")
        parsed = parse_policy_output(generation.text)
        metadata = {
            "model": self.model,
            "prompt_template_version": self.prompt_template_version,
            "temperature": self.generation.temperature,
            "top_p": self.generation.top_p,
            "max_generation_tokens": self.generation.max_tokens,
            "stop_sequences": list(self.generation.stop_sequences),
            "mode": self.mode,
            "prompt_messages": messages,
            **generation.metadata,
        }
        return PolicyOutput(
            reasoning=parsed.reasoning,
            action=parsed.action,
            raw_text=generation.text,
            response_token_ids=generation.token_ids,
            response_logprobs=generation.logprobs,
            metadata=metadata,
        )

    def _messages(self, task: Task, trajectory: Trajectory) -> list[dict[str, str]]:
        system = (
            "You are a text search agent. Emit exactly one action per turn using "
            "<think>...</think> followed by exactly one of "
            "<search>query</search>, <visit>doc_id</visit>, or "
            "<answer>final answer</answer>. Do not emit code or other tags."
        )
        if self.mode == "direct":
            system += " Answer directly without using search or visit."
        elif self.mode == "retrieve_once":
            system += " Use at most one search before answering."
        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": task.question})
        for step in trajectory.steps:
            assistant = (
                f"<think>{step.thought}</think>\n"
                f"<{step.action.kind.value}>{step.action.value}</{step.action.kind.value}>"
            )
            messages.append({"role": "assistant", "content": assistant})
            if step.is_tool_step:
                messages.append(
                    {
                        "role": "user",
                        "content": f"<information>\n{step.observation}\n</information>",
                    }
                )
        return messages


class DirectAnswerPolicy(LLMPolicy):
    def __init__(
        self,
        client: ModelClient,
        *,
        model: str,
        generation: LLMGenerationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(client, model=model, generation=generation, mode="direct", **kwargs)

    async def act(self, task: Task, trajectory: Trajectory) -> PolicyOutput:
        output = await super().act(task, trajectory)
        if output.action.kind is not ActionKind.ANSWER:
            raise ValueError("direct-answer baseline emitted a tool action")
        return output


class RetrieveOncePolicy(LLMPolicy):
    """Retrieve-once baseline: force one initial search, then ask the model."""

    def __init__(
        self,
        client: ModelClient,
        *,
        model: str,
        generation: LLMGenerationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(client, model=model, generation=generation, mode="retrieve_once", **kwargs)

    async def act(self, task: Task, trajectory: Trajectory) -> PolicyOutput:
        if not trajectory.steps:
            return PolicyOutput(
                reasoning="Retrieve-once baseline issues its fixed initial query.",
                action=Action(ActionKind.SEARCH, task.question),
            )
        output = await super().act(task, trajectory)
        if any(step.is_tool_step for step in trajectory.steps) and output.action.kind is not ActionKind.ANSWER:
            raise ValueError("retrieve-once baseline emitted a second tool action")
        return output
