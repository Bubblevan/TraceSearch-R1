"""Model-vendor-neutral policy protocols and deterministic fixture policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol

from tracesearch.agent.llm import (
    SAMPLING_SEED_SCHEME,
    LLMGenerationConfig,
    ModelClient,
    derive_sampling_seed,
)
from tracesearch.agent.parser import ActionParseError, parse_policy_output
from tracesearch.data.schema import Action, ActionKind, Task, Trajectory

if TYPE_CHECKING:
    from tracesearch.training.trace import GenerationRecord


@dataclass(frozen=True)
class PolicyOutput:
    reasoning: str
    action: Action
    raw_text: str | None = None
    response_token_ids: tuple[int, ...] | None = None
    response_logprobs: tuple[float, ...] | None = None
    prompt_token_ids: tuple[int, ...] | None = None
    generation_record: Any | None = None
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
        sampling_seed = derive_sampling_seed(
            task.task_id,
            trajectory.sample_index,
            len(trajectory.steps),
            base_seed=self.generation.sampling_seed,
        )
        generation_config = replace(self.generation, sampling_seed=sampling_seed)
        try:
            generation = await self.client.generate(
                messages,
                model=self.model,
                config=generation_config,
            )
        except Exception as exc:
            # A backend termination can happen before prompt IDs are returned.
            # Preserve the exact attempted seed without fabricating token counts.
            exc.sampling_seed = sampling_seed  # type: ignore[attr-defined]
            exc.sampling_seed_scheme = SAMPLING_SEED_SCHEME  # type: ignore[attr-defined]
            exc.generation_step_index = len(trajectory.steps)  # type: ignore[attr-defined]
            raise
        record = self._generation_record(trajectory, messages, generation, generation_config)
        if self.require_token_ids and generation.token_ids is None:
            error = RuntimeError("model gateway did not return actual response token IDs")
            error.generation_record = record  # type: ignore[attr-defined]
            raise error
        try:
            parsed = parse_policy_output(generation.text)
        except ActionParseError as exc:
            exc.raw_text = generation.text
            exc.generation_record = record  # type: ignore[attr-defined]
            raise
        metadata = {
            "model": self.model,
            "prompt_template_version": self.prompt_template_version,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "top_k": generation_config.top_k,
            "presence_penalty": generation_config.presence_penalty,
            "repetition_penalty": generation_config.repetition_penalty,
            "enable_thinking": generation_config.enable_thinking,
            "sampling_seed": generation.sampling_seed if generation.sampling_seed is not None else sampling_seed,
            "sampling_seed_scheme": SAMPLING_SEED_SCHEME,
            "max_generation_tokens": generation_config.max_tokens,
            "stop_sequences": list(generation_config.stop_sequences),
            "mode": self.mode,
            "prompt_messages": messages,
            "generation_record": record.to_dict(),
            **generation.metadata,
        }
        return PolicyOutput(
            reasoning=parsed.reasoning,
            action=parsed.action,
            raw_text=generation.text,
            response_token_ids=generation.token_ids,
            response_logprobs=generation.logprobs,
            prompt_token_ids=generation.prompt_token_ids,
            generation_record=record,
            metadata=metadata,
        )

    def _generation_record(
        self,
        trajectory: Trajectory,
        messages: list[dict[str, str]],
        generation: Any,
        config: LLMGenerationConfig,
    ) -> GenerationRecord:
        from tracesearch.training.trace import GenerationRecord

        if generation.token_ids is None:
            raise RuntimeError("generation record requires actual response token IDs")
        metadata = dict(generation.metadata)
        metadata.setdefault("sampling_seed_scheme", SAMPLING_SEED_SCHEME)
        sampling_config = dict(metadata.get("sampling_config", {}))
        if not sampling_config:
            sampling_config = {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "presence_penalty": config.presence_penalty,
                "repetition_penalty": config.repetition_penalty,
                "enable_thinking": config.enable_thinking,
                "max_tokens": config.max_tokens,
            }
        return GenerationRecord(
            step_index=len(trajectory.steps),
            prompt_ids=generation.prompt_token_ids,
            response_ids=generation.token_ids,
            response_logprobs=generation.logprobs,
            prompt_messages=tuple(dict(message) for message in messages),
            model=self.model,
            checkpoint=metadata.get("checkpoint") or metadata.get("model_path"),
            tokenizer_version=metadata.get("tokenizer_version"),
            template_version=metadata.get("template_version") or self.prompt_template_version,
            sampling_config=sampling_config,
            sampling_seed=(
                generation.sampling_seed
                if generation.sampling_seed is not None
                else config.sampling_seed
            ),
            finish_reason=generation.finish_reason,
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
        else:
            system += " For factual questions, the first turn must search before answering. Use the returned information before giving the final answer."
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
