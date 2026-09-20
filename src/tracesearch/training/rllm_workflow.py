"""TraceSearch workflow adapter for the upstream rLLM/veRL backend.

The module is intentionally importable without rLLM installed.  The native
Windows environment remains the reference M1-B environment; only a Linux
environment with the optional backend can instantiate ``TraceSearchWorkflow``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from tracesearch.agent.llm import LLMGenerationConfig, ModelGeneration
from tracesearch.agent.policy import LLMPolicy
from tracesearch.agent.loop import SearchAgent
from tracesearch.data.io import load_corpus
from tracesearch.data.schema import ActionKind, Task, TerminationReason, Trajectory
from tracesearch.environment import Corpus, LocalSearchEnvironment, LocalSearchTool, LocalVisitTool
from tracesearch.training.reward import compute_outcome_reward


try:  # Optional: the Windows/core test environment does not install rLLM.
    from rllm.engine.rollout import ModelOutput as _RLLMModelOutput
    from rllm.types import Episode as _RLLMEpisode
    from rllm.types import Step as _RLLMStep
    from rllm.types import Trajectory as _RLLMTrajectory
    from rllm.workflows.workflow import TerminationReason as _RLLMTerminationReason
    from rllm.workflows.workflow import Workflow as _RLLMWorkflow
except ImportError:  # pragma: no cover - exercised only when the optional backend is absent.
    _RLLMModelOutput = None
    _RLLMEpisode = None
    _RLLMStep = None
    _RLLMTrajectory = None
    _RLLMTerminationReason = None

    class _RLLMWorkflow:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs


_LOG_LOCK = threading.Lock()


class RLLMModelClient:
    """Adapt rLLM's token-in/token-out engine to TraceSearch ``ModelClient``."""

    def __init__(self, rollout_engine: Any, *, checkpoint: str, disable_thinking: bool = True) -> None:
        self.rollout_engine = rollout_engine
        self.checkpoint = checkpoint
        self.disable_thinking = disable_thinking

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        config: LLMGenerationConfig,
    ) -> ModelGeneration:
        # rLLM/veRL receives these as sampling_params.  ``None`` is converted
        # to vLLM's explicit disabled top-k value, preserving the M1-C contract.
        params: dict[str, Any] = {
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k if config.top_k is not None else -1,
            "presence_penalty": config.presence_penalty,
            "repetition_penalty": config.repetition_penalty,
        }
        if config.sampling_seed is not None:
            params["seed"] = config.sampling_seed
        if config.stop_sequences:
            params["stop"] = list(config.stop_sequences)
        output = await self.rollout_engine.get_model_response(
            messages,
            **params,
        )
        response_ids = tuple(int(token) for token in (output.completion_ids or []))
        prompt_ids = tuple(int(token) for token in (output.prompt_ids or []))
        response_logprobs = output.logprobs
        effective_top_k = config.top_k if config.top_k is not None else -1
        return ModelGeneration(
            text=str(output.text or output.content or ""),
            token_ids=response_ids,
            logprobs=(tuple(float(value) for value in response_logprobs) if response_logprobs is not None else None),
            prompt_token_ids=prompt_ids,
            sampling_seed=config.sampling_seed,
            finish_reason=output.finish_reason,
            metadata={
                "checkpoint": self.checkpoint,
                "model_path": self.checkpoint,
                "backend": "rllm-verl-vllm",
                "backend_weight_version": output.weight_version,
                "sampling_config": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": effective_top_k,
                    "presence_penalty": config.presence_penalty,
                    "repetition_penalty": config.repetition_penalty,
                    "enable_thinking": not self.disable_thinking,
                    "max_tokens": config.max_tokens,
                },
            },
        )


class TraceSearchWorkflow(_RLLMWorkflow):
    """Run the canonical TraceSearch loop inside rLLM's Workflow boundary."""

    def __init__(
        self,
        rollout_engine: Any,
        executor: Any,
        *,
        corpus_path: str,
        checkpoint: str,
        top_k: int = 5,
        max_turns: int = 4,
        max_tokens: int = 128,
        seed: int = 42,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k_sampling: int | None = None,
        artifact_dir: str | None = None,
        disable_thinking: bool = True,
        **kwargs: Any,
    ) -> None:
        if _RLLMModelOutput is None:
            raise RuntimeError("TraceSearchWorkflow requires the optional rLLM/veRL backend")
        super().__init__(rollout_engine, executor, **kwargs)
        self.corpus_path = str(corpus_path)
        self.checkpoint = str(checkpoint)
        self.top_k = int(top_k)
        self.max_turns = int(max_turns)
        self.max_tokens = int(max_tokens)
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k_sampling = top_k_sampling
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.disable_thinking = bool(disable_thinking)
        self._environment = LocalSearchEnvironment(Corpus(load_corpus(self.corpus_path)), top_k=self.top_k)

    async def run(self, task: dict[str, Any], uid: str, **kwargs: Any) -> Any:
        del kwargs
        canonical_task = Task.from_dict(task)
        task_id, sample_index = self._split_uid(uid, canonical_task.task_id)
        generation = LLMGenerationConfig(
            # M1-C initial on-policy contract: no truncation/penalty warpers.
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k_sampling,
            presence_penalty=0.0,
            repetition_penalty=1.0,
            enable_thinking=not self.disable_thinking,
            # Keep the base experiment seed fixed while giving each member of
            # a task group a stable, distinct sampling stream.
            sampling_seed=self.seed + sample_index,
            max_tokens=self.max_tokens,
        )
        policy = LLMPolicy(
            RLLMModelClient(self.rollout_engine, checkpoint=self.checkpoint, disable_thinking=self.disable_thinking),
            model=self.checkpoint,
            generation=generation,
            require_token_ids=True,
        )
        agent = SearchAgent(
            policy,
            LocalSearchTool(self._environment),
            LocalVisitTool(self._environment),
            max_turns=self.max_turns,
            seed=self.seed,
            rollout_id=uid,
            sample_index=sample_index,
        )
        trajectory = await agent.run_async(canonical_task, rollout_id=uid, sample_index=sample_index)
        reward = compute_outcome_reward(canonical_task, trajectory)
        episode = self._episode(canonical_task, trajectory, reward.total, task_id, uid)
        self._write_artifact(canonical_task, trajectory, reward.to_dict(), uid, sample_index)
        return episode

    def _episode(self, task: Task, trajectory: Trajectory, reward: float, task_id: str, uid: str) -> Any:
        if _RLLMEpisode is None or _RLLMTrajectory is None or _RLLMStep is None or _RLLMModelOutput is None:
            raise RuntimeError("rLLM backend types are unavailable")
        rllm_steps = []
        for step in trajectory.steps:
            raw_record = step.metadata.get("generation_record")
            if raw_record is None:
                continue
            prompt_ids = [int(token) for token in raw_record.get("prompt_ids") or []]
            response_ids = [int(token) for token in raw_record.get("response_ids") or []]
            logprobs = [float(value) for value in raw_record.get("response_logprobs") or []]
            content = step.metadata.get("policy_raw_output", "")
            model_output = _RLLMModelOutput(
                text=content,
                content=content,
                reasoning=step.thought,
                prompt_ids=prompt_ids,
                completion_ids=response_ids,
                logprobs=logprobs,
                finish_reason=raw_record.get("finish_reason"),
            )
            messages = [dict(message) for message in raw_record.get("prompt_messages") or []]
            messages.append({"role": "assistant", "content": content, "reasoning": step.thought})
            rllm_steps.append(
                _RLLMStep(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    logprobs=logprobs,
                    chat_completions=messages,
                    observation=step.observation,
                    thought=step.thought,
                    action={"kind": step.action.kind.value, "value": step.action.value},
                    model_response=content,
                    model_output=model_output,
                    metadata={
                        "task_id": task_id,
                        "rollout_id": uid,
                        "sample_index": trajectory.sample_index,
                        "step_index": step.step_index,
                    },
                    reward=0.0,
                    done=step.action.kind is ActionKind.ANSWER,
                )
            )
        rllm_trajectory = _RLLMTrajectory(
            uid=uid,
            name="search",
            task=task.policy_view().to_dict(),
            steps=rllm_steps,
            reward=float(reward),
            metadata={
                "task_id": task_id,
                "rollout_id": uid,
                "sample_index": trajectory.sample_index,
                "tracesearch_termination": trajectory.termination_reason.value,
            },
        )
        termination = {
            TerminationReason.ANSWER: "env_done",
            TerminationReason.MAX_TURNS: "max_turns_exceeded",
            TerminationReason.TOOL_BUDGET_EXHAUSTED: "max_turns_exceeded",
            TerminationReason.POLICY_ERROR: "error",
            TerminationReason.ENVIRONMENT_ERROR: "error",
            TerminationReason.FATAL_TOOL_FAILURE: "error",
        }[trajectory.termination_reason]
        return _RLLMEpisode(
            id=uid,
            task=task.policy_view().to_dict(),
            termination_reason=_RLLMTerminationReason(termination),
            is_correct=bool(reward > 0.0),
            trajectories=[rllm_trajectory],
            metrics={"reward": float(reward)},
            metadata={"task_id": task_id, "sample_index": trajectory.sample_index},
        )

    def _write_artifact(
        self,
        task: Task,
        trajectory: Trajectory,
        reward: dict[str, Any],
        uid: str,
        sample_index: int,
    ) -> None:
        if self.artifact_dir is None:
            return
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "task_id": task.task_id,
            "rollout_id": uid,
            "sample_index": sample_index,
            "reward": reward,
            "trajectory": trajectory.to_dict(),
        }
        destination = self.artifact_dir / "tracesearch_rollouts.jsonl"
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        with _LOG_LOCK:
            with destination.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)

    @staticmethod
    def _split_uid(uid: str, fallback_task_id: str) -> tuple[str, int]:
        if ":" not in uid:
            return fallback_task_id, 0
        task_id, raw_index = uid.rsplit(":", 1)
        try:
            return task_id, int(raw_index)
        except ValueError:
            return task_id, 0


def backend_import_status() -> dict[str, Any]:
    """Return import/version status without importing the backend eagerly."""

    import importlib
    import importlib.metadata as metadata

    result: dict[str, Any] = {}
    for package in ("rllm", "verl", "vllm", "torch", "transformers"):
        try:
            result[package] = {"version": metadata.version(package), "importable": True}
            importlib.import_module(package)
        except Exception as exc:  # pragma: no cover - depends on the Linux backend environment.
            result[package] = {"version": result.get(package, {}).get("version"), "importable": False, "error": f"{type(exc).__name__}: {exc}"}
    return result
