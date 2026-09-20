"""Framework-neutral token projection for RL training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import hashlib
import json

from tracesearch.data.schema import Trajectory


class TokenProvenanceError(ValueError):
    """Raised when a training trace would require re-tokenizing model text."""


@dataclass(frozen=True)
class GenerationRecord:
    """Exact provenance for one model generation call.

    ``prompt_ids`` may be unavailable for an opaque gateway, but a
    training-ready local record must always contain it.  The response IDs are
    copied directly from the backend generation result and are never rebuilt
    from decoded text.
    """

    step_index: int
    prompt_ids: tuple[int, ...] | None
    response_ids: tuple[int, ...]
    response_logprobs: tuple[float, ...] | None = None
    prompt_messages: tuple[dict[str, str], ...] = ()
    model: str = ""
    checkpoint: str | None = None
    tokenizer_version: str | None = None
    template_version: str | None = None
    sampling_config: dict[str, Any] = field(default_factory=dict)
    sampling_seed: int | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise TokenProvenanceError("generation step_index must be non-negative")
        if self.prompt_ids is not None:
            object.__setattr__(self, "prompt_ids", tuple(self.prompt_ids))
            if not all(isinstance(token, int) for token in self.prompt_ids):
                raise TokenProvenanceError("prompt_ids must be actual integer IDs")
        object.__setattr__(self, "response_ids", tuple(self.response_ids))
        if not all(isinstance(token, int) for token in self.response_ids):
            raise TokenProvenanceError("response_ids must be actual integer IDs")
        if self.response_logprobs is not None:
            object.__setattr__(self, "response_logprobs", tuple(float(value) for value in self.response_logprobs))
            if len(self.response_logprobs) != len(self.response_ids):
                raise TokenProvenanceError("response_logprobs must align with response_ids")
        object.__setattr__(self, "prompt_messages", tuple(dict(message) for message in self.prompt_messages))
        if self.sampling_seed is not None and self.sampling_seed < 0:
            raise TokenProvenanceError("sampling_seed must be non-negative")

    @property
    def prompt_fingerprint(self) -> str | None:
        if self.prompt_ids is None:
            return None
        payload = json.dumps(list(self.prompt_ids), separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "prompt_ids": list(self.prompt_ids) if self.prompt_ids is not None else None,
            "prompt_fingerprint": self.prompt_fingerprint,
            "response_ids": list(self.response_ids),
            "response_logprobs": list(self.response_logprobs) if self.response_logprobs is not None else None,
            "prompt_messages": [dict(message) for message in self.prompt_messages],
            "model": self.model,
            "checkpoint": self.checkpoint,
            "tokenizer_version": self.tokenizer_version,
            "template_version": self.template_version,
            "sampling_config": dict(self.sampling_config),
            "sampling_seed": self.sampling_seed,
            "finish_reason": self.finish_reason,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationRecord":
        prompt_ids = data.get("prompt_ids")
        return cls(
            step_index=int(data["step_index"]),
            prompt_ids=tuple(prompt_ids) if prompt_ids is not None else None,
            response_ids=tuple(data.get("response_ids", ())),
            response_logprobs=(
                tuple(data["response_logprobs"])
                if data.get("response_logprobs") is not None
                else None
            ),
            prompt_messages=tuple(data.get("prompt_messages", ())),
            model=str(data.get("model", "")),
            checkpoint=data.get("checkpoint"),
            tokenizer_version=data.get("tokenizer_version"),
            template_version=data.get("template_version"),
            sampling_config=dict(data.get("sampling_config", {})),
            sampling_seed=data.get("sampling_seed"),
            finish_reason=data.get("finish_reason"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class TokenTurn:
    role: str
    token_ids: tuple[int, ...]
    generated_by_policy: bool = False
    step_index: int | None = None

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported turn role: {self.role}")
        if not all(isinstance(token, int) for token in self.token_ids):
            raise TokenProvenanceError("token_ids must be actual integer IDs")
        if self.generated_by_policy and self.role != "assistant":
            raise ValueError("only assistant turns may be policy-generated")
        if self.step_index is not None and self.step_index < 0:
            raise ValueError("step_index must be non-negative")


@dataclass(frozen=True)
class TrainingTrace:
    task_id: str
    rollout_id: str
    sample_index: int
    policy_version: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    step_ids: tuple[int, ...]
    old_logprobs: tuple[float, ...] | None = None
    reference_logprobs: tuple[float, ...] | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    total_reward: float = 0.0
    sampling_metadata: dict[str, Any] = field(default_factory=dict)
    generation_records: tuple[GenerationRecord, ...] = ()
    observation_token_ids: dict[int, tuple[int, ...]] = field(default_factory=dict)
    group_advantage: float = 0.0
    flattened_sequence_exact: bool = False

    def __post_init__(self) -> None:
        if len(self.response_ids) != len(self.response_mask) or len(self.response_ids) != len(self.step_ids):
            raise ValueError("response_ids, response_mask, and step_ids must have equal lengths")
        if any(mask not in (0, 1) for mask in self.response_mask):
            raise ValueError("response_mask values must be 0 or 1")
        if self.old_logprobs is not None and len(self.old_logprobs) != len(self.response_ids):
            raise ValueError("old_logprobs must align with response_ids")
        if self.reference_logprobs is not None and len(self.reference_logprobs) != len(self.response_ids):
            raise ValueError("reference_logprobs must align with response_ids")
        if any(mask == 1 and step_id < 0 for mask, step_id in zip(self.response_mask, self.step_ids)):
            raise TokenProvenanceError("every policy token must map to a non-negative generation step")
        records = tuple(self.generation_records)
        object.__setattr__(self, "generation_records", records)
        if records:
            steps = [record.step_index for record in records]
            if steps != sorted(set(steps)):
                raise TokenProvenanceError("generation records must have unique sorted step indices")
        object.__setattr__(
            self,
            "observation_token_ids",
            {int(step): tuple(tokens) for step, tokens in self.observation_token_ids.items()},
        )

    @property
    def input_ids(self) -> tuple[int, ...]:
        if self.generation_records and not self.flattened_sequence_exact:
            raise TokenProvenanceError(
                "multi-turn records are turn-wise; use generation_records instead of a fake flattened sequence"
            )
        return self.prompt_ids + self.response_ids

    @property
    def full_response_mask(self) -> tuple[int, ...]:
        if self.generation_records and not self.flattened_sequence_exact:
            raise TokenProvenanceError(
                "multi-turn records do not have one flattened chat sequence; inspect turn_masks"
            )
        return (0,) * len(self.prompt_ids) + self.response_mask

    @property
    def turn_masks(self) -> tuple[tuple[int, ...], ...]:
        """Loss masks for each exact prompt/response generation boundary."""

        if not self.generation_records:
            return (self.full_response_mask,)
        return tuple(
            (0,) * len(record.prompt_ids or ()) + (1,) * len(record.response_ids)
            for record in self.generation_records
        )

    @classmethod
    def from_turns(
        cls,
        *,
        task_id: str,
        rollout_id: str,
        sample_index: int,
        policy_version: str,
        turns: Iterable[TokenTurn],
        reward_components: dict[str, float] | None = None,
        total_reward: float = 0.0,
        group_advantage: float = 0.0,
        sampling_metadata: dict[str, Any] | None = None,
    ) -> "TrainingTrace":
        values = list(turns)
        first_generated = next((index for index, turn in enumerate(values) if turn.generated_by_policy), len(values))
        prompt_ids = tuple(token for turn in values[:first_generated] for token in turn.token_ids)
        response_turns = values[first_generated:]
        response_ids = tuple(token for turn in response_turns for token in turn.token_ids)
        response_mask = tuple(
            int(turn.generated_by_policy)
            for turn in response_turns
            for _ in turn.token_ids
        )
        step_ids = tuple(
            turn.step_index if turn.generated_by_policy and turn.step_index is not None else -1
            for turn in response_turns
            for _ in turn.token_ids
        )
        return cls(
            task_id=task_id,
            rollout_id=rollout_id,
            sample_index=sample_index,
            policy_version=policy_version,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            step_ids=step_ids,
            reward_components=dict(reward_components or {}),
            total_reward=total_reward,
            sampling_metadata=dict(sampling_metadata or {}),
            flattened_sequence_exact=True,
        )

    @classmethod
    def from_generation_records(
        cls,
        *,
        task_id: str,
        rollout_id: str,
        sample_index: int,
        policy_version: str,
        generation_records: Iterable[GenerationRecord],
        tool_step_indices: Iterable[int] = (),
        observation_token_ids: dict[int, Iterable[int]] | None = None,
        reward_components: dict[str, float] | None = None,
        total_reward: float = 0.0,
        group_advantage: float = 0.0,
        sampling_metadata: dict[str, Any] | None = None,
    ) -> "TrainingTrace":
        """Build a turn-wise trace from exact backend records.

        The first prompt is retained for compatibility, but multi-turn
        records are never presented as one naively concatenated chat input.
        Observation IDs are explicitly zero-loss spans in the response
        projection and are not used to derive policy log-probabilities.
        """

        records = tuple(generation_records)
        if not records:
            raise TokenProvenanceError("at least one generation record is required")
        if any(record.prompt_ids is None for record in records):
            raise TokenProvenanceError("exact local training requires prompt_ids for every generation record")
        if any(record.response_logprobs is None for record in records):
            raise TokenProvenanceError("old logprobs are required for every generated response")
        tool_steps = {int(step) for step in tool_step_indices}
        observations = {
            int(step): tuple(tokens) for step, tokens in (observation_token_ids or {}).items()
        }
        missing = sorted(tool_steps - observations.keys())
        if missing:
            raise TokenProvenanceError(f"missing exact observation token IDs for steps: {missing}")
        response_ids: list[int] = []
        response_mask: list[int] = []
        step_ids: list[int] = []
        old_logprobs: list[float] = []
        for record in records:
            response_ids.extend(record.response_ids)
            response_mask.extend([1] * len(record.response_ids))
            step_ids.extend([record.step_index] * len(record.response_ids))
            old_logprobs.extend(record.response_logprobs or ())
            if record.step_index in tool_steps:
                response_ids.extend(observations[record.step_index])
                response_mask.extend([0] * len(observations[record.step_index]))
                step_ids.extend([-1] * len(observations[record.step_index]))
                old_logprobs.extend([0.0] * len(observations[record.step_index]))
        metadata = dict(sampling_metadata or {})
        metadata.setdefault("representation", "turn-wise-exact")
        metadata.setdefault("generation_count", len(records))
        return cls(
            task_id=task_id,
            rollout_id=rollout_id,
            sample_index=sample_index,
            policy_version=policy_version,
            prompt_ids=tuple(records[0].prompt_ids or ()),
            response_ids=tuple(response_ids),
            response_mask=tuple(response_mask),
            step_ids=tuple(step_ids),
            old_logprobs=tuple(old_logprobs),
            reward_components=dict(reward_components or {}),
            total_reward=total_reward,
            sampling_metadata=metadata,
            generation_records=records,
            observation_token_ids=observations,
            group_advantage=group_advantage,
        )

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        *,
        prompt_ids: Iterable[int] | None = None,
        observation_token_ids: dict[int, Iterable[int]] | None = None,
        policy_version: str,
        reward_components: dict[str, float] | None = None,
        total_reward: float = 0.0,
        group_advantage: float = 0.0,
        sampling_metadata: dict[str, Any] | None = None,
    ) -> "TrainingTrace":
        """Project a rollout using gateway IDs; never reconstruct model output IDs."""

        raw_record_values = [step.metadata.get("generation_record") for step in trajectory.steps]
        has_any_record = any(value is not None for value in raw_record_values)
        if has_any_record and not all(isinstance(value, dict) for value in raw_record_values):
            raise TokenProvenanceError("generation provenance is incomplete for the trajectory")
        raw_records = [GenerationRecord.from_dict(value) for value in raw_record_values if isinstance(value, dict)]
        if raw_records:
            expected_steps = [step.step_index for step in trajectory.steps]
            if [record.step_index for record in raw_records] != expected_steps:
                raise TokenProvenanceError("generation record steps do not match trajectory steps")
            tool_steps = [step.step_index for step in trajectory.steps if step.is_tool_step]
            return cls.from_generation_records(
                task_id=trajectory.task_id or "",
                rollout_id=trajectory.rollout_id or "",
                sample_index=trajectory.sample_index,
                policy_version=policy_version,
                generation_records=raw_records,
                tool_step_indices=tool_steps,
                observation_token_ids=observation_token_ids,
                reward_components=reward_components,
                total_reward=total_reward,
                group_advantage=group_advantage,
                sampling_metadata=sampling_metadata,
            )
        if prompt_ids is None:
            raise TokenProvenanceError("prompt_ids are required when generation records are unavailable")
        turns = [TokenTurn("user", tuple(prompt_ids))]
        for step in trajectory.steps:
            raw_response_ids = step.metadata.get("response_token_ids")
            if not isinstance(raw_response_ids, list) or not all(isinstance(token, int) for token in raw_response_ids):
                raise TokenProvenanceError(
                    f"step {step.step_index} has no actual response token IDs"
                )
            turns.append(
                TokenTurn(
                    "assistant",
                    tuple(raw_response_ids),
                    generated_by_policy=True,
                    step_index=step.step_index,
                )
            )
            if step.is_tool_step:
                if observation_token_ids is None or step.step_index not in observation_token_ids:
                    raise TokenProvenanceError(
                        f"step {step.step_index} has no observation token IDs"
                    )
                turns.append(TokenTurn("tool", tuple(observation_token_ids[step.step_index])))
        return cls.from_turns(
            task_id=trajectory.task_id or "",
            rollout_id=trajectory.rollout_id or "",
            sample_index=trajectory.sample_index,
            policy_version=policy_version,
            turns=turns,
            reward_components=reward_components,
            total_reward=total_reward,
            group_advantage=group_advantage,
            sampling_metadata=sampling_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "rollout_id": self.rollout_id,
            "sample_index": self.sample_index,
            "policy_version": self.policy_version,
            "prompt_ids": list(self.prompt_ids),
            "response_ids": list(self.response_ids),
            "response_mask": list(self.response_mask),
            "step_ids": list(self.step_ids),
            "old_logprobs": list(self.old_logprobs) if self.old_logprobs is not None else None,
            "reference_logprobs": list(self.reference_logprobs) if self.reference_logprobs is not None else None,
            "reward_components": dict(self.reward_components),
            "total_reward": self.total_reward,
            "sampling_metadata": dict(self.sampling_metadata),
            "generation_records": [record.to_dict() for record in self.generation_records],
            "observation_token_ids": {str(step): list(tokens) for step, tokens in self.observation_token_ids.items()},
            "group_advantage": self.group_advantage,
            "flattened_sequence_exact": self.flattened_sequence_exact,
        }


def build_training_trace(**kwargs: Any) -> TrainingTrace:
    return TrainingTrace.from_turns(**kwargs)
