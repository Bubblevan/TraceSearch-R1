"""Framework-neutral token projection for RL training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from tracesearch.data.schema import Trajectory


class TokenProvenanceError(ValueError):
    """Raised when a training trace would require re-tokenizing model text."""


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
    reward_components: dict[str, float] = field(default_factory=dict)
    total_reward: float = 0.0
    sampling_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.response_ids) != len(self.response_mask) or len(self.response_ids) != len(self.step_ids):
            raise ValueError("response_ids, response_mask, and step_ids must have equal lengths")
        if any(mask not in (0, 1) for mask in self.response_mask):
            raise ValueError("response_mask values must be 0 or 1")
        if self.old_logprobs is not None and len(self.old_logprobs) != len(self.response_ids):
            raise ValueError("old_logprobs must align with response_ids")

    @property
    def input_ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.response_ids

    @property
    def full_response_mask(self) -> tuple[int, ...]:
        return (0,) * len(self.prompt_ids) + self.response_mask

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
        )

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        *,
        prompt_ids: Iterable[int],
        observation_token_ids: dict[int, Iterable[int]],
        policy_version: str,
        reward_components: dict[str, float] | None = None,
        total_reward: float = 0.0,
        sampling_metadata: dict[str, Any] | None = None,
    ) -> "TrainingTrace":
        """Project a rollout using gateway IDs; never reconstruct model output IDs."""

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
                if step.step_index not in observation_token_ids:
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
            "reward_components": dict(self.reward_components),
            "total_reward": self.total_reward,
            "sampling_metadata": dict(self.sampling_metadata),
        }


def build_training_trace(**kwargs: Any) -> TrainingTrace:
    return TrainingTrace.from_turns(**kwargs)
