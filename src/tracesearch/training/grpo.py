"""Vanilla outcome-based GRPO objective, independent of a trainer backend."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, asdict
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class GRPOConfig:
    algorithm: str = "GRPO"
    group_size: int = 4
    learning_rate: float = 1e-6
    clip_epsilon: float = 0.2
    kl_coef: float = 0.0
    entropy_coef: float = 0.0
    train_batch_size: int = 1
    rollout_concurrency: int = 1
    max_prompt_length: int = 4096
    max_response_length: int = 512
    max_turns: int = 6
    search_budget: int | None = None
    visit_budget: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    optimizer: str = "adamw"
    training_steps: int = 1

    def __post_init__(self) -> None:
        if self.algorithm != "GRPO":
            raise ValueError("M1 training objective must be GRPO")
        for name in ("group_size", "train_batch_size", "rollout_concurrency", "max_prompt_length", "max_response_length", "max_turns", "training_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.clip_epsilon < 1:
            raise ValueError("clip_epsilon must be in [0, 1)")
        if self.kl_coef < 0 or self.entropy_coef < 0:
            raise ValueError("regularization coefficients must be non-negative")
        if self.temperature <= 0 or not 0 < self.top_p <= 1:
            raise ValueError("temperature/top_p are invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_group_advantages(rewards: Sequence[float], *, epsilon: float = 1e-8) -> tuple[float, ...]:
    """Normalize rewards within one group; zero-variance groups get zero."""

    if not rewards:
        return ()
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    mean = statistics.mean(float(value) for value in rewards)
    std = statistics.pstdev(float(value) for value in rewards)
    if std == 0.0:
        return tuple(0.0 for _ in rewards)
    return tuple((float(value) - mean) / (std + epsilon) for value in rewards)


def grpo_loss(
    policy_logprobs: Any,
    old_logprobs: Any,
    advantages: Any,
    response_mask: Any,
    *,
    clip_epsilon: float = 0.2,
    kl_coef: float = 0.0,
    entropy_coef: float = 0.0,
    entropy: Any | None = None,
) -> Any:
    """Compute the vanilla clipped GRPO loss over policy-generated tokens only."""

    import torch

    if policy_logprobs.shape != old_logprobs.shape:
        raise ValueError("policy_logprobs and old_logprobs must have equal shape")
    if policy_logprobs.ndim != 2:
        raise ValueError("logprob tensors must have shape [batch, response_tokens]")
    mask = response_mask.to(dtype=policy_logprobs.dtype)
    if mask.shape != policy_logprobs.shape:
        raise ValueError("response_mask must align with logprob tensors")
    advantage_tensor = advantages
    if not torch.is_tensor(advantage_tensor):
        advantage_tensor = torch.tensor(advantage_tensor, dtype=policy_logprobs.dtype, device=policy_logprobs.device)
    advantage_tensor = advantage_tensor.to(dtype=policy_logprobs.dtype, device=policy_logprobs.device)
    if advantage_tensor.ndim == 1:
        advantage_tensor = advantage_tensor.unsqueeze(1)
    if advantage_tensor.shape != policy_logprobs.shape and advantage_tensor.shape[0] == policy_logprobs.shape[0] and advantage_tensor.shape[1] == 1:
        advantage_tensor = advantage_tensor.expand_as(policy_logprobs)
    if advantage_tensor.shape != policy_logprobs.shape:
        raise ValueError("advantages must have shape [batch] or [batch, response_tokens]")
    if not 0 <= clip_epsilon < 1:
        raise ValueError("clip_epsilon must be in [0, 1)")
    ratio = torch.exp(policy_logprobs - old_logprobs)
    unclipped = ratio * advantage_tensor
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage_tensor
    objective = torch.minimum(unclipped, clipped)
    token_count = mask.sum().clamp_min(1.0)
    loss = -(objective * mask).sum() / token_count
    if kl_coef:
        loss = loss + kl_coef * (((old_logprobs - policy_logprobs) * mask).sum() / token_count)
    if entropy_coef:
        if entropy is None:
            raise ValueError("entropy tensor is required when entropy_coef is non-zero")
        loss = loss - entropy_coef * ((entropy * mask).sum() / token_count)
    return loss


@dataclass(frozen=True)
class GRPOUpdateResult:
    loss: float
    optimizer_step: int
    mean_reward: float
    group_reward_std: float
    zero_variance_group: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def optimizer_step(
    model: Any,
    optimizer: Any,
    logprob_fn: Callable[[Any], Any],
    batch: dict[str, Any],
    *,
    config: GRPOConfig,
    optimizer_step_index: int = 1,
) -> GRPOUpdateResult:
    """Run one real torch optimizer update for a caller-supplied policy model."""

    import torch

    policy_logprobs = logprob_fn(batch)
    loss = grpo_loss(
        policy_logprobs,
        batch["old_logprobs"],
        batch["advantages"],
        batch["response_mask"],
        clip_epsilon=config.clip_epsilon,
        kl_coef=config.kl_coef,
        entropy_coef=config.entropy_coef,
        entropy=batch.get("entropy"),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    rewards = [float(value) for value in batch["rewards"]]
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    return GRPOUpdateResult(
        loss=float(loss.detach().cpu().item()),
        optimizer_step=optimizer_step_index,
        mean_reward=float(statistics.mean(rewards)) if rewards else 0.0,
        group_reward_std=float(std),
        zero_variance_group=std == 0.0,
    )
