"""Runtime-only evidence collection for the pinned rLLM/veRL M1-C path.

The observer is deliberately a TraceSearch wrapper.  It does not modify the
installed rLLM or veRL sources and records raw backend metric names alongside
the small normalized view used by the run report.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from tracesearch.training.grpo import compute_group_advantages


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if hasattr(value, "detach"):
        try:
            return _jsonable(value.detach().cpu().tolist())
        except Exception:
            return str(value)
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            return str(value)
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _groups(state: Any) -> list[dict[str, Any]]:
    rows = []
    for group in getattr(state, "trajectory_groups", None) or []:
        trajectories = list(getattr(group, "trajectories", []) or [])
        rewards = [float(getattr(item, "reward", 0.0) or 0.0) for item in trajectories]
        metadata = [getattr(item, "metadata", None) or {} for item in trajectories]
        rows.append(
            {
                "group_id": str(getattr(group, "group_id", "")),
                "rollout_ids": [str(getattr(item, "uid", "")) for item in trajectories],
                "task_ids": [str(item.get("task_id", "")) for item in metadata],
                "rewards": rewards,
                "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
                "reward_std": (sum((reward - (sum(rewards) / len(rewards))) ** 2 for reward in rewards) / len(rewards)) ** 0.5 if rewards else 0.0,
                "group_exact_match_variance": (
                    sum((reward - (sum(rewards) / len(rewards))) ** 2 for reward in rewards) / len(rewards)
                    if rewards
                    else 0.0
                ),
                "zero_variance": len(set(rewards)) <= 1,
            }
        )
    return rows


def _masked_values(batch: Any, key: str) -> list[float]:
    tensor = batch.batch.get(key) if hasattr(batch, "batch") else None
    if tensor is None:
        return []
    mask = batch.batch.get("response_mask") if hasattr(batch, "batch") else None
    values = tensor.detach().float()
    if mask is not None:
        values = values[mask.bool()]
    return [float(item) for item in values.reshape(-1).cpu().tolist()]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return list(value)


def _row(batch: Any, key: str, index: int) -> list[Any]:
    tensor = batch.batch.get(key) if hasattr(batch, "batch") else None
    if tensor is None:
        return []
    return _as_list(tensor[index])


def _non_tensor_rows(batch: Any, key: str) -> list[Any]:
    values = getattr(batch, "non_tensor_batch", {}).get(key, []) if hasattr(batch, "non_tensor_batch") else []
    return _as_list(values)


def _trajectory_segments(trajectory: Any) -> list[dict[str, Any]]:
    """Reconstruct the pinned rLLM cumulative response representation."""

    valid_steps = [
        step
        for step in getattr(trajectory, "steps", []) or []
        if getattr(step, "model_output", None) is not None
        and getattr(step.model_output, "prompt_ids", None) is not None
    ]
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for step_index, step in enumerate(valid_steps):
        model_output = step.model_output
        prompt = [int(value) for value in _as_list(model_output.prompt_ids)]
        action = [int(value) for value in _as_list(model_output.completion_ids)]
        if current is None or prompt[: len(current["full_seq"])] != current["full_seq"]:
            if current is not None:
                segments.append(current)
            current = {
                "response_ids": action.copy(),
                "response_mask": [1] * len(action),
                "full_seq": prompt + action,
                "step_boundaries": [{"step_index": step_index, "observation_length": 0, "action_length": len(action)}],
            }
            continue

        observation = prompt[len(current["full_seq"]):]
        current["response_ids"].extend(observation)
        current["response_ids"].extend(action)
        current["response_mask"].extend([0] * len(observation))
        current["response_mask"].extend([1] * len(action))
        current["full_seq"].extend(observation)
        current["full_seq"].extend(action)
        current["step_boundaries"].append(
            {"step_index": step_index, "observation_length": len(observation), "action_length": len(action)}
        )

    if current is not None:
        current.pop("full_seq", None)
        segments.append(current)
    return segments


class RuntimeTrainingObserver:
    """Collect one durable JSON record per real training batch."""

    def __init__(self, output_dir: str | Path, *, phase: str) -> None:
        self.output_dir = Path(output_dir)
        self.phase = phase
        self.records: list[dict[str, Any]] = []
        self.actor_update_calls = 0
        self.parameter_probe_files: set[str] = set()
        self.live_mask: dict[str, Any] | None = None
        self.live_advantage: dict[str, Any] | None = None
        self.logprob_diagnostics: list[dict[str, Any]] = []
        self.update_gate: str | None = None
        self.pre_update_snapshot: str | None = None

    def note_actor_update(self) -> None:
        self.actor_update_calls += 1

    def should_skip_actor_update(self) -> bool:
        return self.phase == "c1" and self.update_gate is not None

    def capture_pre_update(self, backend: Any, trainer_state: Any) -> None:
        """Save a real actor-worker snapshot only after both parity gates pass."""

        if self.phase != "c1" or self.update_gate is not None or self.pre_update_snapshot is not None:
            return
        target = self.output_dir / "pre_update" / "actor"
        target.mkdir(parents=True, exist_ok=True)
        try:
            backend.actor_rollout_wg.save_checkpoint(
                str(target),
                global_step=max(0, int(getattr(trainer_state, "global_step", 1)) - 1),
                max_ckpt_to_keep=1,
            )
        except Exception as exc:
            self.update_gate = "update_failed"
            _write_json(
                self.output_dir / "pre_update_error.json",
                {"exception": type(exc).__name__, "message": str(exc)},
            )
            return
        self.pre_update_snapshot = str(target)

    def observe_process_batch(self, state: Any) -> None:
        batch = getattr(state, "backend_batch", None)
        if batch is None:
            return
        old = _masked_values(batch, "old_log_probs")
        rollout = _masked_values(batch, "rollout_log_probs")
        if old and rollout and len(old) == len(rollout):
            diffs = [left - right for left, right in zip(old, rollout, strict=True)]
            diagnostics = {
                "masked_mean_old_minus_rollout": sum(diffs) / len(diffs),
                "max_abs_difference": max(abs(item) for item in diffs),
                "token_count": len(diffs),
                "tis_mode": None,
                "kl_beta": 0.0,
                "backend_offpolicy_metrics": {
                    str(key): _jsonable(value)
                    for key, value in getattr(state, "metrics", {}).items()
                    if str(key).startswith("offpolicy/") or "rollout_correction" in str(key)
                },
            }
            self.logprob_diagnostics.append(diagnostics)
            _write_json(self.output_dir / "rollout_logprob_diagnostics.json", {"records": self.logprob_diagnostics})
        self.live_mask = self._live_mask_parity(state)
        _write_json(self.output_dir / "live_mask_parity.json", self.live_mask)

    def observe_advantages(self, state: Any) -> None:
        batch = getattr(state, "backend_batch", None)
        if batch is None:
            return
        advantage = batch.batch.get("advantages") if hasattr(batch, "batch") else None
        mask = batch.batch.get("response_mask") if hasattr(batch, "batch") else None
        step_ids = _non_tensor_rows(batch, "step_ids")
        is_pad = _non_tensor_rows(batch, "is_pad_step")
        backend_by_uid: dict[str, list[float]] = {}
        if advantage is not None and mask is not None and step_ids:
            for index, step_id in enumerate(step_ids):
                if index < len(is_pad) and bool(is_pad[index]):
                    continue
                raw_values = _as_list(advantage[index])
                raw_mask = _as_list(mask[index])
                values = [float(value) for value, active in zip(raw_values, raw_mask, strict=False) if bool(active)]
                if values:
                    backend_by_uid.setdefault(str(step_id), []).append(sum(values) / len(values))
        groups = []
        for group in _groups(state):
            local = list(compute_group_advantages(group["rewards"]))
            backend: list[float | None] = []
            errors: list[float] = []
            row_consistency = True
            for uid in group["rollout_ids"]:
                values = backend_by_uid.get(uid, [])
                if values and max(values) - min(values) > 1e-6:
                    row_consistency = False
                backend.append(values[0] if values else None)
            if all(item is not None for item in backend) and row_consistency:
                errors = [abs(left - float(right)) for left, right in zip(local, backend, strict=True)]
                parity = max(errors, default=0.0) <= 1e-6
            else:
                parity = False
            groups.append(
                {
                    **group,
                    "tracesearch_advantages": local,
                    "rllm_advantages": backend,
                    "backend_row_values": {uid: backend_by_uid.get(uid, []) for uid in group["rollout_ids"]},
                    "max_abs_error": max(errors, default=None),
                    "parity": parity,
                }
            )
        self.live_advantage = {"groups": groups, "parity": bool(groups) and all(item["parity"] for item in groups), "tolerance": 1e-6}
        _write_json(self.output_dir / "live_advantage_parity.json", self.live_advantage)
        active_advantages = [value for group in groups for value in group["rllm_advantages"] if value is not None]
        if self.phase == "c1":
            if not self.live_mask or not self.live_mask.get("parity") or not self.live_advantage["parity"]:
                self.update_gate = "parity_failed"
            elif active_advantages and all(abs(float(value)) <= 1e-12 for value in active_advantages):
                self.update_gate = "no_learning_signal"

    def record_batch(self, state: Any) -> None:
        metrics = dict(getattr(state, "metrics", {}) or {})
        timing = dict(getattr(state, "timing_dict", {}) or {})
        groups = _groups(state)
        probe_files = sorted(self.output_dir.glob("parameter_probe/actor_update_*.json"))
        self.parameter_probe_files.update(str(path) for path in probe_files)
        normalized = {
            "actor_loss": self._first_metric(metrics, ("actor/loss", "actor/loss/mean", "loss")),
            "grad_norm": self._first_metric(metrics, ("actor/grad_norm", "grad_norm")),
            "learning_rate": self._first_metric(metrics, ("actor/lr", "actor/learning_rate", "lr")),
            "clip_fraction": self._first_metric(metrics, ("actor/clipfrac", "actor/clip_fraction")),
            "entropy": self._first_metric(metrics, ("actor/entropy", "entropy")),
            "approx_kl": self._first_metric(metrics, ("actor/approx_kl", "approx_kl", "policy/approx_kl")),
            "response_length": self._first_metric(metrics, ("response_length/mean", "response_length")),
            "active_response_tokens": self._first_metric(metrics, ("response_length/mean", "response_length")),
            "rollout_throughput": self._first_metric(metrics, ("throughput/rollout", "perf/throughput", "throughput")),
        }
        record = {
            "global_step": int(getattr(state, "global_step", 0)),
            "groups": groups,
            "actor_update_called": self.actor_update_calls > 0,
            "observed_actor_update_calls": self.actor_update_calls,
            "normalized_metrics": normalized,
            "raw_backend_metrics": _jsonable(metrics),
            "timing_s": _jsonable(timing),
            "update_actor_s": timing.get("update_actor"),
            "update_weights_s": timing.get("update_weights"),
            "checkpoint_save_s": timing.get("save_checkpoint"),
            "live_mask_parity": self.live_mask,
            "live_advantage_parity": self.live_advantage,
            "update_gate": self.update_gate,
            "pre_update_snapshot": self.pre_update_snapshot,
        }
        self.records.append(record)
        _write_json(self.output_dir / "training_observer.json", {"phase": self.phase, "batches": self.records})
        self.actor_update_calls = 0

    @staticmethod
    def _first_metric(metrics: dict[str, Any], names: tuple[str, ...]) -> Any:
        for name in names:
            if name in metrics:
                return _jsonable(metrics[name])
        return None

    @staticmethod
    def _live_mask_parity(state: Any) -> dict[str, Any]:
        batch = getattr(state, "backend_batch", None)
        if batch is None or "responses" not in batch.batch or "response_mask" not in batch.batch:
            return {"parity": False, "reason": "backend_response_rows_unavailable", "rows": []}
        step_ids = _non_tensor_rows(batch, "step_ids")
        is_pad = _non_tensor_rows(batch, "is_pad_step")
        if not step_ids:
            return {"parity": False, "reason": "backend_step_ids_unavailable", "rows": []}

        local_by_uid: dict[str, list[dict[str, Any]]] = {}
        for group in getattr(state, "trajectory_groups", None) or []:
            for trajectory in getattr(group, "trajectories", []) or []:
                local_by_uid[str(getattr(trajectory, "uid", ""))] = _trajectory_segments(trajectory)

        rows: list[dict[str, Any]] = []
        segment_indexes: dict[str, int] = {}
        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"]
        attention_mask = batch.batch.get("attention_mask")
        for index, step_id in enumerate(step_ids):
            if index < len(is_pad) and bool(is_pad[index]):
                continue
            key = str(step_id)
            segments = local_by_uid.get(key, [])
            segment_index = segment_indexes.get(key, 0)
            segment_indexes[key] = segment_index + 1
            expected = segments[segment_index] if segment_index < len(segments) else None
            backend_response = _row(batch, "responses", index)
            backend_mask = _row(batch, "response_mask", index)
            if attention_mask is None:
                rows.append({"step_id": key, "backend_row_index": index, "parity": False, "reason": "attention_mask_unavailable"})
                continue
            attention = _row(batch, "attention_mask", index)
            response_width = len(backend_response)
            response_attention = attention[-response_width:] if response_width else []
            active_positions = [position for position, active in enumerate(response_attention) if bool(active)]
            backend_ids = [int(backend_response[position]) for position in active_positions]
            backend_mask_values = [int(backend_mask[position]) for position in active_positions]
            expected_ids = expected["response_ids"] if expected else []
            expected_mask = expected["response_mask"] if expected else []
            mismatch_positions = [
                position
                for position in range(max(len(expected_ids), len(backend_ids), len(expected_mask), len(backend_mask_values)))
                if position >= len(expected_ids)
                or position >= len(backend_ids)
                or expected_ids[position] != backend_ids[position]
                or position >= len(expected_mask)
                or position >= len(backend_mask_values)
                or expected_mask[position] != backend_mask_values[position]
            ]
            rows.append(
                {
                    "step_id": key,
                    "backend_row_index": index,
                    "segment_index": segment_index,
                    "step_boundaries": expected["step_boundaries"] if expected else [],
                    "expected_response_ids": expected_ids,
                    "backend_unpadded_response_ids": backend_ids,
                    "expected_response_mask": expected_mask,
                    "backend_response_mask": backend_mask_values,
                    "mismatch_positions": mismatch_positions,
                    "padding_tokens_excluded": len(backend_response) - len(active_positions),
                    "parity": expected is not None and not mismatch_positions,
                }
            )
        return {
            "parity": bool(rows) and all(row["parity"] for row in rows),
            "rows": rows,
            "semantic_contract": "assistant action tokens=1; tool/environment observation tokens=0; prompt and padding=0",
        }


def install_training_observer(backend_module: Any, observer: RuntimeTrainingObserver) -> str:
    """Attach the observer to the pinned backend class in the driver process."""

    backend_type = getattr(backend_module, "VerlBackend", None)
    if backend_type is None:
        raise RuntimeError("pinned rLLM backend does not expose VerlBackend")
    if getattr(backend_type, "_tracesearch_training_observer", False):
        return "already_installed"
    original_init = backend_type.__init__
    original_process = backend_type.process_backend_batch
    original_advantages = backend_type.compute_advantages
    original_update = backend_type.update_policy
    original_update_actor = backend_type._update_actor_with_loss_routing
    original_batch_end = backend_type.on_batch_end

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self._tracesearch_training_observer = observer
        self._tracesearch_actor_updates_this_batch = 0

    async def process(self: Any, trainer_state: Any, *args: Any, **kwargs: Any) -> Any:
        result = await original_process(self, trainer_state, *args, **kwargs)
        observer.observe_process_batch(trainer_state)
        return result

    async def advantages(self: Any, trainer_state: Any, *args: Any, **kwargs: Any) -> Any:
        result = await original_advantages(self, trainer_state, *args, **kwargs)
        observer.observe_advantages(trainer_state)
        return result

    def update_actor(self: Any, *args: Any, **kwargs: Any) -> Any:
        self._tracesearch_actor_updates_this_batch = getattr(self, "_tracesearch_actor_updates_this_batch", 0) + 1
        observer.note_actor_update()
        return original_update_actor(self, *args, **kwargs)

    async def update(self: Any, trainer_state: Any, *args: Any, **kwargs: Any) -> Any:
        self._tracesearch_actor_updates_this_batch = 0
        if observer.should_skip_actor_update():
            return None
        observer.capture_pre_update(self, trainer_state)
        if observer.should_skip_actor_update():
            return None
        return await original_update(self, trainer_state, *args, **kwargs)

    async def batch_end(self: Any, trainer_state: Any, *args: Any, **kwargs: Any) -> Any:
        if observer.should_skip_actor_update():
            # In proof mode a zero-signal or unverifiable batch must not save a
            # fake step or pay the expensive colocated weight synchronization.
            observer.record_batch(trainer_state)
            return None
        try:
            return await original_batch_end(self, trainer_state, *args, **kwargs)
        finally:
            observer.record_batch(trainer_state)

    backend_type.__init__ = init
    backend_type.process_backend_batch = process
    backend_type.compute_advantages = advantages
    backend_type._update_actor_with_loss_routing = update_actor
    backend_type.update_policy = update
    backend_type.on_batch_end = batch_end
    backend_type._tracesearch_training_observer = True
    return "installed"


def install_actor_update_probe(worker_module: Any) -> str:
    """Probe the engine's unregistered train call inside the worker process.

    Patching ``ActorRolloutRefWorker.update_actor`` would remove veRL's
    dispatch registration in this pinned release.  The actor method delegates
    to ``BaseEngine.train_batch``; that method is not a Ray-registered surface,
    so it is safe to wrap for tensor evidence.
    """

    del worker_module
    from importlib import import_module

    engine_module = import_module("verl.workers.engine.base")
    engine_type = getattr(engine_module, "BaseEngine", None)
    if engine_type is None:
        return "unavailable"
    original = getattr(engine_type, "train_batch", None)
    if original is None or getattr(original, "_tracesearch_parameter_probe", False):
        return "already_installed"

    def snapshot(engine: Any, *, keep_values: bool = False) -> dict[str, Any]:
        module = getattr(engine, "module", None)
        result: dict[str, Any] = {"trainable_tensors": {}, "total_model_parameters": None}
        if keep_values:
            result["_values"] = {}
        if module is None or not hasattr(module, "named_parameters"):
            return result
        digest = hashlib.sha256()
        trainable_count = 0
        total_count = 0
        for name, parameter in module.named_parameters():
            try:
                tensor = parameter.detach()
                if hasattr(tensor, "full_tensor"):
                    tensor = tensor.full_tensor()
                tensor_cpu = tensor.float().cpu().contiguous()
                total_count += tensor_cpu.numel()
                if not parameter.requires_grad and "lora" not in name.lower():
                    continue
                trainable_count += tensor_cpu.numel()
                digest.update(name.encode("utf-8"))
                digest.update(tensor_cpu.numpy().tobytes())
                result["trainable_tensors"][name] = {"shape": list(tensor_cpu.shape), "numel": tensor_cpu.numel(), "l2_norm": float(tensor_cpu.norm().item()), "digest": hashlib.sha256(tensor_cpu.numpy().tobytes()).hexdigest()}
                if keep_values:
                    result["_values"][name] = tensor_cpu
            except Exception as exc:
                result["trainable_tensors"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        result["total_model_parameters"] = total_count
        result["total_trainable_parameters"] = trainable_count
        result["trainable_parameter_digest"] = digest.hexdigest()
        return result

    def wrapped(self: Any, data: Any) -> Any:
        output_dir = os.environ.get("TRACESEARCH_M1C_OBSERVER_DIR")
        before = snapshot(self, keep_values=True)
        try:
            result = original(self, data)
        except Exception as exc:
            if output_dir:
                public_before = {key: value for key, value in before.items() if key != "_values"}
                _write_json(Path(output_dir) / "parameter_probe" / f"actor_update_{os.getpid()}_{time.time_ns()}.json", {"update_called": True, "update_succeeded": False, "error": f"{type(exc).__name__}: {exc}", "before": public_before})
            raise
        if output_dir:
            after = snapshot(self, keep_values=True)
            before_values = before.get("_values", {})
            after_values = after.get("_values", {})
            changed = []
            max_abs = 0.0
            l2_sq = 0.0
            for name in sorted(set(before_values) | set(after_values)):
                left, right = before_values.get(name), after_values.get(name)
                if left is None or right is None or left.shape != right.shape:
                    continue
                delta = (right - left).float()
                current_max = float(delta.abs().max().item()) if delta.numel() else 0.0
                current_l2 = float(delta.norm().item())
                max_abs = max(max_abs, current_max)
                l2_sq += current_l2 * current_l2
                if current_max != 0.0:
                    changed.append(name)
            public_before = {key: value for key, value in before.items() if key != "_values"}
            public_after = {key: value for key, value in after.items() if key != "_values"}
            _write_json(
                Path(output_dir) / "parameter_probe" / f"actor_update_{os.getpid()}_{time.time_ns()}.json",
                {
                    "update_called": True,
                    "update_succeeded": True,
                    "before": public_before,
                    "after": public_after,
                    "changed_trainable_tensors": changed,
                    "changed_trainable_tensor_count": len(changed),
                    "max_abs_parameter_delta": max_abs,
                    "total_l2_parameter_delta": l2_sq**0.5,
                },
            )
        return result

    wrapped._tracesearch_parameter_probe = True  # type: ignore[attr-defined]
    engine_type.train_batch = wrapped
    return "installed"
