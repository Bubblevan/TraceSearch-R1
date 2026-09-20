"""Run the real rLLM/veRL M1-C backend smoke phases.

This module is intentionally a Linux/WSL-only entry point.  The native
Windows environment can still import the project and run the M1-B tests, but
the optional rLLM/veRL/vLLM stack must remain in the isolated uv environment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as package_metadata
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from tracesearch.data.adapters import NQTaskAdapter
from tracesearch.training.grpo import compute_group_advantages
from tracesearch.training.trace import GenerationRecord, TrainingTrace


RLLM_COMMIT = "3b40c37cf6a262cf4d28cc987ebe4f4cf797956c"
VERL_COMMIT = "7aed6b230776f963fa09509c10d9c3a767d1102c"
VLLM_COMMIT = "0decac0d96c42b49572498019f0a0e3600f50398"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("c0", "c1", "c2"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="data/m1/dev.jsonl")
    parser.add_argument("--corpus", default="data/m0/corpus.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--task-count", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git(repo: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _source_tree_hash(repo: Path) -> str | None:
    paths = _git(repo, "ls-files", "--cached", "--others", "--exclude-standard")
    if paths is None:
        return None
    digest = hashlib.sha256()
    for relative in sorted(item for item in paths.splitlines() if item):
        path = repo / relative
        if not path.is_file():
            continue
        digest.update(relative.replace("\\", "/").encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def _backend_versions() -> dict[str, Any]:
    packages = ("rllm", "verl", "vllm", "torch", "transformers", "ray", "numpy", "flash-attn")
    result: dict[str, Any] = {}
    for name in packages:
        try:
            result[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            result[name] = None
    try:
        import torch

        result["torch_cuda"] = torch.version.cuda
        result["cuda_available"] = bool(torch.cuda.is_available())
        result["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:  # pragma: no cover - backend environment dependent.
        result["torch_runtime_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _load_tasks(path: Path, task_count: int) -> tuple[list[Any], dict[str, Any]]:
    dataset = NQTaskAdapter(
        upstream_dataset="natural_questions_style_fixture",
        upstream_revision="local-m1-fixture",
    ).load_jsonl(path, split="train")
    if task_count > 0:
        tasks = list(dataset.tasks[:task_count])
    else:
        tasks = list(dataset.tasks)
    if not tasks:
        raise ValueError(f"no tasks loaded from {path}")
    return tasks, dataset.manifest_fields()


def _compose_config(args: argparse.Namespace, output: Path) -> Any:
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    with initialize_config_module(version_base=None, config_module="rllm.trainer.config"):
        config = compose(config_name="unified")

    def set_value(path: str, value: Any, *, force_add: bool = False) -> None:
        OmegaConf.update(config, path, value, merge=False, force_add=force_add)

    steps = {"c0": 1, "c1": 1, "c2": 10}[args.phase]
    max_response_length = max(args.max_tokens * 8, 512)

    # rLLM is the source of truth; its Verl sync layer mirrors the shared
    # values into the native ``data``, ``trainer`` and actor/rollout paths.
    set_value("rllm.data.train_batch_size", 1)
    set_value("rllm.data.val_batch_size", -1)
    set_value("rllm.data.max_prompt_length", 512)
    set_value("rllm.data.max_response_length", max_response_length)
    set_value("rllm.data.seed", args.seed)
    set_value("rllm.rollout.n", args.group_size)
    set_value("rllm.rollout.n_val", 1)
    set_value("rllm.rollout.train.temperature", 1.0)
    set_value("rllm.rollout.train.top_p", 1.0)
    set_value("rllm.rollout.train.top_k", -1, force_add=True)
    set_value("rllm.rollout.train.max_tokens", args.max_tokens)
    set_value("rllm.rollout.val.temperature", 1.0)
    set_value("rllm.rollout.val.top_p", 1.0)
    set_value("rllm.rollout.val.top_k", -1, force_add=True)
    set_value("rllm.rollout.val.max_tokens", args.max_tokens)
    set_value("rllm.algorithm.adv_estimator", "grpo")
    set_value("rllm.algorithm.norm_adv_by_std_in_grpo", True)
    set_value("rllm.algorithm.loss_agg_mode", "seq-mean-token-mean")
    set_value("rllm.algorithm.kl_beta", 0.0)
    set_value("rllm.algorithm.eps_clip", 0.2)
    set_value("rllm.algorithm.rollout_correction.bypass_mode", False)
    set_value("rllm.stepwise_advantage.enable", False)
    set_value("rllm.stepwise_advantage.mode", "broadcast")
    set_value("rllm.rejection_sample.enable", False)
    set_value("rllm.rejection_sample.multiplier", 1)
    set_value("rllm.compact_filtering.enable", False)
    set_value("rllm.disable_thinking", True)
    set_value("rllm.accumulate_reasoning", False)
    set_value("rllm.trainer.total_epochs", 1)
    set_value("rllm.trainer.total_batches", steps)
    set_value("rllm.trainer.logger", ["console"])
    set_value("rllm.trainer.project_name", "tracesearch-m1c")
    set_value("rllm.trainer.experiment_name", args.phase)
    set_value("rllm.trainer.save_freq", 1)
    set_value("rllm.trainer.test_freq", -1)
    set_value("rllm.trainer.val_before_train", False)
    set_value("rllm.trainer.val_only", False)
    set_value("rllm.workflow.n_parallel_tasks", args.group_size)
    set_value("rllm.workflow.retry_limit", 1)
    set_value("rllm.workflow.raise_on_error", True)

    set_value("trainer.nnodes", 1)
    set_value("trainer.n_gpus_per_node", 1)
    set_value("trainer.device", "cuda")
    set_value("trainer.logger", ["console"])
    set_value("trainer.project_name", "tracesearch-m1c")
    set_value("trainer.experiment_name", args.phase)
    set_value("trainer.val_before_train", False)
    set_value("trainer.val_only", False)
    set_value("trainer.test_freq", -1)
    set_value("trainer.save_freq", 1)
    set_value("trainer.resume_mode", "disable")
    set_value("trainer.default_local_dir", str(output / "checkpoints"))
    set_value("trainer.critic_warmup", 999999 if args.phase == "c0" else 0)
    set_value("rollout.nnodes", 1)
    set_value("rollout.n_gpus_per_node", 1)

    set_value("data.train_batch_size", 1)
    set_value("data.max_prompt_length", 512)
    set_value("data.max_response_length", max_response_length)
    set_value("data.trust_remote_code", False)
    set_value("actor_rollout_ref.rollout.name", "vllm")
    set_value("actor_rollout_ref.rollout.mode", "async")
    set_value("actor_rollout_ref.rollout.n", args.group_size)
    set_value("actor_rollout_ref.rollout.temperature", 1.0)
    set_value("actor_rollout_ref.rollout.top_p", 1.0)
    set_value("actor_rollout_ref.rollout.top_k", -1)
    set_value("actor_rollout_ref.rollout.do_sample", True)
    set_value("actor_rollout_ref.rollout.response_length", max_response_length)
    set_value("actor_rollout_ref.rollout.prompt_length", 512)
    set_value("actor_rollout_ref.rollout.calculate_log_probs", True)
    set_value("actor_rollout_ref.rollout.n_gpus_per_node", 1)
    set_value("actor_rollout_ref.rollout.nnodes", 1)
    set_value("actor_rollout_ref.rollout.gpu_memory_utilization", args.gpu_memory_utilization)
    set_value("actor_rollout_ref.rollout.max_num_batched_tokens", 4096)
    set_value("actor_rollout_ref.rollout.max_num_seqs", args.group_size)
    # The pinned veRL config defaults this legacy rollout field to two-way
    # tensor parallelism.  M1-C's WSL smoke/training contract is one GPU, so
    # make the native rollout setting explicit as well as the rLLM setting.
    set_value("actor_rollout_ref.rollout.tensor_model_parallel_size", 1)
    # The single-GPU WSL colocated run keeps the FSDP actor resident while
    # vLLM reserves KV-cache blocks.  1024 covers the M1-C prompt/response
    # contract and leaves room for those blocks on a 16 GiB card.
    set_value("actor_rollout_ref.rollout.max_model_len", 1024)
    set_value("actor_rollout_ref.rollout.enable_prefix_caching", False)
    set_value("actor_rollout_ref.rollout.enforce_eager", True)
    set_value("actor_rollout_ref.rollout.load_format", "auto")
    set_value("actor_rollout_ref.rollout.free_cache_engine", True)
    # veRL's colocated actor keeps FSDP resident while vLLM starts.  On the
    # 16 GiB WSL GPU, CPU offload is the supported vLLM escape hatch that
    # leaves enough device memory for at least one KV-cache block.
    set_value(
        "actor_rollout_ref.rollout.engine_kwargs.vllm.cpu_offload_gb",
        4.0,
        force_add=True,
    )
    set_value("actor_rollout_ref.rollout.val_kwargs.do_sample", True)
    set_value("actor_rollout_ref.rollout.val_kwargs.temperature", 1.0)
    set_value("actor_rollout_ref.rollout.val_kwargs.top_p", 1.0)
    set_value("actor_rollout_ref.rollout.val_kwargs.top_k", -1)
    set_value("actor_rollout_ref.rollout.agent.num_workers", 0)
    set_value("actor_rollout_ref.rollout.multi_turn.enable", False)
    set_value("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", 1)
    set_value("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", 1)

    set_value("actor_rollout_ref.model.path", str(args.model))
    set_value("actor_rollout_ref.model.use_shm", False)
    # The isolated WSL env intentionally does not require flash-attn.  veRL's
    # HFModelConfig defaults to FlashAttention2, so select the portable SDPA
    # path explicitly for this first backend integration.
    set_value("actor_rollout_ref.model.override_config.attn_implementation", "sdpa", force_add=True)
    set_value("actor_rollout_ref.model.enable_gradient_checkpointing", True)
    set_value("actor_rollout_ref.model.use_remove_padding", True)
    set_value("actor_rollout_ref.model.lora_rank", 8)
    set_value("actor_rollout_ref.model.lora_alpha", 16)
    set_value("actor_rollout_ref.model.lora.rank", 8)
    set_value("actor_rollout_ref.model.lora.alpha", 16)
    # Keep target names explicit for the train-side PEFT model.  The pinned
    # veRL/vLLM pair currently has an incompatible dynamic-TensorLoRA path for
    # Qwen2, so M1-C uses veRL's merged-LoRA weight-sync path.  This keeps the
    # trainable adapter while sending ordinary base-model tensors to vLLM.
    set_value(
        "actor_rollout_ref.model.lora.target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    set_value("actor_rollout_ref.model.lora.merge", True)

    set_value("actor_rollout_ref.actor.ppo_mini_batch_size", 1)
    set_value("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", 1)
    set_value("actor_rollout_ref.actor.ppo_max_token_len_per_gpu", 2048)
    # A single 16 GiB GPU colocates FSDP and vLLM.  veRL's FSDP default is
    # fp32, which leaves no room for the rollout engine after loading a 3B
    # model; bfloat16 is the intended mixed-precision training path here.
    set_value("actor_rollout_ref.actor.fsdp_config.model_dtype", "bfloat16")
    set_value("actor_rollout_ref.actor.loss_agg_mode", "seq-mean-token-mean")
    set_value("actor_rollout_ref.actor.ppo_epochs", 1)
    set_value("actor_rollout_ref.actor.use_dynamic_bsz", False)
    set_value("actor_rollout_ref.actor.shuffle", False)
    set_value("actor_rollout_ref.actor.use_torch_compile", False)
    set_value("actor_rollout_ref.actor.entropy_coeff", 0.0)
    set_value("actor_rollout_ref.actor.use_kl_loss", False)
    set_value("actor_rollout_ref.actor.kl_loss_coef", 0.0)
    set_value("actor_rollout_ref.actor.use_rollout_log_probs", True)
    set_value("actor_rollout_ref.actor.optim.lr", 1e-6)
    set_value("actor_rollout_ref.actor.optim.weight_decay", 0.0)

    return config


def _register_dataset(tasks: list[Any], phase: str) -> Any:
    from rllm.data import DatasetRegistry

    name = f"tracesearch_m1c_{phase}_{os.getpid()}"
    rows = [task.to_dict() for task in tasks]
    return DatasetRegistry.register_dataset(
        name,
        rows,
        split="train",
        source="TraceSearch local M1 fixture",
        description="TraceSearch M1-C train-side fixture; gold fields stay in evaluator metadata.",
        category="search-agent",
    )


def _mask_parity() -> dict[str, Any]:
    """Compare TraceSearch's exact turn mask with rLLM's merge transform."""

    from rllm.engine.rollout import ModelOutput
    from rllm.trainer.verl.dataclass import AccumulatedData
    from rllm.trainer.verl.transform import _process_trajectory
    from rllm.types import Step as RLLMStep
    from rllm.types import Trajectory as RLLMTrajectory

    first = ModelOutput(prompt_ids=[10, 11], completion_ids=[20, 21], logprobs=[-0.1, -0.2], content="a")
    second = ModelOutput(
        prompt_ids=[10, 11, 20, 21, 30, 31],
        completion_ids=[40, 41],
        logprobs=[-0.3, -0.4],
        content="b",
    )
    trajectory = RLLMTrajectory(
        uid="mask-parity",
        name="search",
        steps=[
            RLLMStep(model_output=first, prompt_ids=[10, 11], response_ids=[20, 21], logprobs=[-0.1, -0.2]),
            RLLMStep(model_output=second, prompt_ids=[10, 11, 20, 21, 30, 31], response_ids=[40, 41], logprobs=[-0.3, -0.4]),
        ],
        reward=1.0,
    )
    accumulated = AccumulatedData()
    _process_trajectory(trajectory, "mask-task", accumulated)
    backend_mask = accumulated.traj_mask[0].tolist()
    records = (
        GenerationRecord(0, (10, 11), (20, 21), (-0.1, -0.2)),
        GenerationRecord(1, (10, 11, 20, 21, 30, 31), (40, 41), (-0.3, -0.4)),
    )
    local_trace = TrainingTrace.from_generation_records(
        task_id="mask-task",
        rollout_id="mask-parity",
        sample_index=0,
        policy_version="0",
        generation_records=records,
        tool_step_indices=(0,),
        observation_token_ids={0: (30, 31)},
    )
    local_mask = list(local_trace.response_mask)
    return {
        "expected_mask": [1, 1, 0, 0, 1, 1],
        "backend_mask": backend_mask,
        "local_mask": local_mask,
        "backend_matches_expected": backend_mask == [1, 1, 0, 0, 1, 1],
        "local_matches_expected": local_mask == [1, 1, 0, 0, 1, 1],
        "parity": backend_mask == local_mask == [1, 1, 0, 0, 1, 1],
        "semantics": "mask=1 policy action tokens; mask=0 interleaved observation tokens",
    }


def _advantage_parity() -> dict[str, Any]:
    import numpy as np
    from rllm.trainer.algorithms.rl_algo import calculate_grpo_advantages_per_group

    cases = {
        "binary_0101": [0.0, 1.0, 0.0, 1.0],
        "all_zero": [0.0, 0.0, 0.0, 0.0],
        "all_one": [1.0, 1.0, 1.0, 1.0],
    }
    output: dict[str, Any] = {"estimator": "grpo", "normalization": "population_std", "cases": {}}
    for name, rewards in cases.items():
        local = list(compute_group_advantages(rewards, epsilon=1e-6))
        upstream, _ = calculate_grpo_advantages_per_group(np.asarray(rewards, dtype=np.float64), True, episilon=1e-6)
        upstream_values = [float(value) for value in upstream]
        output["cases"][name] = {
            "rewards": rewards,
            "local": local,
            "rllm": upstream_values,
            "max_abs_error": max((abs(left - right) for left, right in zip(local, upstream_values, strict=True)), default=0.0),
            "parity": all(abs(left - right) <= 1e-6 for left, right in zip(local, upstream_values, strict=True)),
        }
    output["parity"] = all(case["parity"] for case in output["cases"].values())
    return output


def _summarize_rollouts(output: Path, tasks: list[Any], group_size: int, phase: str) -> dict[str, Any]:
    source = output / "tracesearch_rollouts.jsonl"
    rows: list[dict[str, Any]] = []
    if source.exists():
        with source.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]

    # C2 contains repeated updates.  The last occurrence is the final
    # on-policy group for each task/sample slot; older occurrences remain in
    # the raw JSONL for auditability.
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    occurrence_counts: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (str(row["task_id"]), int(row["sample_index"]))
        latest[key] = row
        occurrence_counts[key] = occurrence_counts.get(key, 0) + 1

    groups: list[dict[str, Any]] = []
    for task in tasks:
        slots: list[dict[str, Any] | None] = [None] * group_size
        for sample_index in range(group_size):
            row = latest.get((task.task_id, sample_index))
            if row is not None:
                slots[sample_index] = {
                    "sample_index": sample_index,
                    "rollout_id": row["rollout_id"],
                    "reward": float(row["reward"]["total"]),
                    "status": row["reward"]["status"],
                    "occurrences": occurrence_counts[(task.task_id, sample_index)],
                }
        rewards = [float(slot["reward"]) if slot is not None else 0.0 for slot in slots]
        group_mean = statistics.mean(rewards) if rewards else 0.0
        group_variance = statistics.pvariance(rewards) if rewards else 0.0
        groups.append(
            {
                "task_id": task.task_id,
                "expected_rollouts": group_size,
                "slots": slots,
                "missing_slots": [index for index, slot in enumerate(slots) if slot is None],
                "pass@1": bool(slots[0] is not None and slots[0]["reward"] > 0.0),
                "pass@k": {
                    str(k): any(slot is not None and slot["reward"] > 0.0 for slot in slots[:k])
                    for k in range(1, group_size + 1)
                },
                "mean_exact_match": group_mean,
                "group_exact_match_variance": group_variance,
                "reward_definition": "normalized_exact_match",
            }
        )

    _write_jsonl(output / "rollout_groups.jsonl", groups)
    reward_values = [float(slot["reward"]) for group in groups for slot in group["slots"] if slot is not None]
    summary = {
        "phase": phase,
        "expected_task_count": len(tasks),
        "expected_rollouts_per_task": group_size,
        "raw_rollout_rows": len(rows),
        "unique_rollout_slots": len(latest),
        "missing_rollout_slots": sum(len(group["missing_slots"]) for group in groups),
        "completed_rollout_slots": sum(slot is not None for group in groups for slot in group["slots"]),
        "mean_exact_match": statistics.mean(reward_values) if reward_values else 0.0,
        "group_exact_match_variance": statistics.mean(group["group_exact_match_variance"] for group in groups) if groups else 0.0,
        "denominator": len(tasks) * group_size,
        "note": "group_reward_variance is intentionally not emitted here; evaluator variance is binary normalized-EM and trainer reward variance belongs to the future M1 training logger.",
    }
    _write_json(output / "backend_batch_summary.json", summary)
    return summary


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_artifacts(output: Path, args: argparse.Namespace, config: Any, tasks: list[Any], provenance: dict[str, Any]) -> None:
    from omegaconf import OmegaConf

    repo = Path(__file__).resolve().parents[3]
    versions = _backend_versions()
    manifest = {
        "schema_version": "m1c.backend.v1",
        "phase": args.phase,
        "git_commit": _git(repo, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(repo, "status", "--porcelain", "--untracked-files=all")),
        "source_tree_hash": _source_tree_hash(repo),
        "backend_source_commits": {
            "rllm": RLLM_COMMIT,
            "verl": VERL_COMMIT,
            "vllm": VLLM_COMMIT,
        },
        "backend_versions": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "uv_environment": os.environ.get("VIRTUAL_ENV"),
        "sampling_contract": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "disable_thinking": True,
            "rollout_seed": "base_seed + sample_index",
        },
        "algorithm_contract": {
            "estimator": "grpo",
            "group_std": "population",
            "stepwise_advantage": False,
            "process_reward": False,
            "kl_beta": 0.0,
            "rejection_sampling": False,
            "loss_agg_mode": "seq-mean-token-mean",
            "trainable_method": "LoRA",
            "lora_rank": 8,
            "lora_sync": "merged_base_weights",
        },
        "dataset": provenance,
        "model": str(args.model),
        "seed": args.seed,
        "group_size": args.group_size,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "dependency_resolution_note": "rLLM current metadata conflicts with veRL/vLLM numpy constraints; this isolated env uses the explicitly recorded numpy override.",
    }
    _write_json(output / "backend_manifest.json", manifest)
    _write_json(output / "resolved_config.json", OmegaConf.to_container(config, resolve=False))
    _write_json(output / "mask_parity.json", _mask_parity())
    _write_json(output / "advantage_parity.json", _advantage_parity())
    summary = _summarize_rollouts(output, tasks, args.group_size, args.phase)
    metrics = {
        "phase": args.phase,
        "requested_optimizer_steps": {"c0": 0, "c1": 1, "c2": 10}[args.phase],
        "optimizer_steps": {"c0": 0, "c1": 1, "c2": 10}[args.phase],
        "optimizer_steps_source": "configured trainer.total_batches; c0 critic_warmup suppresses actor update",
        "mean_exact_match": summary["mean_exact_match"],
        "group_exact_match_variance": summary["group_exact_match_variance"],
        "zero_variance_groups": summary["group_exact_match_variance"] == 0.0,
        "loss_agg_mode": "seq-mean-token-mean",
        "status": "completed",
    }
    _write_json(output / "metrics.json", metrics)
    (output / "summary.md").write_text(
        "\n".join(
            [
                f"# TraceSearch M1-C {args.phase}",
                "",
                f"- Backend commit: `{RLLM_COMMIT}` / veRL `{VERL_COMMIT}` / vLLM `{VLLM_COMMIT}`",
                f"- Project commit: `{manifest['git_commit']}`",
                f"- Git dirty: `{manifest['git_dirty']}`",
                f"- Optimizer steps: `{metrics['optimizer_steps']}`",
                f"- Rollout slots: `{summary['completed_rollout_slots']}/{summary['denominator']}`",
                f"- Mean exact match: `{summary['mean_exact_match']}`",
                f"- Group exact-match variance: `{summary['group_exact_match_variance']}`",
                "- Reward: normalized exact match only; no process reward, KL, rejection sampling, or fatal-aware shaping.",
                "- `group_reward_variance` is intentionally not used: evaluator variance is named `group_exact_match_variance`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> int:
    # Ask vLLM V1 not to add its own client-side multiprocessing layer.  veRL
    # 0.8's async server still passes distributed_executor_backend=mp, so this
    # does not guarantee an in-process EngineCore; the resolved config and
    # console log remain the source of truth for the actual runtime topology.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks, provenance = _load_tasks(Path(args.dataset), args.task_count)
    config = _compose_config(args, output)

    from rllm.trainer import AgentTrainer
    from tracesearch.training.rllm_workflow import TraceSearchWorkflow

    train_dataset = _register_dataset(tasks, args.phase)
    trainer = AgentTrainer(
        config=config,
        workflow_class=TraceSearchWorkflow,
        train_dataset=train_dataset,
        val_dataset=None,
        backend="verl",
        workflow_args={
            "corpus_path": str(Path(args.corpus).resolve()),
            "checkpoint": str(Path(args.model).resolve()),
            "top_k": args.top_k,
            "max_turns": args.max_turns,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k_sampling": None,
            "artifact_dir": str(output),
            "disable_thinking": True,
        },
    )
    trainer.train()
    _write_artifacts(output, args, config, tasks, provenance)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
