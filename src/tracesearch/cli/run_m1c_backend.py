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
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from tracesearch.data.adapters import NQTaskAdapter
from tracesearch.training.grpo import compute_group_advantages
from tracesearch.training.m1c_observer import RuntimeTrainingObserver
from tracesearch.experiment.provenance import flash_attn_provenance
from tracesearch.training.trace import GenerationRecord, TrainingTrace


RLLM_COMMIT = "3b40c37cf6a262cf4d28cc987ebe4f4cf797956c"
VERL_COMMIT = "7aed6b230776f963fa09509c10d9c3a767d1102c"
VLLM_COMMIT = "0decac0d96c42b49572498019f0a0e3600f50398"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("c0", "c1", "c2"), required=True)
    parser.add_argument(
        "--config",
        default="configs/m1/runtime/4090-16g-wsl.yaml",
        help="TraceSearch-owned runtime profile; explicit CLI values override it.",
    )
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
    parser.add_argument("--task-id", default=None, help="Select exactly one declared task by task_id without reordering the dataset.")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Clear known M1-C artifacts in an existing output directory before starting.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--cpu-offload-gb", type=float, default=None)
    parser.add_argument(
        "--rollout-engine",
        choices=("vllm", "sglang"),
        default="vllm",
        help="Select the veRL rollout engine; SGLang requires the separate WSL uv environment.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use eager execution in vLLM (use --no-enforce-eager to benchmark CUDA graphs).",
    )
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_run_status(output: Path, status: str, **extra: Any) -> None:
    _write_json(output / "run_status.json", {"status": status, "updated_at": time.time(), **extra})


_KNOWN_RUN_ARTIFACTS = (
    "backend_manifest.json",
    "resolved_config.json",
    "run_status.json",
    "failure.json",
    "tracesearch_rollouts.jsonl",
    "rollout_groups.jsonl",
    "backend_batch_summary.json",
    "metrics.json",
    "training_observer.json",
    "live_mask_parity.json",
    "live_advantage_parity.json",
    "mask_parity.json",
    "advantage_parity.json",
    "flash_attn_provenance.json",
    "rollout_logprob_diagnostics.json",
    "parameter_delta.json",
    "checkpoint_proof.json",
    "checkpoint_reload.json",
    "checkpoint_reload_process.log",
    "summary.md",
    "pre_update",
    "checkpoints",
    "parameter_probe",
    "pre_update_error.json",
    "console.log",
)


def _prepare_output(output: Path, *, overwrite: bool) -> None:
    """Refuse stale M1-C evidence unless the caller explicitly opts in."""

    output.mkdir(parents=True, exist_ok=True)
    existing = [output / name for name in _KNOWN_RUN_ARTIFACTS if (output / name).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"output directory contains prior M1-C artifacts: {names}; use --overwrite-output")
    if overwrite:
        for path in existing:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


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
    result["flash_attn_provenance"] = flash_attn_provenance()
    return result


def _load_tasks(path: Path, task_count: int, task_id: str | None = None) -> tuple[list[Any], dict[str, Any]]:
    dataset = NQTaskAdapter(
        upstream_dataset="natural_questions_style_fixture",
        upstream_revision="local-m1-fixture",
    ).load_jsonl(path, split="train")
    if task_id is not None:
        matches = [task for task in dataset.tasks if task.task_id == task_id]
        if not matches:
            raise ValueError(f"task_id {task_id!r} not found in {path}")
        tasks = matches
    elif task_count > 0:
        tasks = list(dataset.tasks[:task_count])
    else:
        tasks = list(dataset.tasks)
    if not tasks:
        raise ValueError(f"no tasks loaded from {path}")
    return tasks, dataset.manifest_fields()


def _apply_runtime_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Load the owned profile and let explicit CLI values take precedence."""

    config_path = Path(getattr(args, "config", "configs/m1/runtime/4090-16g-wsl.yaml"))
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    profile: dict[str, Any] = {}
    if config_path.is_file():
        try:
            from omegaconf import OmegaConf

            profile = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)  # type: ignore[assignment]
        except ModuleNotFoundError:
            # The CPU/dev installation intentionally does not pull the Linux
            # Hydra stack.  Keep profile loading testable without making the
            # backend dependency mandatory for ordinary project commands.
            current: list[tuple[int, dict[str, Any]]] = [(-1, profile)]
            for raw_line in config_path.read_text(encoding="utf-8").splitlines():
                if not raw_line.strip() or raw_line.lstrip().startswith("#") or ":" not in raw_line:
                    continue
                indent = len(raw_line) - len(raw_line.lstrip())
                key, raw_value = raw_line.strip().split(":", 1)
                while current[-1][0] >= indent:
                    current.pop()
                parent = current[-1][1]
                value = raw_value.strip()
                if not value:
                    parent[key] = {}
                    current.append((indent, parent[key]))
                elif value.lower() in {"true", "false"}:
                    parent[key] = value.lower() == "true"
                else:
                    try:
                        parent[key] = json.loads(value)
                    except json.JSONDecodeError:
                        try:
                            parent[key] = float(value) if "." in value or "e" in value.lower() else int(value)
                        except ValueError:
                            parent[key] = value.strip('"\'')
    runtime = profile.get("runtime", {})

    def resolve(name: str, profile_name: str, default: Any) -> Any:
        value = getattr(args, name, None)
        if value is not None:
            return value
        return runtime.get(profile_name, default)

    args.gpu_memory_utilization = float(resolve("gpu_memory_utilization", "gpu_memory_utilization", 0.35))
    args.cpu_offload_gb = float(resolve("cpu_offload_gb", "cpu_offload_gb", 4.0))
    args.enforce_eager = bool(resolve("enforce_eager", "enforce_eager", True))
    args.tensor_model_parallel_size = int(resolve("tensor_model_parallel_size", "tensor_model_parallel_size", 1))
    args.max_model_len = int(resolve("max_model_len", "max_model_len", 1024))
    args.max_num_batched_tokens = int(resolve("max_num_batched_tokens", "max_num_batched_tokens", 4096))
    args._runtime_profile_path = str(config_path)
    return profile


def _compose_config(args: argparse.Namespace, output: Path) -> Any:
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    profile = _apply_runtime_profile(args)
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
    set_value("actor_rollout_ref.rollout.name", args.rollout_engine)
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
    set_value("actor_rollout_ref.rollout.max_num_batched_tokens", args.max_num_batched_tokens)
    set_value("actor_rollout_ref.rollout.max_num_seqs", args.group_size)
    # The pinned veRL config defaults this legacy rollout field to two-way
    # tensor parallelism.  M1-C's WSL smoke/training contract is one GPU, so
    # make the native rollout setting explicit as well as the rLLM setting.
    set_value("actor_rollout_ref.rollout.tensor_model_parallel_size", args.tensor_model_parallel_size)
    # The single-GPU WSL colocated run keeps the FSDP actor resident while
    # vLLM reserves KV-cache blocks.  1024 covers the M1-C prompt/response
    # contract and leaves room for those blocks on a 16 GiB card.
    set_value("actor_rollout_ref.rollout.max_model_len", args.max_model_len)
    set_value("actor_rollout_ref.rollout.enable_prefix_caching", False)
    set_value("actor_rollout_ref.rollout.enforce_eager", args.enforce_eager)
    set_value("actor_rollout_ref.rollout.load_format", "auto")
    # SGLang's veRL HYBRID adapter does not implement the generic wake_up()
    # path.  C0 skips the initial weight transfer, so keep its KV cache
    # resident instead of leaving the request pool CPU-backed.
    set_value("actor_rollout_ref.rollout.free_cache_engine", args.rollout_engine != "sglang")
    # veRL's colocated actor keeps FSDP resident while vLLM starts.  On the
    # 16 GiB WSL GPU, CPU offload is the supported vLLM escape hatch that
    # leaves enough device memory for at least one KV-cache block.
    set_value(
        "actor_rollout_ref.rollout.engine_kwargs.vllm.cpu_offload_gb",
        args.cpu_offload_gb,
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
    # The unified config carries a separate reference-model path.  Leaving its
    # DeepSeek example default in place can make colocated initialization load
    # an unrelated checkpoint (or fail while probing a nonexistent path).
    set_value("actor_rollout_ref.ref.model.path", str(args.model), force_add=True)
    set_value("actor_rollout_ref.ref.model.use_shm", False, force_add=True)
    set_value("actor_rollout_ref.ref.model.override_config.attn_implementation", "sdpa", force_add=True)
    set_value("actor_rollout_ref.ref.use_torch_compile", False)
    set_value("actor_rollout_ref.ref.fsdp_config.use_torch_compile", False)
    set_value("actor_rollout_ref.model.use_shm", False)
    # The isolated WSL env records and imports the external flash-attn wheel;
    # the actual FSDP model still uses padded SDPA so sequences cannot attend
    # across sample boundaries.  Any pure-PyTorch fallback lives under the
    # explicit ``tracesearch.compat.flash_attn`` namespace and is never a
    # package-name shadow.
    set_value("actor_rollout_ref.model.override_config.attn_implementation", "sdpa", force_add=True)
    set_value("actor_rollout_ref.model.enable_gradient_checkpointing", True)
    set_value("actor_rollout_ref.model.use_remove_padding", False)
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
    if args.rollout_engine == "sglang":
        # SGLang 0.5.11 cannot build its Qwen2/Qwen2.5 LoRA memory pool on
        # this stack (get_hidden_dim is not implemented for the model type).
        # C0 has a zero-initialized adapter and no optimizer update, so using
        # the base model is behaviorally equivalent for this serving comparison.
        set_value("actor_rollout_ref.model.lora_rank", 0)
        set_value("actor_rollout_ref.model.lora.rank", 0)

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
    set_value("actor_rollout_ref.actor.fsdp_config.use_torch_compile", False)
    set_value("actor_rollout_ref.actor.entropy_coeff", 0.0)
    set_value("actor_rollout_ref.actor.use_kl_loss", False)
    set_value("actor_rollout_ref.actor.kl_loss_coef", 0.0)
    set_value("actor_rollout_ref.actor.use_rollout_log_probs", True)
    set_value("actor_rollout_ref.actor.optim.lr", 1e-6)
    set_value("actor_rollout_ref.actor.optim.weight_decay", 0.0)

    # Preserve the owned (fully resolved) profile in the backend config.  This
    # is part of the artifact contract, not a second source of runtime values.
    set_value("tracesearch.runtime_profile_path", getattr(args, "_runtime_profile_path", None), force_add=True)
    set_value("tracesearch.runtime_profile", profile, force_add=True)

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


def _load_lora_checkpoint(path: Path) -> dict[str, Any]:
    """Load only LoRA tensors from one real veRL actor checkpoint tree."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - backend environment dependent.
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}", "files": [], "tensors": {}}

    files = sorted(path.rglob("model_world_size_*_rank_*.pt")) if path.exists() else []
    tensors: dict[str, Any] = {}
    errors: list[str] = []
    for file in files:
        try:
            payload = torch.load(file, map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
                payload = payload["state_dict"]
            if not isinstance(payload, dict):
                errors.append(f"{file}: payload is {type(payload).__name__}")
                continue
            for name, value in payload.items():
                if "lora" not in str(name).lower() or not hasattr(value, "detach"):
                    continue
                tensors[str(name)] = value.detach().float().cpu().contiguous()
        except Exception as exc:
            errors.append(f"{file}: {type(exc).__name__}: {exc}")

    digest = hashlib.sha256()
    tensor_meta: dict[str, Any] = {}
    total_parameters = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        raw = tensor.numpy().tobytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        tensor_meta[name] = {"shape": list(tensor.shape), "numel": int(tensor.numel())}
        total_parameters += int(tensor.numel())
    return {
        "path": str(path),
        "files": [str(file) for file in files],
        "file_count": len(files),
        "tensors": tensors,
        "tensor_meta": tensor_meta,
        "tensor_count": len(tensors),
        "total_parameters": total_parameters,
        "digest": digest.hexdigest() if tensors else None,
        "errors": errors,
    }


def _checkpoint_delta_evidence(output: Path) -> dict[str, Any]:
    """Compare the pre-update worker snapshot with the saved actor checkpoint.

    The comparison intentionally uses files written by veRL's actor worker.
    Parameter probes remain useful diagnostics, but cannot prove that the
    optimizer changed policy parameters because they wrap an internal method.
    """

    before = _load_lora_checkpoint(output / "pre_update" / "actor")
    after = _load_lora_checkpoint(output / "checkpoints" / "global_step_1" / "actor")
    before_tensors = before.get("tensors", {})
    after_tensors = after.get("tensors", {})
    common = sorted(set(before_tensors) & set(after_tensors))
    changed_names: list[str] = []
    max_abs = 0.0
    squared_l2 = 0.0
    for name in common:
        left = before_tensors[name]
        right = after_tensors[name]
        if tuple(left.shape) != tuple(right.shape):
            changed_names.append(name)
            max_abs = float("inf")
            continue
        delta = (right - left).abs()
        current_max = float(delta.max().item()) if delta.numel() else 0.0
        if current_max > 0.0:
            changed_names.append(name)
        max_abs = max(max_abs, current_max)
        squared_l2 += float((delta * delta).sum().item())
    return {
        "evidence_source": "checkpoint_delta",
        "before_checkpoint": before.get("path"),
        "after_checkpoint": after.get("path"),
        "before_file_count": before.get("file_count", 0),
        "after_file_count": after.get("file_count", 0),
        "before_tensor_count": before.get("tensor_count", 0),
        "after_tensor_count": after.get("tensor_count", 0),
        "common_tensor_count": len(common),
        "before_total_trainable_parameters": before.get("total_parameters", 0),
        "after_total_trainable_parameters": after.get("total_parameters", 0),
        "before_policy_digest": before.get("digest"),
        "after_policy_digest": after.get("digest"),
        "changed_trainable_tensor_count": len(changed_names),
        "changed_tensor_names": changed_names,
        "max_abs_parameter_delta": max_abs if common else None,
        "total_l2_parameter_delta": squared_l2**0.5 if common else None,
        "nonzero_parameter_updates": len(changed_names),
        "before_errors": before.get("errors", []),
        "after_errors": after.get("errors", []),
    }


def _parameter_evidence(output: Path) -> dict[str, Any]:
    checkpoint = _checkpoint_delta_evidence(output)
    probes = []
    for path in sorted((output / "parameter_probe").glob("actor_update_*.json")):
        try:
            probes.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return {
        "probe_count": len(probes),
        "successful_probe_count": sum(1 for item in probes if item.get("update_succeeded")),
        "probe_evidence_source": "diagnostic_only",
        **checkpoint,
        "probes": probes,
    }


def _observer_batches(output: Path, observer: RuntimeTrainingObserver | None) -> list[dict[str, Any]]:
    batches = list(observer.records) if observer is not None else []
    observer_path = output / "training_observer.json"
    if not batches and observer_path.exists():
        try:
            remote = json.loads(observer_path.read_text(encoding="utf-8"))
            batches = list(remote.get("batches", []))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return batches


def _runtime_groups(output: Path, tasks: list[Any], group_size: int, observer: RuntimeTrainingObserver | None) -> list[dict[str, Any]]:
    """Read groups from the live Ray observer before derived artifacts exist."""

    groups_by_key: dict[str, dict[str, Any]] = {}
    for batch in _observer_batches(output, observer):
        for group in batch.get("groups", []):
            key = str(group.get("group_id") or (group.get("task_ids") or [""])[0])
            groups_by_key[key] = group
    if groups_by_key:
        return list(groups_by_key.values())

    # A backend may finish its durable rollout JSONL write without reaching
    # the observer hook.  Use the same explicit missing-slot policy as the
    # evaluator, but do not silently treat an absent group as a successful run.
    if (output / "tracesearch_rollouts.jsonl").exists():
        return _groups_from_rollouts(output, tasks, group_size)
    return []


def _groups_from_rollouts(output: Path, tasks: list[Any], group_size: int) -> list[dict[str, Any]]:
    source = output / "tracesearch_rollouts.jsonl"
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[(str(row["task_id"]), int(row["sample_index"]))] = row
    groups: list[dict[str, Any]] = []
    for task in tasks:
        slots = [latest.get((task.task_id, index)) for index in range(group_size)]
        rewards = [float(row["reward"]["total"]) if row is not None else 0.0 for row in slots]
        mean = statistics.mean(rewards) if rewards else 0.0
        groups.append(
            {
                "group_id": task.task_id,
                "task_ids": [task.task_id] * group_size,
                "rewards": rewards,
                "group_exact_match_variance": statistics.pvariance(rewards) if rewards else 0.0,
                "missing_slots": [index for index, row in enumerate(slots) if row is None],
                "reward_mean": mean,
            }
        )
    return groups


def _runtime_metrics(output: Path, args: argparse.Namespace, observer: RuntimeTrainingObserver | None) -> dict[str, Any]:
    batches = _observer_batches(output, observer)
    parameter_evidence = _parameter_evidence(output)
    observed_actor_update_calls = sum(int(row.get("observed_actor_update_calls", 0)) for row in batches)
    gates = [row.get("update_gate") for row in batches if row.get("update_gate")]
    mask_parities = [row.get("live_mask_parity") for row in batches if row.get("live_mask_parity") is not None]
    advantage_parities = [row.get("live_advantage_parity") for row in batches if row.get("live_advantage_parity") is not None]
    return {
        "phase": args.phase,
        "requested_optimizer_steps": {"c0": 0, "c1": 1, "c2": 10}[args.phase],
        "observed_actor_update_calls": observed_actor_update_calls if batches else None,
        "observed_optimizer_steps": None,
        "nonzero_parameter_updates": parameter_evidence["nonzero_parameter_updates"],
        "optimizer_steps_source": "not_claimed; checkpoint delta is the C1 update proof",
        "update_gates": gates,
        "live_mask_parity": all(item.get("parity") is True for item in mask_parities) if mask_parities else None,
        "live_advantage_parity": all(item.get("parity") is True for item in advantage_parities) if advantage_parities else None,
        "batches": batches,
        "parameter_evidence": {key: value for key, value in parameter_evidence.items() if key != "probes"},
    }


def _c1_status(phase: str, groups: list[dict[str, Any]], metrics: dict[str, Any]) -> str:
    """Apply the TRD's C1 proof gate without changing reward semantics."""

    if phase != "c1":
        return "completed"
    if not groups:
        return "backend_failed"
    if not any(float(group.get("group_exact_match_variance", 0.0)) > 0.0 for group in groups):
        return "no_learning_signal"
    if "parity_failed" in metrics.get("update_gates", []):
        return "parity_failed"
    if "update_failed" in metrics.get("update_gates", []):
        return "update_failed"
    if metrics.get("live_mask_parity") is False or metrics.get("live_advantage_parity") is False:
        return "parity_failed"
    if int(metrics.get("observed_actor_update_calls") or 0) <= 0:
        return "update_failed"
    if int(metrics.get("nonzero_parameter_updates") or 0) <= 0:
        return "update_failed"
    return "completed"


def _run_checkpoint_reload(output: Path, model: str) -> bool:
    checkpoint = output / "checkpoints" / "global_step_1"
    result_path = output / "checkpoint_reload.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tracesearch.cli.reload_m1c_checkpoint",
            "--base-model",
            model,
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(result_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    (output / "checkpoint_reload_process.log").write_text(
        completed.stdout + ("\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if not result_path.exists():
        return False
    try:
        return bool(json.loads(result_path.read_text(encoding="utf-8")).get("success")) and completed.returncode == 0
    except json.JSONDecodeError:
        return False


def _write_artifacts(
    output: Path,
    args: argparse.Namespace,
    config: Any,
    tasks: list[Any],
    provenance: dict[str, Any],
    *,
    c0_batch_weight_sync: str,
    observer: RuntimeTrainingObserver | None = None,
    status: str = "completed",
    failure: BaseException | None = None,
) -> None:
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
        "selected_task_ids": [task.task_id for task in tasks],
        "model": str(args.model),
        "seed": args.seed,
        "group_size": args.group_size,
        "max_turns": args.max_turns,
        "max_tokens": args.max_tokens,
        "runtime_optimizations": {
            "c0_batch_end_weight_sync": c0_batch_weight_sync,
            "cpu_offload_gb": args.cpu_offload_gb,
            "enforce_eager": args.enforce_eager,
            "rollout_engine": args.rollout_engine,
            "sglang_lora_disabled": args.rollout_engine == "sglang",
        },
        "runtime_profile": getattr(args, "_runtime_profile_path", None),
        "dependency_resolution_note": "rLLM current metadata conflicts with veRL/vLLM numpy constraints; this isolated env uses the explicitly recorded numpy override.",
    }
    _write_json(output / "backend_manifest.json", manifest)
    _write_json(output / "resolved_config.json", OmegaConf.to_container(config, resolve=False))
    _write_json(output / "flash_attn_provenance.json", versions.get("flash_attn_provenance", {}))
    _write_json(output / "mask_parity.json", _mask_parity())
    _write_json(output / "advantage_parity.json", _advantage_parity())
    summary = _summarize_rollouts(output, tasks, args.group_size, args.phase)
    metrics = _runtime_metrics(output, args, observer)
    metrics.update(
        {
            "c0_batch_end_weight_sync": c0_batch_weight_sync,
            "mean_exact_match": summary["mean_exact_match"],
            "group_exact_match_variance": summary["group_exact_match_variance"],
            "zero_variance_groups": summary["group_exact_match_variance"] == 0.0,
            "loss_agg_mode": "seq-mean-token-mean",
            "status": status,
        }
    )
    _write_json(output / "metrics.json", metrics)
    parameter_evidence = _parameter_evidence(output)
    _write_json(output / "parameter_delta.json", parameter_evidence)
    checkpoint_path = output / "checkpoints" / "global_step_1"
    _write_json(
        output / "checkpoint_proof.json",
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_step": 1 if checkpoint_path.exists() else None,
            "pre_update_snapshot": str(output / "pre_update" / "actor") if (output / "pre_update" / "actor").exists() else None,
            "policy_tensor_digest_before_update": parameter_evidence["before_policy_digest"],
            "policy_tensor_digest_after_update": parameter_evidence["after_policy_digest"],
            "observed_actor_update_count": metrics["observed_actor_update_calls"],
            "observed_nonzero_policy_update_count": parameter_evidence["nonzero_parameter_updates"],
            "observed_optimizer_steps": metrics["observed_optimizer_steps"],
            "evidence_source": parameter_evidence["evidence_source"],
        },
    )
    if failure is not None:
        _write_json(
            output / "failure.json",
            {"exception": type(failure).__name__, "message": str(failure), "traceback": traceback.format_exc()},
        )
    (output / "summary.md").write_text(
        "\n".join(
            [
                f"# TraceSearch M1-C {args.phase}",
                "",
                f"- Backend commit: `{RLLM_COMMIT}` / veRL `{VERL_COMMIT}` / vLLM `{VLLM_COMMIT}`",
                f"- Project commit: `{manifest['git_commit']}`",
                f"- Git dirty: `{manifest['git_dirty']}`",
                f"- Requested optimizer steps: `{metrics['requested_optimizer_steps']}`",
                f"- Observed actor update calls: `{metrics['observed_actor_update_calls']}`",
                f"- Observed optimizer steps: `{metrics['observed_optimizer_steps']}`",
                f"- C0 batch-end weight sync: `{c0_batch_weight_sync}`",
                f"- Rollout slots: `{summary['completed_rollout_slots']}/{summary['denominator']}`",
                f"- Mean exact match: `{summary['mean_exact_match']}`",
                f"- Group exact-match variance: `{summary['group_exact_match_variance']}`",
                "- Reward: normalized exact match only; no process reward, KL, rejection sampling, or fatal-aware shaping.",
                "- `group_reward_variance` is intentionally not used: evaluator variance is named `group_exact_match_variance`.",
                f"- Run status: `{status}`",
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
    os.environ["TRACESEARCH_M1C_PHASE"] = args.phase
    if args.phase == "c0":
        # AgentTrainer constructs the real VerlBackend inside its Ray worker;
        # install the post-batch C0 guard from the workflow module there too.
        os.environ["TRACESEARCH_C0_SKIP_BATCH_END_SYNC"] = "1"
    if args.phase == "c0" and args.rollout_engine == "sglang":
        # The actual veRL backend is instantiated in the Ray worker; the
        # workflow module applies the C0 initial-sync guard there on import.
        os.environ["TRACESEARCH_C0_SGLANG_SKIP_INITIAL_SYNC"] = "1"
    output = Path(args.output).resolve()
    _prepare_output(output, overwrite=args.overwrite_output)
    _write_run_status(output, "initializing", phase=args.phase)
    os.environ["TRACESEARCH_M1C_OBSERVER_DIR"] = str(output)
    tasks: list[Any] = []
    provenance: dict[str, Any] = {}
    config: Any = None
    observer: RuntimeTrainingObserver | None = None
    c0_batch_weight_sync = "not_applicable"
    try:
        tasks, provenance = _load_tasks(Path(args.dataset), args.task_count, args.task_id)
        config = _compose_config(args, output)
        _write_json(output / "resolved_config.json", __import__("omegaconf").OmegaConf.to_container(config, resolve=False))
        _write_run_status(output, "running", phase=args.phase, task_count=len(tasks), group_size=args.group_size)

        from importlib import import_module
        from rllm.trainer import AgentTrainer
        from tracesearch.training.rllm_workflow import TraceSearchWorkflow
        from tracesearch.training.m1c_observer import install_training_observer

        verl_backend = import_module("rllm.trainer.verl.verl_backend")
        observer = RuntimeTrainingObserver(output, phase=args.phase)
        install_training_observer(verl_backend, observer)

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
        if args.phase == "c0":
            from tracesearch.training.rllm_optimizations import install_c0_post_batch_weight_sync_skip

            c0_batch_weight_sync = install_c0_post_batch_weight_sync_skip(verl_backend, phase="c0")
            if args.rollout_engine == "sglang":
                c0_batch_weight_sync = f"{c0_batch_weight_sync}; initial_sglang=worker_import_guard"
        trainer.train()

        runtime_metrics = _runtime_metrics(output, args, observer)
        runtime_groups = _runtime_groups(output, tasks, args.group_size, observer)
        status = _c1_status(args.phase, runtime_groups, runtime_metrics)
        if status == "completed" and args.phase == "c1":
            if not _run_checkpoint_reload(output, str(args.model)):
                status = "reload_failed"
        _write_artifacts(
            output,
            args,
            config,
            tasks,
            provenance,
            c0_batch_weight_sync=c0_batch_weight_sync,
            observer=observer,
            status=status,
        )
        _write_run_status(output, status, phase=args.phase)
        if status in {"backend_failed", "update_failed", "reload_failed"}:
            raise RuntimeError(f"C1 proof gate failed with status={status}")
        return 0
    except Exception as exc:
        if config is not None and tasks:
            try:
                _write_artifacts(
                    output,
                    args,
                    config,
                    tasks,
                    provenance,
                    c0_batch_weight_sync=c0_batch_weight_sync,
                    observer=observer,
                    status="failed",
                    failure=exc,
                )
            except Exception as artifact_exc:
                _write_json(output / "failure.json", {"exception": type(exc).__name__, "message": str(exc), "artifact_error": f"{type(artifact_exc).__name__}: {artifact_exc}"})
        else:
            _write_json(output / "backend_manifest.json", {"schema_version": "m1c.backend.v1", "phase": args.phase, "git_dirty": True})
            _write_json(output / "resolved_config.json", {})
            _write_json(output / "failure.json", {"exception": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        _write_run_status(output, "failed", phase=args.phase, exception=type(exc).__name__)
        raise
    finally:
        os.environ.pop("TRACESEARCH_M1C_OBSERVER_DIR", None)
        os.environ.pop("TRACESEARCH_M1C_PHASE", None)
        os.environ.pop("TRACESEARCH_C0_SKIP_BATCH_END_SYNC", None)
        os.environ.pop("TRACESEARCH_C0_SGLANG_SKIP_INITIAL_SYNC", None)


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
