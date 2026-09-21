from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tracesearch.cli.run_m1c_backend import _apply_runtime_profile, _c1_status, _runtime_metrics
from tracesearch.training.rllm_optimizations import install_c0_post_batch_weight_sync_skip


def test_owned_runtime_profile_changes_composed_inputs():
    args = Namespace(
        config="configs/m1/runtime/4090-16g-wsl.yaml",
        gpu_memory_utilization=None,
        cpu_offload_gb=None,
        enforce_eager=None,
    )
    profile = _apply_runtime_profile(args)
    assert profile["profile_name"] == "4090-16g-wsl"
    assert args.gpu_memory_utilization == 0.9
    assert args.cpu_offload_gb == 0.0
    assert args.enforce_eager is False
    assert args.tensor_model_parallel_size == 1
    assert args.max_model_len == 1024


def test_c0_sync_guard_rejects_c1_before_patching():
    module = SimpleNamespace(VerlBackend=object)
    with pytest.raises(RuntimeError, match="phase='c1'"):
        install_c0_post_batch_weight_sync_skip(module, phase="c1")


def test_c1_status_requires_natural_signal_and_parameter_delta(tmp_path: Path):
    args = Namespace(phase="c1")
    observer = SimpleNamespace(records=[{"observed_actor_update_calls": 1}])
    metrics = _runtime_metrics(tmp_path, args, observer)
    assert _c1_status("c1", [{"group_exact_match_variance": 0.0}], metrics) == "no_learning_signal"
    assert _c1_status("c1", [{"group_exact_match_variance": 0.25}], metrics) == "failed"

    probe_dir = tmp_path / "parameter_probe"
    probe_dir.mkdir()
    (probe_dir / "actor_update_1.json").write_text(
        '{"update_succeeded": true, "changed_trainable_tensor_count": 1, "max_abs_parameter_delta": 0.1, "total_l2_parameter_delta": 0.2}',
        encoding="utf-8",
    )
    metrics = _runtime_metrics(tmp_path, args, observer)
    assert _c1_status("c1", [{"group_exact_match_variance": 0.25}], metrics) == "completed"


def test_runtime_counters_are_not_phase_derived(tmp_path: Path):
    args = Namespace(phase="c1")
    metrics = _runtime_metrics(tmp_path, args, None)
    assert metrics["requested_optimizer_steps"] == 1
    assert metrics["observed_actor_update_calls"] is None
    assert metrics["observed_optimizer_steps"] is None


def test_repository_does_not_shadow_external_flash_attn():
    repo = Path(__file__).resolve().parents[1]
    shadow_root = repo / "src" / "flash_attn"
    assert not any(path.suffix == ".py" for path in shadow_root.rglob("*.py"))
