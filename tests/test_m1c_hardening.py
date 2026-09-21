from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tracesearch.cli.run_m1c_backend import (
    _apply_runtime_profile,
    _c1_status,
    _checkpoint_delta_evidence,
    _load_tasks,
    _prepare_output,
    _runtime_metrics,
)
from tracesearch.data.schema import Action, ActionKind, Step, TerminationReason, Trajectory
from tracesearch.training.m1c_observer import RuntimeTrainingObserver
from tracesearch.training.rllm_optimizations import install_c0_post_batch_weight_sync_skip
from tracesearch.training.rllm_workflow import _rllm_termination_value


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
    assert _c1_status("c1", [{"group_exact_match_variance": 0.25}], metrics) == "update_failed"

    metrics = {
        "observed_actor_update_calls": 1,
        "nonzero_parameter_updates": 1,
        "live_mask_parity": True,
        "live_advantage_parity": True,
        "update_gates": [],
    }
    assert _c1_status("c1", [{"group_exact_match_variance": 0.25}], metrics) == "completed"


def test_probe_does_not_claim_optimizer_step_or_policy_update(tmp_path: Path):
    args = Namespace(phase="c1")
    observer = SimpleNamespace(records=[{"observed_actor_update_calls": 1}])
    probe_dir = tmp_path / "parameter_probe"
    probe_dir.mkdir()
    (probe_dir / "actor_update_1.json").write_text(
        '{"update_succeeded": true, "changed_trainable_tensor_count": 1, "max_abs_parameter_delta": 0.1, "total_l2_parameter_delta": 0.2}',
        encoding="utf-8",
    )
    metrics = _runtime_metrics(tmp_path, args, observer)
    assert metrics["observed_optimizer_steps"] is None
    assert metrics["nonzero_parameter_updates"] == 0


def test_runtime_counters_are_not_phase_derived(tmp_path: Path):
    args = Namespace(phase="c1")
    metrics = _runtime_metrics(tmp_path, args, None)
    assert metrics["requested_optimizer_steps"] == 1
    assert metrics["observed_actor_update_calls"] is None
    assert metrics["observed_optimizer_steps"] is None


def test_runtime_counters_read_remote_ray_observer(tmp_path: Path):
    args = Namespace(phase="c1")
    (tmp_path / "training_observer.json").write_text(
        '{"batches": [{"observed_actor_update_calls": 1}]}',
        encoding="utf-8",
    )
    metrics = _runtime_metrics(tmp_path, args, None)
    assert metrics["observed_actor_update_calls"] == 1
    assert metrics["observed_optimizer_steps"] is None


def test_stale_output_requires_explicit_overwrite(tmp_path: Path):
    output = tmp_path / "run"
    _prepare_output(output, overwrite=False)
    (output / "training_observer.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite-output"):
        _prepare_output(output, overwrite=False)
    _prepare_output(output, overwrite=True)
    assert not (output / "training_observer.json").exists()


def test_task_id_selector_is_exact():
    tasks, _ = _load_tasks(Path("data/m1/dev.jsonl"), task_count=3, task_id="m1-two-hop")
    assert [task.task_id for task in tasks] == ["m1-two-hop"]


def _fake_trajectory(uid: str, reward: float = 1.0):
    first = SimpleNamespace(
        model_output=SimpleNamespace(prompt_ids=[10, 11], completion_ids=[20, 21]),
    )
    second = SimpleNamespace(
        model_output=SimpleNamespace(prompt_ids=[10, 11, 20, 21, 30, 31], completion_ids=[40, 41]),
    )
    return SimpleNamespace(uid=uid, reward=reward, metadata={"task_id": "fake"}, steps=[first, second])


def test_live_mask_parity_uses_step_ids_and_excludes_padding():
    trajectory = _fake_trajectory("u0")
    batch = SimpleNamespace(
        batch={
            "responses": [[20, 21, 30, 31, 40, 41], [999, 999, 999, 999, 999, 999]],
            "response_mask": [[1, 1, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1]],
            "attention_mask": [[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]],
        },
        non_tensor_batch={"step_ids": ["u0", "pad"], "is_pad_step": [False, True]},
    )
    state = SimpleNamespace(backend_batch=batch, trajectory_groups=[SimpleNamespace(trajectories=[trajectory])])
    parity = RuntimeTrainingObserver._live_mask_parity(state)
    assert parity["parity"] is True
    assert len(parity["rows"]) == 1
    assert parity["rows"][0]["padding_tokens_excluded"] == 0
    assert parity["rows"][0]["step_boundaries"] == [
        {"step_index": 0, "observation_length": 0, "action_length": 2},
        {"step_index": 1, "observation_length": 2, "action_length": 2},
    ]


def test_live_mask_parity_matches_reordered_segments_within_rollout():
    trajectory = SimpleNamespace(
        uid="u0",
        steps=[
            SimpleNamespace(model_output=SimpleNamespace(prompt_ids=[10], completion_ids=[20, 21])),
            SimpleNamespace(model_output=SimpleNamespace(prompt_ids=[99], completion_ids=[40, 41, 42])),
        ],
    )
    batch = SimpleNamespace(
        batch={
            "responses": [[0, 40, 41, 42], [0, 0, 20, 21]],
            "response_mask": [[0, 1, 1, 1], [0, 0, 1, 1]],
            "attention_mask": [[0, 1, 1, 1], [0, 0, 1, 1]],
        },
        non_tensor_batch={"step_ids": ["u0", "u0"], "is_pad_step": [False, False]},
    )
    state = SimpleNamespace(backend_batch=batch, trajectory_groups=[SimpleNamespace(trajectories=[trajectory])])
    parity = RuntimeTrainingObserver._live_mask_parity(state)
    assert parity["parity"] is True
    assert [row["segment_index"] for row in parity["rows"]] == [1, 0]


def test_logprob_diagnostics_persist_distribution(tmp_path: Path):
    torch = pytest.importorskip("torch")
    observer = RuntimeTrainingObserver(tmp_path, phase="c1")
    batch = SimpleNamespace(
        batch={
            "old_log_probs": torch.tensor([[1.0, 2.0, 9.0, 4.0]]),
            "rollout_log_probs": torch.tensor([[0.0, 0.0, 9.0, 0.0]]),
            "response_mask": torch.tensor([[1, 1, 0, 1]]),
        }
    )
    state = SimpleNamespace(backend_batch=batch, metrics={})
    observer.observe_process_batch(state)
    record = json.loads((tmp_path / "rollout_logprob_diagnostics.json").read_text())["records"][0]
    assert record["mean"] == pytest.approx(7 / 3)
    assert record["std"] == pytest.approx(1.2472191289)
    assert record["p50"] == pytest.approx(2.0)
    assert record["p90"] == pytest.approx(3.6)
    assert record["p95"] == pytest.approx(3.8)
    assert record["p99"] == pytest.approx(3.96)
    assert record["max_abs"] == 4.0
    assert record["token_count"] == 3


def test_live_advantage_parity_reads_step_ids_and_sets_zero_signal_gate(tmp_path: Path):
    observer = RuntimeTrainingObserver(tmp_path, phase="c1")
    observer.live_mask = {"parity": True}
    trajectories = [_fake_trajectory("u0", 1.0), _fake_trajectory("u1", 1.0)]
    batch = SimpleNamespace(
        batch={
            "advantages": [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
            "response_mask": [[1, 1, 0, 0, 1, 1], [1, 1, 0, 0, 1, 1]],
        },
        non_tensor_batch={"step_ids": ["u0", "u1"], "is_pad_step": [False, False]},
    )
    state = SimpleNamespace(backend_batch=batch, trajectory_groups=[SimpleNamespace(trajectories=trajectories)])
    observer.observe_advantages(state)
    assert observer.live_advantage["parity"] is True
    assert observer.update_gate == "no_learning_signal"


def test_live_advantage_parity_uses_backend_epsilon_for_mixed_rewards(tmp_path: Path):
    observer = RuntimeTrainingObserver(tmp_path, phase="c1")
    observer.live_mask = {"parity": True}
    trajectories = [
        _fake_trajectory("u0", 0.0),
        _fake_trajectory("u1", 0.0),
        _fake_trajectory("u2", 1.0),
        _fake_trajectory("u3", 1.0),
    ]
    backend_advantage = 0.999998000004
    batch = SimpleNamespace(
        batch={
            "advantages": [
                [-backend_advantage] * 6,
                [-backend_advantage] * 6,
                [backend_advantage] * 6,
                [backend_advantage] * 6,
            ],
            "response_mask": [[1, 1, 0, 0, 1, 1]] * 4,
        },
        non_tensor_batch={"step_ids": ["u0", "u1", "u2", "u3"], "is_pad_step": [False] * 4},
    )
    state = SimpleNamespace(backend_batch=batch, trajectory_groups=[SimpleNamespace(trajectories=trajectories)])
    observer.observe_advantages(state)
    assert observer.live_advantage["parity"] is True
    assert observer.live_advantage["groups"][0]["max_abs_error"] == pytest.approx(0.0)
    assert observer.update_gate is None


def test_checkpoint_delta_is_primary_policy_update_evidence(tmp_path: Path):
    torch = pytest.importorskip("torch")
    before = tmp_path / "pre_update" / "actor"
    after = tmp_path / "checkpoints" / "global_step_1" / "actor"
    before.mkdir(parents=True)
    after.mkdir(parents=True)
    torch.save({"base.lora_A.weight": torch.tensor([[1.0, 2.0]])}, before / "model_world_size_1_rank_0.pt")
    torch.save({"base.lora_A.weight": torch.tensor([[1.0, 3.0]])}, after / "model_world_size_1_rank_0.pt")
    evidence = _checkpoint_delta_evidence(tmp_path)
    assert evidence["evidence_source"] == "checkpoint_delta"
    assert evidence["changed_trainable_tensor_count"] == 1
    assert evidence["nonzero_parameter_updates"] == 1


def test_policy_parse_failure_is_trainable_not_backend_fatal():
    trajectory = Trajectory(
        question="q",
        termination_reason=TerminationReason.POLICY_ERROR,
        steps=[
            Step(
                thought="",
                action=Action(ActionKind.ANSWER, ""),
                metadata={
                    "parse_failure_type": "malformed_tag",
                    "generation_record": {"prompt_ids": [1], "response_ids": [2]},
                },
            )
        ],
    )
    assert _rllm_termination_value(trajectory) == "unknown"


def test_policy_backend_failure_remains_fatal():
    trajectory = Trajectory(
        question="q",
        termination_reason=TerminationReason.POLICY_ERROR,
        steps=[Step(thought="", action=Action(ActionKind.ANSWER, ""), metadata={"failure_class": "policy_error"})],
    )
    assert _rllm_termination_value(trajectory) == "error"


@pytest.mark.parametrize(
    "backend_reason",
    ["max_prompt_length_exceeded", "max_response_length_exceeded", "timeout"],
)
def test_bounded_backend_termination_is_a_nonfatal_negative(backend_reason: str):
    trajectory = Trajectory(
        question="q",
        termination_reason=TerminationReason.POLICY_ERROR,
        steps=[
            Step(
                thought="",
                action=Action(ActionKind.ANSWER, ""),
                metadata={
                    "failure_class": "policy_error",
                    "exception_class": "TerminationEvent",
                    "backend_termination_reason": backend_reason,
                },
            )
        ],
    )
    assert _rllm_termination_value(trajectory) == backend_reason


def test_repository_does_not_shadow_external_flash_attn():
    repo = Path(__file__).resolve().parents[1]
    shadow_root = repo / "src" / "flash_attn"
    assert not any(path.suffix == ".py" for path in shadow_root.rglob("*.py"))
