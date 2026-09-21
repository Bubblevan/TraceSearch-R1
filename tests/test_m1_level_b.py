import asyncio

import pytest

from tracesearch.agent import (
    Action,
    ActionKind,
    LLMGenerationConfig,
    LLMPolicy,
    ModelGeneration,
    SearchAgent,
    SAMPLING_SEED_SCHEME,
    derive_sampling_seed,
    gather_response_logprobs,
    sampling_seed_table,
)
from tracesearch.data import Step, Task, Trajectory
from tracesearch.environment import StaticSearchTool, StaticVisitTool
from tracesearch.training import GenerationRecord, TokenProvenanceError, TrainingTrace
from tracesearch.training.grpo import grpo_loss


class ExactFakeClient:
    def __init__(self):
        self.calls = 0

    async def generate(self, messages, *, model, config):
        self.calls += 1
        if self.calls == 1:
            return ModelGeneration(
                "<think>search</think><search>query</search>",
                token_ids=(11, 12),
                logprobs=(-0.1, -0.2),
                prompt_token_ids=(1, 2),
                sampling_seed=config.sampling_seed,
                finish_reason="stop",
                metadata={"checkpoint": "fake", "tokenizer_version": "fake-tokenizer"},
            )
        return ModelGeneration(
            "<think>answer</think><answer>yes</answer>",
            token_ids=(13, 14),
            logprobs=(-0.3, -0.4),
            prompt_token_ids=(1, 2, 90, 91),
            sampling_seed=config.sampling_seed,
            finish_reason="stop",
            metadata={"checkpoint": "fake", "tokenizer_version": "fake-tokenizer"},
        )


def test_exact_generation_records_and_zero_loss_observation_projection():
    client = ExactFakeClient()
    policy = LLMPolicy(client, model="fake", generation=LLMGenerationConfig(), require_token_ids=True)
    task = Task("task", "Question", ["yes"], "dev")
    trajectory = asyncio.run(
        SearchAgent(
            policy,
            StaticSearchTool({"query": "evidence"}),
            StaticVisitTool({}),
            max_turns=2,
            seed=7,
            sample_index=0,
        ).run_async(task)
    )
    trace = TrainingTrace.from_trajectory(
        trajectory,
        observation_token_ids={0: (90, 91)},
        policy_version="fake",
    )
    assert [record.prompt_ids for record in trace.generation_records] == [(1, 2), (1, 2, 90, 91)]
    assert [record.response_ids for record in trace.generation_records] == [(11, 12), (13, 14)]
    assert trace.response_ids == (11, 12, 90, 91, 13, 14)
    assert trace.response_mask == (1, 1, 0, 0, 1, 1)
    assert trace.step_ids == (0, 0, -1, -1, 1, 1)
    assert trace.old_logprobs == pytest.approx((-0.1, -0.2, 0.0, 0.0, -0.3, -0.4))
    with pytest.raises(TokenProvenanceError, match="fake flattened"):
        _ = trace.input_ids


def test_missing_old_logprobs_is_not_silently_repaired():
    record = GenerationRecord(step_index=0, prompt_ids=(1,), response_ids=(2,), response_logprobs=None)
    with pytest.raises(TokenProvenanceError, match="old logprobs"):
        TrainingTrace.from_generation_records(
            task_id="task",
            rollout_id="rollout",
            sample_index=0,
            policy_version="fake",
            generation_records=[record],
        )


def test_partial_generation_provenance_is_rejected():
    task = Task("task", "Question", ["yes"], "dev")
    first = GenerationRecord(step_index=0, prompt_ids=(1,), response_ids=(2,), response_logprobs=(0.0,))
    trajectory = Trajectory(
        question=task.question,
        task_id=task.task_id,
        steps=[
            Step(thought="", action=Action(ActionKind.SEARCH, "q"), metadata={"generation_record": first.to_dict()}),
            Step(thought="", action=Action(ActionKind.ANSWER, "yes"), metadata={"response_token_ids": [3]}),
        ],
    )
    with pytest.raises(TokenProvenanceError, match="incomplete"):
        TrainingTrace.from_trajectory(trajectory, observation_token_ids={0: (4,)}, policy_version="fake")


def test_grpo_reduces_each_response_before_batch_average():
    torch = pytest.importorskip("torch")
    policy = torch.zeros((2, 20), requires_grad=True)
    policy.data[1, :] = torch.log(torch.tensor(1.1))
    old = torch.zeros_like(policy)
    advantages = torch.ones(2)
    mask = torch.zeros((2, 20))
    mask[0, :2] = 1
    mask[1, :] = 1
    loss = grpo_loss(policy, old, advantages, mask)
    assert float(loss.detach()) == pytest.approx(-1.05, abs=1e-6)


def test_kl_requires_explicit_reference_policy():
    torch = pytest.importorskip("torch")
    policy = torch.zeros((1, 2))
    old = torch.zeros_like(policy)
    mask = torch.ones_like(policy)
    with pytest.raises(ValueError, match="reference_logprobs"):
        grpo_loss(policy, old, torch.ones(1), mask, kl_coef=0.1)
    reference = torch.full_like(policy, -0.5)
    value = grpo_loss(policy, old, torch.ones(1), mask, kl_coef=0.1, reference_logprobs=reference)
    assert value.item() == pytest.approx(-1.0 + 0.05, abs=1e-6)


def test_sampling_seed_is_identity_stable():
    left = derive_sampling_seed("task", "rollout-a", 0, 1, base_seed=42)
    right = derive_sampling_seed("task", "rollout-a", 0, 1, base_seed=42)
    different = derive_sampling_seed("task", "rollout-b", 1, 1, base_seed=42)
    assert left == right
    assert left != different


def test_sampling_seed_v2_ignores_transient_rollout_id():
    left = derive_sampling_seed("task", "rollout-a", 0, 1, base_seed=42)
    right = derive_sampling_seed("task", "rollout-b", 0, 1, base_seed=42)
    assert left == right
    assert SAMPLING_SEED_SCHEME == "v2:blake2b(base_seed,task_id,sample_index,step_index)"


def test_sampling_seed_table_is_process_independent_and_index_sensitive():
    first = sampling_seed_table("task", 2, 3, base_seed=42)
    second = sampling_seed_table("task", 2, 3, base_seed=42)
    assert first == second
    assert len(first) == 6
    assert first[0]["derived_sampling_seed"] != first[3]["derived_sampling_seed"]
    assert first[0]["derived_sampling_seed"] != first[1]["derived_sampling_seed"]


def test_llm_policy_sampling_does_not_depend_on_rollout_uuid():
    class CaptureClient:
        def __init__(self):
            self.seeds = []

        async def generate(self, messages, *, model, config):
            del messages, model
            self.seeds.append(config.sampling_seed)
            return ModelGeneration(
                "<think>answer</think><answer>yes</answer>",
                token_ids=(7,),
                prompt_token_ids=(1,),
                sampling_seed=config.sampling_seed,
            )

    client = CaptureClient()
    policy = LLMPolicy(
        client,
        model="fake",
        generation=LLMGenerationConfig(sampling_seed=42),
        require_token_ids=True,
    )
    task = Task("task", "Question", ["yes"], "dev")
    first = Trajectory(task.question, task_id=task.task_id, rollout_id="uuid-a", sample_index=1)
    second = Trajectory(task.question, task_id=task.task_id, rollout_id="uuid-b", sample_index=1)
    asyncio.run(policy.act(task, first))
    asyncio.run(policy.act(task, second))
    assert client.seeds[0] == client.seeds[1]


def test_logprob_gather_uses_supplied_response_ids():
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
    )
    values = gather_response_logprobs(logits, prompt_length=2, response_token_ids=[1, 0])
    expected = torch.log_softmax(logits[:, 1:3, :], dim=-1).gather(
        -1, torch.tensor([[[1], [0]]])
    ).squeeze().tolist()
    assert values == pytest.approx(expected)
