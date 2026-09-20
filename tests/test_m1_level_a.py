import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from tracesearch.agent import (
    Action,
    ActionKind,
    ActionParseError,
    LLMPolicy,
    LLMGenerationConfig,
    ModelGeneration,
    OpenAICompatibleClient,
    ParseFailureType,
    SearchAgent,
    parse_policy_output,
)
from tracesearch.data import NQTaskAdapter, Step, Task, Trajectory
from tracesearch.environment import HTTPRetriever, StaticSearchTool, StaticVisitTool
from tracesearch.training import (
    TokenTurn,
    TrainingTrace,
    aggregate_group_diagnostics,
    compute_group_advantages,
    compute_outcome_reward,
    run_group_rollouts,
)
from tracesearch.training.grpo import grpo_loss


@pytest.mark.parametrize(
    ("text", "kind", "value"),
    [
        ("<think>find it</think><search>cats</search>", ActionKind.SEARCH, "cats"),
        ("<think>open result</think><visit>doc-1</visit>", ActionKind.VISIT, "doc-1"),
        ("<think>done</think><answer>Paris</answer>", ActionKind.ANSWER, "Paris"),
    ],
)
def test_action_parser_valid_actions(text, kind, value):
    parsed = parse_policy_output(text)
    assert parsed.action == Action(kind, value)
    assert parsed.raw_text == text


@pytest.mark.parametrize(
    ("text", "failure"),
    [
        ("<think>oops</think><search>cats", ParseFailureType.UNTERMINATED_TAG),
        ("<search>cats</search><answer>Paris</answer>", ParseFailureType.MULTIPLE_ACTIONS),
        ("<answer></answer>", ParseFailureType.EMPTY_PAYLOAD),
        ("<python>print(1)</python>", ParseFailureType.MALFORMED_TAG),
    ],
)
def test_action_parser_rejects_ambiguous_or_malformed_output(text, failure):
    with pytest.raises(ActionParseError) as error:
        parse_policy_output(text)
    assert error.value.failure_type is failure


class FakeModelClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.messages = []

    async def generate(self, messages, *, model, config):
        self.messages.append(messages)
        return self.outputs.pop(0)


def test_learned_policy_uses_policy_view_and_preserves_token_ids():
    client = FakeModelClient(
        [
            ModelGeneration("<think>search</think><search>capital</search>", token_ids=(11, 12)),
            ModelGeneration("<think>answer</think><answer>Paris</answer>", token_ids=(13, 14)),
        ]
    )
    policy = LLMPolicy(client, model="fake", require_token_ids=True)
    task = Task(
        "nq-1",
        "What is the capital?",
        ["Paris"],
        "dev",
        metadata={"eval_only_answer": "Paris", "secret": "do-not-send"},
    )

    async def run():
        return await SearchAgent(
            policy,
            StaticSearchTool({"capital": "France"}),
            StaticVisitTool({}),
            max_turns=2,
        ).run_async(task)

    trajectory = asyncio.run(run())
    assert trajectory.answer == "Paris"
    assert trajectory.steps[0].metadata["response_token_ids"] == [11, 12]
    assert trajectory.steps[1].metadata["response_token_ids"] == [13, 14]
    serialized_messages = json.dumps(client.messages, ensure_ascii=False)
    assert "do-not-send" not in serialized_messages
    assert "Paris" not in serialized_messages


def test_openai_compatible_client_fake_server_returns_actual_ids():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers["Content-Length"])
            received.append(json.loads(self.rfile.read(length)))
            payload = {
                "id": "fake-generation",
                "choices": [{"message": {"content": "<answer>Paris</answer>", "token_ids": [21, 22]}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = OpenAICompatibleClient(f"http://127.0.0.1:{server.server_port}")
        generation = asyncio.run(
            client.generate(
                [{"role": "user", "content": "Question"}],
                model="fake-checkpoint",
                config=LLMGenerationConfig(max_tokens=8, sampling_seed=17),
            )
        )
        assert generation.text == "<answer>Paris</answer>"
        assert generation.token_ids == (21, 22)
        assert received[0]["model"] == "fake-checkpoint"
        assert received[0]["seed"] == 17
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_training_trace_masks_only_policy_generated_tokens():
    trace = TrainingTrace.from_turns(
        task_id="task",
        rollout_id="rollout-0",
        sample_index=0,
        policy_version="fake-v1",
        turns=[
            TokenTurn("system", (1,)),
            TokenTurn("user", (2, 3)),
            TokenTurn("assistant", (4, 5), generated_by_policy=True, step_index=0),
            TokenTurn("tool", (6, 7)),
            TokenTurn("assistant", (8,), generated_by_policy=True, step_index=1),
        ],
    )
    assert trace.prompt_ids == (1, 2, 3)
    assert trace.response_ids == (4, 5, 6, 7, 8)
    assert trace.response_mask == (1, 1, 0, 0, 1)
    assert trace.full_response_mask == (0, 0, 0, 1, 1, 0, 0, 1)
    assert trace.step_ids == (0, 0, -1, -1, 1)


def test_training_trace_from_trajectory_requires_gateway_ids():
    trajectory = Trajectory(
        question="Question",
        task_id="task",
        rollout_id="rollout-0",
        sample_index=0,
        steps=[],
        answer="yes",
        termination="answer",
    )
    trajectory.steps.append(
        Step(thought="", action=Action(ActionKind.ANSWER, "yes"), metadata={"response_token_ids": [9, 10]})
    )
    trace = TrainingTrace.from_trajectory(
        trajectory,
        prompt_ids=[1, 2],
        observation_token_ids={},
        policy_version="fake-v1",
    )
    assert trace.response_ids == (9, 10)
    assert trace.response_mask == (1, 1)


def test_nq_adapter_is_deterministic_and_has_explicit_policy_surface():
    rows = [
        {
            "example_id": "b",
            "question_text": "Second?",
            "short_answers": ["two"],
            "gold_evidence_ids": ["doc-2"],
        },
        {
            "example_id": "a",
            "question": {"text": "First?"},
            "answers": ["one"],
        },
    ]
    adapter = NQTaskAdapter(upstream_revision="rev-1")
    dataset = adapter.adapt_rows(rows, split="dev")
    assert [task.task_id for task in dataset.tasks] == ["a", "b"]
    assert dataset.tasks[0].policy_view().answers == []
    assert dataset.tasks[0].policy_view().metadata == {"dataset": "natural_questions", "split": "dev"}
    assert dataset.provenance.data_hash == adapter.adapt_rows(rows, split="dev").provenance.data_hash


def test_reward_and_group_advantage_contract():
    task = Task("task", "Question", ["yes"], "dev")
    trajectory = Trajectory(
        question="Question", task_id="task", answer="yes", termination="answer"
    )
    reward = compute_outcome_reward(task, trajectory)
    assert reward.to_dict()["total"] == 1.0
    assert compute_outcome_reward(task, None).total == 0.0
    assert compute_group_advantages([0.0, 1.0]) == pytest.approx((-1.0, 1.0), abs=1e-6)
    assert compute_group_advantages([1.0, 1.0]) == (0.0, 0.0)


def test_group_rollouts_keep_failed_slots_and_ids():
    task = Task("task", "Question", ["yes"], "dev")

    def policy_factory(sample_index):
        if sample_index == 1:
            raise RuntimeError("worker failed")
        return lambda _question, _steps: ("answer", Action(ActionKind.ANSWER, "yes"))

    result = asyncio.run(
        run_group_rollouts(
            task,
            policy_factory,
            StaticSearchTool({}),
            StaticVisitTool({}),
            group_size=3,
            seed=9,
        )
    )
    assert [record.sample_index for record in result.records] == [0, 1, 2]
    assert len({record.rollout_id for record in result.records}) == 3
    assert result.completed_rollouts == 2
    assert result.missing_rollouts == 1
    assert result.records[1].status == "failed"
    assert result.group_reward_mean == pytest.approx(2 / 3)
    assert result.diagnostics.zero_variance_group is False
    aggregate = aggregate_group_diagnostics([result])
    assert aggregate["zero_variance_group_rate"] == 0.0


def test_http_retriever_structured_search_and_visit():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            if self.path == "/search":
                payload = {"results": [{"doc_id": "doc-1", "title": "Title", "snippet": "Snippet", "score": 0.9}]}
            elif self.path == "/visit/doc-1":
                payload = {"document": {"doc_id": "doc-1", "title": "Title", "text": "Full text"}}
            else:
                self.send_response(404)
                self.end_headers()
                return
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        retriever = HTTPRetriever(f"http://127.0.0.1:{server.server_port}", backend_version="test-v1")
        search = asyncio.run(retriever.search("query"))
        visit = asyncio.run(retriever.visit("doc-1"))
        assert search.ok and search.evidence[0].doc_id == "doc-1"
        assert visit.ok and visit.evidence[0].snippet == "Full text"
        assert search.metadata["backend_version"] == "test-v1"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_http_retriever_maps_invalid_and_unavailable_responses():
    invalid = HTTPRetriever("http://127.0.0.1:1", timeout_s=0.1)
    result = asyncio.run(invalid.search("query"))
    assert result.ok is False
    assert result.error_type.value == "transient"


def test_grpo_loss_ignores_observation_tokens():
    torch = pytest.importorskip("torch")
    policy = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    old = torch.zeros_like(policy)
    advantages = torch.tensor([1.0])
    mask = torch.tensor([[1, 0, 1]])
    loss = grpo_loss(policy, old, advantages, mask)
    loss.backward()
    assert policy.grad.tolist() == [[-0.5, 0.0, -0.5]]
