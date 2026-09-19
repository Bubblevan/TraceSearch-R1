import asyncio

from tracesearch.agent import Action, ActionKind, SearchAgent, Task, ToolErrorType, Trajectory
from tracesearch.data.schema import Step
from tracesearch.environment import FailureInjector, StaticVisitTool
from tracesearch.evaluation import evaluate_metrics
from tracesearch.experiment.manifest import ExperimentManifest, current_git_dirty, source_tree_hash


def test_rollout_identity_is_unique_for_sample_index():
    task = Task("task", "Question", ["answer"], "test")

    async def run():
        async def policy(_task, _trajectory):
            return Action(ActionKind.ANSWER, "answer")

        first = await SearchAgent(policy, StaticVisitTool({}), StaticVisitTool({}), seed=7, sample_index=0).run_async(task)
        second = await SearchAgent(policy, StaticVisitTool({}), StaticVisitTool({}), seed=7, sample_index=1).run_async(task)
        return first, second

    first, second = asyncio.run(run())
    assert first.rollout_id != second.rollout_id
    assert first.trajectory_id != second.trajectory_id
    assert first.sample_index == 0
    assert second.sample_index == 1


def test_keyed_fault_decisions_do_not_depend_on_call_order():
    surfaces = [
        ("task-a", "rollout-0", 0, "search"),
        ("task-a", "rollout-1", 0, "search"),
        ("task-b", "rollout-0", 1, "visit"),
    ]
    config = dict(timeout_rate=0.4, exception_rate=0.2, irrelevant_result_rate=0.3)
    left = FailureInjector(seed=11, **config)
    right = FailureInjector(seed=11, **config)
    left_values = {
        surface: left.next_fault(surface[3], surface[2], task_id=surface[0], rollout_id=surface[1])
        for surface in surfaces
    }
    right_values = {
        surface: right.next_fault(surface[3], surface[2], task_id=surface[0], rollout_id=surface[1])
        for surface in reversed(surfaces)
    }
    assert left_values == right_values


def test_evaluator_pads_missing_rollouts_and_reports_group_metrics():
    tasks = [Task("a", "Question A", ["yes"], "test"), Task("b", "Question B", ["yes"], "test")]
    trajectory = Trajectory(
        question="Question A",
        task_id="a",
        rollout_id="a-0",
        sample_index=0,
        answer="yes",
        steps=[Step("answer", Action(ActionKind.ANSWER, "yes"))],
        termination="answer",
    )
    metrics = evaluate_metrics(tasks, {"a": [trajectory], "b": []}, expected_rollouts=2)
    assert metrics["expected_rollout_count"] == 2
    assert metrics["observed_rollout_count"] == 1
    assert metrics["missing_trajectory_count"] == 3
    assert metrics["pass@1"] == 0.5
    assert metrics["pass@k"] == 0.5
    assert metrics["group_reward_variance"] == 0.125
    assert metrics["termination_histogram"]["missing"] == 3


def test_exception_mapping_records_specific_tool_error_type():
    class TimeoutSearch:
        def search(self, _query):
            raise TimeoutError("late")

    def policy(_question, steps):
        return ("search", Action(ActionKind.SEARCH, "q")) if not steps else ("answer", Action(ActionKind.ANSWER, "done"))

    result = SearchAgent(policy, TimeoutSearch(), StaticVisitTool({}), max_turns=2).run("q")
    assert result.steps[0].tool_result is not None
    assert result.steps[0].tool_result.error_type is ToolErrorType.TIMEOUT


def test_policy_view_filters_evaluation_metadata():
    task = Task(
        "task",
        "Question",
        ["secret answer"],
        "test",
        ["gold-doc"],
        {"topic": "public", "eval_only": "secret", "hop_count": 2, "answer_alias": "secret"},
    )
    view = task.policy_view()
    assert view.answers == []
    assert view.gold_evidence_ids == []
    assert view.metadata == {"topic": "public"}


def test_manifest_records_source_state_fields():
    manifest = ExperimentManifest(
        run_id="run",
        git_commit="538c6835653a5af1c7d8565e35bed34c9506c41f",
        git_dirty=False,
        source_tree_hash="abc",
    )
    restored = ExperimentManifest.from_dict(manifest.to_dict())
    assert restored.git_dirty is False
    assert restored.source_tree_hash == "abc"
    assert isinstance(current_git_dirty(), bool)
    assert len(source_tree_hash()) == 64
