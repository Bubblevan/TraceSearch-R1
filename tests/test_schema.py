import pytest

from tracesearch.agent import Action, ActionKind, Step, TerminationReason, ToolErrorType, ToolResult, Trajectory
from tracesearch.data.schema import Evidence, Task


def test_schema_enums_and_derived_properties():
    result = ToolResult(tool_name="search", ok=True, request="q", evidence=[Evidence("d", "D", "text", 1.0, 1)])
    trajectory = Trajectory(
        question="q",
        task_id="t",
        steps=[
            Step("search", Action(ActionKind.SEARCH, "q"), tool_result=result),
            Step("answer", Action(ActionKind.ANSWER, "yes")),
        ],
        answer="yes",
        termination_reason=TerminationReason.ANSWER,
    )
    assert ActionKind.SEARCH.value == "search"
    assert ToolErrorType.TIMEOUT.value == "timeout"
    assert trajectory.tool_turns == 1
    assert trajectory.search_calls == 1
    assert trajectory.successful_tool_calls == 1
    assert trajectory.failed_tool_calls == 0
    assert trajectory.termination_reason is TerminationReason.ANSWER


def test_invalid_states_are_rejected():
    with pytest.raises(ValueError):
        Action("unknown", "x")
    with pytest.raises(ValueError):
        ToolResult(tool_name="search", ok=False, request="q")
    with pytest.raises(ValueError):
        Task(task_id="", question="q")
