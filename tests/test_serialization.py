from tracesearch.agent import Action, ActionKind, Step, ToolResult, Trajectory
from tracesearch.data.io import load_trajectories, write_trajectories
from tracesearch.data.schema import Document, Evidence, Task


def test_canonical_objects_round_trip(tmp_path):
    document = Document("d-1", "Title", "Body", "local://d-1", {"x": 1})
    evidence = Evidence("d-1", "Title", "Body", 0.5, 1, document.url)
    tool_result = ToolResult("search", True, "q", "Body", [evidence], metadata={"x": True})
    trajectory = Trajectory(
        question="Question",
        task_id="task-1",
        steps=[Step("look", Action(ActionKind.SEARCH, "q"), tool_result=tool_result)],
        seed=7,
    )
    path = write_trajectories(tmp_path / "trajectories.jsonl", [trajectory])
    loaded = load_trajectories(path)
    assert loaded[0].to_dict() == trajectory.to_dict()
    assert Document.from_dict(document.to_dict()) == document
    assert Evidence.from_dict(evidence.to_dict()) == evidence
    assert Task.from_dict(Task("t", "q", ["a"], "test", ["d-1"]).to_dict()).task_id == "t"
