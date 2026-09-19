from tracesearch.agent import Action, ActionKind, Step, ToolResult, Trajectory
from tracesearch.data.schema import Evidence, Task
from tracesearch.evaluation import evaluate_metrics, normalize_answer


def test_metrics_cover_answer_evidence_tools_and_termination():
    task = Task("t", "Where?", ["Aster City"], "test", ["d"])
    result = ToolResult("search", True, "where", evidence=[Evidence("d", "D", "Aster City", 1, 1)])
    trajectory = Trajectory(
        "Where?",
        [Step("search", Action(ActionKind.SEARCH, "where"), tool_result=result), Step("answer", Action(ActionKind.ANSWER, "aster city"))],
        answer="aster city",
        task_id="t",
        termination="answer",
    )
    metrics = evaluate_metrics([task], [trajectory], top_k=1)
    assert normalize_answer(" Aster-City! ") == "aster city"
    assert metrics["normalized_exact_match"] == 1.0
    assert metrics["search_recall_at_k"] == 1.0
    assert metrics["avg_search_calls"] == 1.0
    assert metrics["termination_histogram"] == {"answer": 1}
