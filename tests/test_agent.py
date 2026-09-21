import unittest
import asyncio
from types import SimpleNamespace

from tracesearch.agent import Action, ActionKind, SearchAgent, Task
from tracesearch.environment import StaticSearchTool, StaticVisitTool


class SearchAgentTests(unittest.TestCase):
    def test_agent_records_interleaved_search_and_answer(self):
        def policy(_, steps):
            if not steps:
                return "Find a primary source.", Action(ActionKind.SEARCH, "trace search")
            return "Evidence is sufficient.", Action(ActionKind.ANSWER, "Answer [source](https://example.test)")

        agent = SearchAgent(policy, StaticSearchTool({"trace search": "https://example.test"}), StaticVisitTool({}))
        result = agent.run("What is TraceSearch?")

        self.assertEqual(result.termination, "answer")
        self.assertEqual(result.tool_turns, 1)
        self.assertIsNotNone(result.answer)

    def test_async_policy_does_not_receive_evaluation_answers(self):
        seen = []

        class Policy:
            async def act(self, task, trajectory):
                seen.append((task.answers, task.gold_evidence_ids))
                return Action(ActionKind.ANSWER, "fixture answer")

        agent = SearchAgent(Policy(), StaticSearchTool({}), StaticVisitTool({}))
        result = asyncio.run(
            agent.run_async(Task("task-1", "Question", ["fixture answer"], "test", ["doc-1"]))
        )

        self.assertEqual(result.termination, "answer")
        self.assertEqual(seen, [([], [])])

    def test_policy_exception_preserves_structured_backend_termination(self):
        class BackendTermination(Exception):
            reason = SimpleNamespace(value="max_prompt_length_exceeded")

        class Policy:
            async def act(self, task, trajectory):
                del task, trajectory
                raise BackendTermination("prompt limit")

        agent = SearchAgent(Policy(), StaticSearchTool({}), StaticVisitTool({}))
        result = asyncio.run(agent.run_async(Task("task-1", "Question", ["answer"], "test")))

        self.assertEqual(result.termination, "policy_error")
        self.assertEqual(result.steps[-1].metadata["exception_class"], "BackendTermination")
        self.assertEqual(
            result.steps[-1].metadata["backend_termination_reason"],
            "max_prompt_length_exceeded",
        )
