import unittest

from tracesearch.agent import Action, ActionKind, SearchAgent
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
