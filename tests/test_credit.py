import unittest

from tracesearch.agent import Action, ActionKind, Step
from tracesearch.rewards import fatal_step_index, weighted_advantages


class CreditTests(unittest.TestCase):
    def test_fatal_step_keeps_a_recoverable_prefix(self):
        steps = [
            Step("useful search", Action(ActionKind.SEARCH, "q")),
            Step("bad", Action(ActionKind.SEARCH, "q1"), error="timeout"),
            Step("bad", Action(ActionKind.SEARCH, "q2"), error="timeout"),
            Step("bad", Action(ActionKind.SEARCH, "q3"), error="timeout"),
        ]
        self.assertEqual(fatal_step_index(steps), 1)

    def test_contributions_are_normalized_to_mean_one(self):
        steps = [
            Step("a", Action(ActionKind.SEARCH, "a"), contribution=1.0),
            Step("b", Action(ActionKind.SEARCH, "b"), contribution=3.0),
        ]
        values = weighted_advantages(2.0, steps)
        self.assertEqual(values, [1.0, 3.0])
        self.assertEqual(sum(values), 4.0)
