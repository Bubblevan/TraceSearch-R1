import asyncio

from tracesearch.data.schema import Document, ToolErrorType
from tracesearch.environment import Corpus, FaultSchedule, FaultType, FailureInjector, LocalSearchEnvironment


def test_fault_schedule_only_affects_declared_step():
    async def run():
        env = LocalSearchEnvironment(
            Corpus([Document("d", "D", "fact")]),
            fault_schedule=FaultSchedule({(2, "search"): "timeout"}),
        )
        return await env.search("fact", step_index=1), await env.search("fact", step_index=2), await env.search("fact", step_index=3)

    first, failed, third = asyncio.run(run())
    assert first.ok and not failed.ok and failed.error_type is ToolErrorType.TIMEOUT and third.ok


def test_seeded_injector_reproduces_and_semantic_corruption_is_not_failure():
    left = FailureInjector(seed=11, irrelevant_result_rate=0.5)
    right = FailureInjector(seed=11, irrelevant_result_rate=0.5)
    assert [left.next_fault("search", i) for i in range(8)] == [right.next_fault("search", i) for i in range(8)]

    async def run():
        env = LocalSearchEnvironment(
            Corpus([Document("d", "D", "fact")]),
            fault_schedule=FaultSchedule({(0, "search"): FaultType.IRRELEVANT_RESULT}),
        )
        return await env.search("fact", step_index=0)

    result = asyncio.run(run())
    assert result.ok and result.error_type is None
    assert result.metadata["injected_fault"] == "irrelevant_result"
