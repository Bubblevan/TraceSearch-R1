import asyncio

from tracesearch.data.schema import Document, ToolErrorType
from tracesearch.environment import Corpus, LocalSearchEnvironment


def test_search_and_visit_are_structured_and_unknown_visit_fails():
    async def run():
        environment = LocalSearchEnvironment(Corpus([Document("d-1", "Title", "A useful fact.")]))
        search = await environment.search("useful")
        visit = await environment.visit("d-1")
        missing = await environment.visit("missing")
        return search, visit, missing

    search, visit, missing = asyncio.run(run())
    assert search.ok and search.evidence[0].doc_id == "d-1"
    assert visit.ok and visit.metadata["document"]["doc_id"] == "d-1"
    assert not missing.ok and missing.error_type is ToolErrorType.NOT_FOUND
