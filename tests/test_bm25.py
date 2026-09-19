from tracesearch.data.schema import Document
from tracesearch.environment import BM25Index


def test_bm25_relevance_top_k_and_deterministic_ties():
    documents = [
        Document("b", "Blue", "blue marsh keeper"),
        Document("a", "Blue", "blue marsh keeper"),
        Document("c", "Other", "red hill"),
    ]
    index = BM25Index(documents)
    results = index.search("blue marsh keeper", top_k=2)
    assert [item.doc_id for item in results] == ["a", "b"]
    assert len(index.search("blue", top_k=1)) == 1
