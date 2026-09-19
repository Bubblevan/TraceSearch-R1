"""Small transparent deterministic BM25 implementation."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from tracesearch.data.schema import Document, Evidence


TOKEN_RE = re.compile(r"[\w]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Casefolded Unicode word tokens; punctuation is a separator."""

    return [token.casefold() for token in TOKEN_RE.findall(text)]


class BM25Index:
    """BM25Okapi-style ranker with deterministic ``(-score, doc_id)`` ties.

    The score is ``idf * tf * (k1 + 1) / (tf + k1 * (1-b+b*dl/avgdl))`` where
    ``idf = log(1 + (N-df+0.5)/(df+0.5))``.  Titles and body text are indexed
    together, and no stemming or stop-word removal is performed.
    """

    def __init__(self, documents: Iterable[Document], *, k1: float = 1.5, b: float = 0.75):
        self.documents = sorted(list(documents), key=lambda document: document.doc_id)
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self.k1 = k1
        self.b = b
        self._tokens = {
            document.doc_id: tokenize(f"{document.title} {document.text}") for document in self.documents
        }
        self._term_frequency = {doc_id: Counter(tokens) for doc_id, tokens in self._tokens.items()}
        self._lengths = {doc_id: len(tokens) for doc_id, tokens in self._tokens.items()}
        self._avgdl = sum(self._lengths.values()) / len(self.documents) if self.documents else 0.0
        self._df: Counter[str] = Counter()
        for tokens in self._tokens.values():
            self._df.update(set(tokens))

    def search(self, query: str, top_k: int = 5) -> list[Evidence]:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        query_terms = tokenize(query)
        if not query_terms or not self.documents or top_k == 0:
            return []
        scores: list[tuple[float, Document]] = []
        for document in self.documents:
            frequencies = self._term_frequency[document.doc_id]
            length = self._lengths[document.doc_id]
            score = 0.0
            for term in query_terms:
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                df = self._df[term]
                idf = math.log(1.0 + (len(self.documents) - df + 0.5) / (df + 0.5))
                denominator = tf + self.k1 * (1.0 - self.b + self.b * length / self._avgdl)
                score += idf * tf * (self.k1 + 1.0) / denominator
            if score > 0:
                scores.append((score, document))
        scores.sort(key=lambda pair: (-pair[0], pair[1].doc_id))
        return [
            Evidence(
                doc_id=document.doc_id,
                title=document.title,
                snippet=self._snippet(document, query_terms),
                score=score,
                rank=rank,
                url=document.url,
                metadata={"backend": "bm25"},
            )
            for rank, (score, document) in enumerate(scores[:top_k], 1)
        ]

    @staticmethod
    def _snippet(document: Document, query_terms: list[str], limit: int = 240) -> str:
        sentences = re.split(r"(?<=[.!?。！？])\s+", document.text.strip())
        wanted = set(query_terms)
        for sentence in sentences:
            if wanted.intersection(tokenize(sentence)):
                return sentence[:limit]
        return document.text[:limit]
