"""Deterministic in-memory corpus used by the offline environment."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from tracesearch.data.io import load_corpus
from tracesearch.data.schema import Document


class Corpus:
    def __init__(self, documents: Iterable[Document]):
        values = list(documents)
        ids = [document.doc_id for document in values]
        if len(ids) != len(set(ids)):
            raise ValueError("corpus doc_id values must be unique")
        self._documents = {document.doc_id: document for document in values}

    @classmethod
    def from_jsonl(cls, path: str) -> "Corpus":
        return cls(load_corpus(path))

    def get(self, doc_id: str) -> Document | None:
        return self._documents.get(doc_id)

    def require(self, doc_id: str) -> Document:
        document = self.get(doc_id)
        if document is None:
            raise KeyError(doc_id)
        return document

    def __len__(self) -> int:
        return len(self._documents)

    def __iter__(self) -> Iterator[Document]:
        for doc_id in sorted(self._documents):
            yield self._documents[doc_id]

    @property
    def documents(self) -> list[Document]:
        return list(self)
