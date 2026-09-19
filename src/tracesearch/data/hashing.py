"""Stable hashes for proving which offline data snapshot a run used."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from tracesearch.data.schema import Document


def canonical_json(value: object) -> str:
    payload = value.to_dict() if hasattr(value, "to_dict") else value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_corpus(documents: Iterable[Document]) -> str:
    """Hash documents in canonical doc_id order, independent of input order."""

    ordered = sorted((document.to_dict() for document in documents), key=lambda item: item["doc_id"])
    digest = hashlib.sha256()
    for document in ordered:
        digest.update(canonical_json(document).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
