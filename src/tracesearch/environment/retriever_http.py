"""Generic HTTP retriever/visit adapter for real M1 experiments."""

from __future__ import annotations

import asyncio
import json
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from tracesearch.data.schema import Evidence, ToolErrorType, ToolResult


@dataclass
class HTTPRetriever:
    """Call an external retriever without coupling TraceSearch to its API."""

    base_url: str
    search_path: str = "/search"
    visit_path: str = "/visit/{doc_id}"
    top_k: int = 5
    timeout_s: float = 10.0
    backend_name: str = "http_retriever"
    backend_version: str = "unspecified"
    headers: dict[str, str] = field(default_factory=dict)
    retries: int = 0

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ValueError("base_url must be non-empty")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.retries < 0:
            raise ValueError("retries must be non-negative")
        self.base_url = self.base_url.rstrip("/")

    async def search(
        self,
        query: str,
        *,
        step_index: int = 0,
        task_id: str | None = None,
        rollout_id: str | None = None,
    ) -> ToolResult:
        if not isinstance(query, str) or not query.strip():
            return self._failure(
                "search",
                query,
                ToolErrorType.INVALID_ARGUMENT,
                "query must be non-empty",
            )
        started = time.perf_counter()
        try:
            body = await asyncio.to_thread(self._request_json, self.search_path, {"query": query, "top_k": self.top_k})
            evidence = self._parse_candidates(body)
        except Exception as exc:
            return self._exception_result("search", query, exc, started)
        text = "\n".join(f"{item.rank}. {item.title}: {item.snippet}" for item in evidence)
        return self._success(
            "search",
            query,
            text,
            evidence,
            started,
            metadata={"backend": self.backend_name, "backend_version": self.backend_version},
        )

    async def visit(
        self,
        doc_id: str,
        *,
        step_index: int = 0,
        task_id: str | None = None,
        rollout_id: str | None = None,
    ) -> ToolResult:
        if not isinstance(doc_id, str) or not doc_id.strip():
            return self._failure(
                "visit",
                doc_id,
                ToolErrorType.INVALID_ARGUMENT,
                "doc_id must be non-empty",
            )
        started = time.perf_counter()
        path = self.visit_path.format(doc_id=urllib.parse.quote(doc_id, safe=""))
        try:
            body = await asyncio.to_thread(self._request_json, path, {"doc_id": doc_id})
            document = self._parse_document(body, requested_id=doc_id)
            evidence = [
                Evidence(
                    doc_id=document["doc_id"],
                    title=document["title"],
                    snippet=document["text"],
                    score=1.0,
                    rank=1,
                    url=document.get("url"),
                    metadata={"visited": True},
                )
            ]
        except Exception as exc:
            return self._exception_result("visit", doc_id, exc, started)
        return self._success(
            "visit",
            doc_id,
            document["text"],
            evidence,
            started,
            metadata={
                "backend": self.backend_name,
                "backend_version": self.backend_version,
                "document": document,
            },
        )

    def _request_json(self, path: str, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            f"{self.base_url}{path if path.startswith('/') else '/' + path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **self.headers},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise _HTTPRetrieverError(ToolErrorType.NOT_FOUND, f"HTTP 404 for {path}") from exc
                if exc.code >= 500 and attempt < self.retries:
                    last_error = exc
                    continue
                error_type = ToolErrorType.TRANSIENT if exc.code >= 500 else ToolErrorType.MALFORMED_RESULT
                raise _HTTPRetrieverError(error_type, f"HTTP {exc.code} for {path}") from exc
            except (socket.timeout, TimeoutError) as exc:
                if attempt < self.retries:
                    last_error = exc
                    continue
                raise _HTTPRetrieverError(ToolErrorType.TIMEOUT, f"timeout calling {path}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.retries:
                    last_error = exc
                    continue
                raise _HTTPRetrieverError(ToolErrorType.TRANSIENT, f"retriever connection failed: {exc}") from exc
            except json.JSONDecodeError as exc:
                raise _HTTPRetrieverError(ToolErrorType.MALFORMED_RESULT, "retriever response is not JSON") from exc
        raise _HTTPRetrieverError(ToolErrorType.TRANSIENT, f"retriever request failed: {last_error}")

    @staticmethod
    def _parse_candidates(body: Any) -> list[Evidence]:
        if not isinstance(body, dict):
            raise _HTTPRetrieverError(ToolErrorType.MALFORMED_RESULT, "search response must be an object")
        candidates = body.get("results", body.get("candidates", body.get("evidence")))
        if not isinstance(candidates, list):
            raise _HTTPRetrieverError(ToolErrorType.MALFORMED_RESULT, "search response must contain a results list")
        evidence: list[Evidence] = []
        for index, candidate in enumerate(candidates, 1):
            if not isinstance(candidate, dict):
                raise _HTTPRetrieverError(ToolErrorType.MALFORMED_RESULT, "search candidate must be an object")
            doc_id = candidate.get("doc_id", candidate.get("id"))
            title = candidate.get("title")
            snippet = candidate.get("snippet", candidate.get("text", ""))
            if not all(isinstance(value, str) and value.strip() for value in (doc_id, title)):
                raise _HTTPRetrieverError(ToolErrorType.MALFORMED_RESULT, "candidate requires doc_id and title")
            if not isinstance(snippet, str):
                raise _HTTPRetrieverError(ToolErrorType.MALFORMED_RESULT, "candidate snippet must be text")
            evidence.append(
                Evidence(
                    doc_id=doc_id,
                    title=title,
                    snippet=snippet,
                    score=float(candidate.get("score", 0.0)),
                    rank=int(candidate.get("rank", index)),
                    url=candidate.get("url"),
                    metadata=dict(candidate.get("metadata", {})),
                )
            )
        return evidence

    @staticmethod
    def _parse_document(body: Any, *, requested_id: str) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise _HTTPRetrieverError(ToolErrorType.MALFORMED_RESULT, "visit response must be an object")
        document = body.get("document", body)
        if not isinstance(document, dict):
            raise _HTTPRetrieverError(ToolErrorType.MALFORMED_RESULT, "visit response document must be an object")
        doc_id = document.get("doc_id", document.get("id", requested_id))
        title = document.get("title")
        text = document.get("text", document.get("content"))
        if not all(isinstance(value, str) and value.strip() for value in (doc_id, title, text)):
            raise _HTTPRetrieverError(ToolErrorType.MALFORMED_RESULT, "document requires doc_id, title, and text")
        return {
            "doc_id": doc_id,
            "title": title,
            "text": text,
            "url": document.get("url"),
            "metadata": dict(document.get("metadata", {})),
        }

    def _exception_result(self, tool_name: str, request: Any, exc: Exception, started: float) -> ToolResult:
        if isinstance(exc, _HTTPRetrieverError):
            error_type, message = exc.error_type, str(exc)
        else:
            error_type, message = ToolErrorType.INTERNAL, f"{type(exc).__name__}: {exc}"
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            request=request,
            error_type=error_type,
            error_message=message,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata={"backend": self.backend_name, "backend_version": self.backend_version},
        )

    def _success(
        self,
        tool_name: str,
        request: Any,
        text: str,
        evidence: list[Evidence],
        started: float,
        *,
        metadata: dict[str, Any],
    ) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            ok=True,
            request=request,
            text=text,
            evidence=evidence,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            metadata=metadata,
        )

    def _failure(self, tool_name: str, request: Any, error_type: ToolErrorType, message: str) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            ok=False,
            request=request,
            error_type=error_type,
            error_message=message,
            metadata={"backend": self.backend_name, "backend_version": self.backend_version},
        )


class _HTTPRetrieverError(RuntimeError):
    def __init__(self, error_type: ToolErrorType, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
