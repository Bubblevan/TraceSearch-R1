"""Aligned offline search and visit environment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from tracesearch.data.schema import Document, Evidence, ToolErrorType, ToolResult
from tracesearch.environment.bm25 import BM25Index
from tracesearch.environment.corpus import Corpus
from tracesearch.environment.faults import FaultSchedule, FaultType, FailureInjector


@dataclass
class LocalSearchEnvironment:
    corpus: Corpus
    top_k: int = 5
    fault_schedule: FaultSchedule | None = None
    failure_injector: FailureInjector | None = None
    configured_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.configured_latency_ms < 0:
            raise ValueError("configured_latency_ms must be non-negative")
        self.index = BM25Index(self.corpus.documents)

    async def search(self, query: str, *, step_index: int = 0, task_id: str | None = None) -> ToolResult:
        started = time.perf_counter()
        fault = self._fault("search", step_index, task_id)
        injected = self._injected_result("search", query, fault, started)
        if injected is not None:
            return injected
        if not isinstance(query, str) or not query.strip():
            return self._result(
                tool_name="search",
                request=query,
                ok=False,
                error_type=ToolErrorType.INVALID_ARGUMENT,
                error_message="query must be non-empty",
                started=started,
            )
        evidence = self.index.search(query, top_k=self.top_k)
        if not evidence:
            return self._result(
                tool_name="search",
                request=query,
                ok=False,
                error_type=ToolErrorType.EMPTY_RESULT,
                error_message="no documents matched the query",
                started=started,
            )
        text = "\n".join(f"{item.rank}. {item.title}: {item.snippet}" for item in evidence)
        return self._result(
            tool_name="search",
            request=query,
            ok=True,
            text=text,
            evidence=evidence,
            started=started,
        )

    async def visit(self, doc_id: str, *, step_index: int = 0, task_id: str | None = None) -> ToolResult:
        started = time.perf_counter()
        fault = self._fault("visit", step_index, task_id)
        injected = self._injected_result("visit", doc_id, fault, started)
        if injected is not None:
            return injected
        if not isinstance(doc_id, str) or not doc_id.strip():
            return self._result(
                tool_name="visit",
                request=doc_id,
                ok=False,
                error_type=ToolErrorType.INVALID_ARGUMENT,
                error_message="doc_id must be non-empty",
                started=started,
            )
        document = self.corpus.get(doc_id)
        if document is None:
            return self._result(
                tool_name="visit",
                request=doc_id,
                ok=False,
                error_type=ToolErrorType.NOT_FOUND,
                error_message=f"unknown doc_id: {doc_id}",
                started=started,
            )
        evidence = [
            Evidence(
                doc_id=document.doc_id,
                title=document.title,
                snippet=document.text,
                score=1.0,
                rank=1,
                url=document.url,
                metadata={"visited": True},
            )
        ]
        return self._result(
            tool_name="visit",
            request=doc_id,
            ok=True,
            text=document.text,
            evidence=evidence,
            metadata={"document": document.to_dict()},
            started=started,
        )

    def _fault(self, tool_name: str, step_index: int, task_id: str | None) -> FaultType | None:
        scheduled = self.fault_schedule.get_fault(step_index=step_index, tool_name=tool_name, task_id=task_id) if self.fault_schedule else None
        return scheduled or (self.failure_injector.next_fault(tool_name, step_index) if self.failure_injector else None)

    def _injected_result(self, tool_name: str, request: Any, fault: FaultType | None, started: float) -> ToolResult | None:
        if fault is None:
            return None
        metadata = {"injected_fault": fault.value}
        if fault is FaultType.IRRELEVANT_RESULT:
            distractors = self.corpus.documents[: self.top_k]
            evidence = [
                Evidence(
                    doc_id=document.doc_id,
                    title=document.title,
                    snippet=document.text[:240],
                    score=0.0,
                    rank=rank,
                    url=document.url,
                    metadata={"semantic_corruption": True},
                )
                for rank, document in enumerate(distractors, 1)
            ]
            return self._result(
                tool_name=tool_name,
                request=request,
                ok=True,
                text="Injected irrelevant evidence",
                evidence=evidence,
                metadata=metadata,
                started=started,
            )
        error_type = {
            FaultType.TIMEOUT: ToolErrorType.TIMEOUT,
            FaultType.TRANSIENT: ToolErrorType.TRANSIENT,
            FaultType.EMPTY_RESULT: ToolErrorType.EMPTY_RESULT,
            FaultType.MALFORMED_RESULT: ToolErrorType.MALFORMED_RESULT,
            FaultType.EXCEPTION: ToolErrorType.INJECTED_FAILURE,
            FaultType.INJECTED_FAILURE: ToolErrorType.INJECTED_FAILURE,
        }[fault]
        return self._result(
            tool_name=tool_name,
            request=request,
            ok=False,
            error_type=error_type,
            error_message=f"injected {fault.value}",
            metadata=metadata,
            started=started,
        )

    def _result(self, *, tool_name: str, request: Any, ok: bool, started: float, text: str = "", evidence: list[Evidence] | None = None, error_type: ToolErrorType | None = None, error_message: str | None = None, metadata: dict[str, Any] | None = None) -> ToolResult:
        measured = (time.perf_counter() - started) * 1000.0
        latency = self.configured_latency_ms + measured
        return ToolResult(
            tool_name=tool_name,
            ok=ok,
            request=request,
            text=text,
            evidence=evidence or [],
            error_type=error_type,
            error_message=error_message,
            latency_ms=latency,
            metadata=metadata or {},
        )


@dataclass
class LocalSearchTool:
    environment: LocalSearchEnvironment

    async def search(self, query: str, **context: object) -> ToolResult:
        return await self.environment.search(query, **context)


@dataclass
class LocalVisitTool:
    environment: LocalSearchEnvironment

    async def visit(self, doc_id: str, **context: object) -> ToolResult:
        return await self.environment.visit(doc_id, **context)
