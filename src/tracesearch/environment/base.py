"""Protocols shared by asynchronous environments and agent adapters."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from tracesearch.data.schema import ToolResult


class AsyncSearchEnvironment(Protocol):
    async def search(self, query: str, **context: object) -> ToolResult: ...

    async def visit(self, doc_id: str, **context: object) -> ToolResult: ...
