from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SearchTool(Protocol):
    def search(self, query: str) -> str: ...


class VisitTool(Protocol):
    def visit(self, url: str) -> str: ...


@dataclass
class StaticSearchTool:
    """Offline fixture for deterministic development and unit tests."""

    results: dict[str, str]

    def search(self, query: str) -> str:
        return self.results.get(query, "No results.")


@dataclass
class StaticVisitTool:
    """Offline page fixture. Production adapters implement the same interface."""

    pages: dict[str, str]

    def visit(self, url: str) -> str:
        return self.pages.get(url, "Page not found.")
