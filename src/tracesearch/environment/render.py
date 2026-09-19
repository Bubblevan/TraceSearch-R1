"""Model-facing rendering derived from structured tool results."""

from __future__ import annotations

from tracesearch.data.schema import ToolResult


def render_tool_result(result: ToolResult) -> str:
    if not result.ok:
        detail = result.error_message or result.error_type.value if result.error_type else "tool failure"
        return f"[{result.tool_name} failed: {detail}]"
    if result.evidence:
        rows = [f"{item.rank}. {item.title} ({item.doc_id}) — {item.snippet}" for item in result.evidence]
        return f"[{result.tool_name}]\n" + "\n".join(rows)
    return f"[{result.tool_name}]\n{result.text}".strip()
