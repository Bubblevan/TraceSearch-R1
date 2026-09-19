"""Strict textual action protocol for learned search policies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from tracesearch.data.schema import Action, ActionKind


class ParseFailureType(StrEnum):
    NO_ACTION = "no_action"
    MULTIPLE_ACTIONS = "multiple_actions"
    MALFORMED_TAG = "malformed_tag"
    UNTERMINATED_TAG = "unterminated_tag"
    EMPTY_PAYLOAD = "empty_payload"


class ActionParseError(ValueError):
    def __init__(self, failure_type: ParseFailureType, message: str) -> None:
        super().__init__(message)
        self.failure_type = failure_type


@dataclass(frozen=True)
class ParsedPolicyOutput:
    raw_text: str
    reasoning: str
    action: Action
    action_span: tuple[int, int]


_ACTION_NAMES = "search|visit|answer"
_ACTION_BLOCK = re.compile(
    rf"<(?P<kind>{_ACTION_NAMES})>(?P<payload>.*?)</(?P=kind)>",
    re.IGNORECASE | re.DOTALL,
)
_OPEN_ACTION = re.compile(rf"<(?P<kind>{_ACTION_NAMES})\b[^>]*>", re.IGNORECASE)
_CLOSE_ACTION = re.compile(rf"</(?P<kind>{_ACTION_NAMES})\s*>", re.IGNORECASE)
_TAG = re.compile(r"</?([A-Za-z][A-Za-z0-9_-]*)(?:\s[^>]*)?>", re.DOTALL)


def parse_policy_output(text: str) -> ParsedPolicyOutput:
    if not isinstance(text, str) or not text.strip():
        raise ActionParseError(ParseFailureType.NO_ACTION, "model output is empty")
    for tag in _TAG.finditer(text):
        name = tag.group(1).casefold()
        if name not in {"think", "search", "visit", "answer"}:
            raise ActionParseError(ParseFailureType.MALFORMED_TAG, f"unsupported tag <{name}>")
        if not re.fullmatch(r"</?(?:think|search|visit|answer)>", tag.group(0), re.IGNORECASE):
            raise ActionParseError(ParseFailureType.MALFORMED_TAG, f"malformed tag {tag.group(0)!r}")
    lowered = text.casefold()
    for name in ("think", "search", "visit", "answer"):
        if lowered.count(f"<{name}>") != lowered.count(f"</{name}>"):
            raise ActionParseError(ParseFailureType.UNTERMINATED_TAG, f"unterminated <{name}> tag")
    actions = list(_ACTION_BLOCK.finditer(text))
    openings = list(_OPEN_ACTION.finditer(text))
    closings = list(_CLOSE_ACTION.finditer(text))
    if len(openings) != len(actions) or len(closings) != len(actions):
        if openings or closings:
            raise ActionParseError(ParseFailureType.UNTERMINATED_TAG, "action tag is not properly terminated")
        raise ActionParseError(ParseFailureType.NO_ACTION, "model output contains no action")
    if len(actions) != 1:
        raise ActionParseError(ParseFailureType.MULTIPLE_ACTIONS, "model output must contain exactly one action")
    match = actions[0]
    if text[match.end():].strip():
        raise ActionParseError(ParseFailureType.MALFORMED_TAG, "terminal action must be the final output")
    payload = match.group("payload").strip()
    if not payload:
        raise ActionParseError(ParseFailureType.EMPTY_PAYLOAD, "action payload is empty")
    think_matches = list(re.finditer(r"<think>(?P<reasoning>.*?)</think>", text, re.IGNORECASE | re.DOTALL))
    if len(think_matches) > 1:
        raise ActionParseError(ParseFailureType.MALFORMED_TAG, "model output contains multiple think blocks")
    reasoning = think_matches[0].group("reasoning").strip() if think_matches else text[: match.start()].strip()
    return ParsedPolicyOutput(
        raw_text=text,
        reasoning=reasoning,
        action=Action(ActionKind(match.group("kind").casefold()), payload),
        action_span=(match.start(), match.end()),
    )


parse_action_output = parse_policy_output
