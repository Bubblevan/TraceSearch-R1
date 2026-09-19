"""Vendor-neutral asynchronous model clients for M1."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMGenerationConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    stop_sequences: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True)
class ModelGeneration:
    text: str
    token_ids: tuple[int, ...] | None = None
    logprobs: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelClient(Protocol):
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        config: LLMGenerationConfig,
    ) -> ModelGeneration: ...


class ModelClientError(RuntimeError):
    """Raised when an OpenAI-compatible gateway cannot produce a generation."""


class OpenAICompatibleClient:
    """Small stdlib client for OpenAI-compatible chat-completions endpoints."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must be non-empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.extra_headers = dict(extra_headers or {})

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        config: LLMGenerationConfig,
    ) -> ModelGeneration:
        return await asyncio.to_thread(self._generate_sync, messages, model, config)

    def _generate_sync(
        self,
        messages: list[dict[str, str]],
        model: str,
        config: LLMGenerationConfig,
    ) -> ModelGeneration:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
        }
        if config.stop_sequences:
            payload["stop"] = list(config.stop_sequences)
        request = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
                **self.extra_headers,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise ModelClientError(f"model gateway HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise ModelClientError(f"model gateway request failed: {exc}") from exc
        return self._parse_generation(body)

    @staticmethod
    def _parse_generation(body: Any) -> ModelGeneration:
        if not isinstance(body, dict) or not isinstance(body.get("choices"), list) or not body["choices"]:
            raise ModelClientError("model gateway response has no choices")
        choice = body["choices"][0]
        if not isinstance(choice, dict):
            raise ModelClientError("model gateway choice is not an object")
        message = choice.get("message") or {}
        text = message.get("content", choice.get("text")) if isinstance(message, dict) else None
        if not isinstance(text, str):
            raise ModelClientError("model gateway choice has no text content")
        raw_token_ids = (
            choice.get("token_ids")
            or (message.get("token_ids") if isinstance(message, dict) else None)
            or body.get("token_ids")
        )
        token_ids = None
        if raw_token_ids is not None:
            if not isinstance(raw_token_ids, list) or not all(isinstance(item, int) for item in raw_token_ids):
                raise ModelClientError("model gateway token_ids must be a list of integers")
            token_ids = tuple(raw_token_ids)
        raw_logprobs = choice.get("logprobs")
        logprobs: tuple[float, ...] | None = None
        if isinstance(raw_logprobs, list) and all(isinstance(item, (int, float)) for item in raw_logprobs):
            logprobs = tuple(float(item) for item in raw_logprobs)
        return ModelGeneration(
            text=text,
            token_ids=token_ids,
            logprobs=logprobs,
            metadata={"usage": body.get("usage"), "id": body.get("id")},
        )
