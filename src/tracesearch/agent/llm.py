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
    top_k: int | None = 20
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    enable_thinking: bool = False
    max_tokens: int = 256
    stop_sequences: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("top_k must be positive when set")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
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
            "presence_penalty": config.presence_penalty,
            "repetition_penalty": config.repetition_penalty,
            "max_tokens": config.max_tokens,
        }
        if config.top_k is not None:
            payload["top_k"] = config.top_k
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


class TransformersModelClient:
    """Local Transformers client that returns the IDs sampled by ``generate``."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        local_files_only: bool = True,
    ) -> None:
        if not model_path.strip():
            raise ValueError("model_path must be non-empty")
        self.model_path = model_path
        self.device = device
        self.dtype_name = dtype
        self.local_files_only = local_files_only
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoModelForImageTextToText,
                AutoProcessor,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError("local model client requires torch and transformers") from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for the local model but is unavailable")
        dtype_value = getattr(torch, dtype, None)
        if dtype_value is None:
            raise ValueError(f"unsupported torch dtype: {dtype}")
        model_config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
        self.text_only = getattr(model_config, "model_type", "") == "qwen2"
        if self.text_only:
            self.processor = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=local_files_only,
            )
            model_class = AutoModelForCausalLM
        else:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=local_files_only,
            )
            model_class = AutoModelForImageTextToText
        self.model = model_class.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            dtype=dtype_value,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self._torch = torch

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        config: LLMGenerationConfig,
    ) -> ModelGeneration:
        return await asyncio.to_thread(self._generate_sync, messages, config)

    def _generate_sync(
        self,
        messages: list[dict[str, str]],
        config: LLMGenerationConfig,
    ) -> ModelGeneration:
        chat_template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if not self.text_only:
            chat_template_kwargs["enable_thinking"] = config.enable_thinking
        text = self.processor.apply_chat_template(messages, **chat_template_kwargs)
        inputs = self.processor(text=[text], return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        do_sample = config.temperature > 0
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": config.max_tokens,
            "do_sample": do_sample,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
        }
        if config.top_k is not None:
            generation_kwargs["top_k"] = config.top_k
        if do_sample:
            generation_kwargs["temperature"] = config.temperature
        if config.stop_sequences:
            generation_kwargs["stop_strings"] = list(config.stop_sequences)
        output = self.model.generate(**generation_kwargs)
        prompt_length = inputs["input_ids"].shape[1]
        token_ids = tuple(int(token) for token in output[0, prompt_length:].tolist())
        decoded = self.processor.batch_decode(output[:, prompt_length:], skip_special_tokens=True)[0]
        return ModelGeneration(
            text=decoded,
            token_ids=token_ids,
            metadata={
                "backend": "transformers",
                "model_path": self.model_path,
                "device": str(model_device),
                "prompt_token_count": int(prompt_length),
                "response_token_count": len(token_ids),
            },
        )
