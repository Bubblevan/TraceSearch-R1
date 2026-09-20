"""Vendor-neutral asynchronous model clients for M1."""

from __future__ import annotations

import asyncio
import hashlib
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
    sampling_seed: int | None = None
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
        if self.sampling_seed is not None and self.sampling_seed < 0:
            raise ValueError("sampling_seed must be non-negative when set")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True)
class ModelGeneration:
    text: str
    token_ids: tuple[int, ...] | None = None
    logprobs: tuple[float, ...] | None = None
    prompt_token_ids: tuple[int, ...] | None = None
    sampling_seed: int | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def derive_sampling_seed(
    task_id: str,
    rollout_id: str,
    sample_index: int,
    step_index: int,
    *,
    base_seed: int | None = None,
) -> int:
    """Derive a stable per-generation seed from rollout identity."""

    material = "|".join(
        (
            str(base_seed if base_seed is not None else 0),
            task_id,
            rollout_id,
            str(sample_index),
            str(step_index),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") & 0x7FFFFFFF


def gather_response_logprobs(
    logits: Any,
    prompt_length: int,
    response_token_ids: tuple[int, ...] | list[int],
) -> tuple[float, ...]:
    """Gather causal log-probabilities for supplied response IDs only."""

    if getattr(logits, "ndim", None) != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if prompt_length < 1:
        raise ValueError("prompt_length must be positive")
    response = tuple(int(token) for token in response_token_ids)
    if not response:
        return ()
    start = prompt_length - 1
    end = start + len(response)
    if end > logits.shape[1] - 1:
        raise ValueError("logits do not contain one prediction for every response token")
    import torch

    response_logits = logits[:, start:end, :].float()
    target = torch.tensor([response], dtype=torch.long, device=logits.device).unsqueeze(-1)
    gathered = response_logits.log_softmax(dim=-1).gather(-1, target).squeeze(-1)[0]
    return tuple(float(value) for value in gathered.detach().cpu().tolist())


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
        raw_prompt_ids = (
            choice.get("prompt_token_ids")
            or (message.get("prompt_token_ids") if isinstance(message, dict) else None)
            or body.get("prompt_token_ids")
        )
        prompt_token_ids = None
        if raw_prompt_ids is not None:
            if not isinstance(raw_prompt_ids, list) or not all(isinstance(item, int) for item in raw_prompt_ids):
                raise ModelClientError("model gateway prompt_token_ids must be a list of integers")
            prompt_token_ids = tuple(raw_prompt_ids)
        return ModelGeneration(
            text=text,
            token_ids=token_ids,
            logprobs=logprobs,
            prompt_token_ids=prompt_token_ids,
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
        template = getattr(self.processor, "chat_template", None)
        self.tokenizer_version = getattr(self.processor, "name_or_path", None) or "local-tokenizer"
        self.template_version = hashlib.sha256(str(template).encode("utf-8")).hexdigest()[:16]

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
        if self.text_only:
            return self._generate_text_only(messages, text, inputs, model_device, config)
        do_sample = config.temperature > 0
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": config.max_tokens,
            "do_sample": do_sample,
            "top_p": config.top_p,
            "repetition_penalty": config.repetition_penalty,
        }
        if do_sample and config.top_k is not None:
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
            prompt_token_ids=tuple(int(token) for token in inputs["input_ids"][0].tolist()),
            sampling_seed=config.sampling_seed,
            finish_reason="length",
            metadata=self._generation_metadata(config, model_device, prompt_length, len(token_ids)),
        )

    def _generate_text_only(
        self,
        messages: list[dict[str, str]],
        text: str,
        inputs: dict[str, Any],
        model_device: Any,
        config: LLMGenerationConfig,
    ) -> ModelGeneration:
        """Generate with a per-call torch.Generator for decoder-only text models."""

        torch = self._torch
        prompt_ids = tuple(int(token) for token in inputs["input_ids"][0].tolist())
        actual_seed = config.sampling_seed if config.sampling_seed is not None else 0
        generator = torch.Generator(device=model_device)
        generator.manual_seed(actual_seed)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs["input_ids"])
        current_ids = inputs["input_ids"]
        past_key_values = None
        response_ids: list[int] = []
        eos_token_id = getattr(self.processor, "eos_token_id", None)
        for _ in range(config.max_tokens):
            model_inputs: dict[str, Any] = {
                "input_ids": current_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
            }
            if past_key_values is not None:
                model_inputs["past_key_values"] = past_key_values
            outputs = self.model(**model_inputs)
            logits = outputs.logits[:, -1, :].float()
            logits = self._apply_repetition_penalty(logits, response_ids, config.repetition_penalty)
            if config.temperature > 0:
                logits = logits / config.temperature
                logits = self._apply_top_k(logits, config.top_k)
                logits = self._apply_top_p(logits, config.top_p)
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1, generator=generator)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token = int(next_token.item())
            response_ids.append(token)
            past_key_values = getattr(outputs, "past_key_values", None)
            current_ids = next_token
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=model_device)],
                dim=1,
            )
            if eos_token_id is not None and token == eos_token_id:
                break
        response_tuple = tuple(response_ids)
        decoded = self.processor.batch_decode([response_tuple], skip_special_tokens=True)[0]
        finish_reason = "eos_token" if response_ids and response_ids[-1] == eos_token_id else "length"
        response_logprobs = self.score_response_logprobs(prompt_ids, response_tuple)
        return ModelGeneration(
            text=decoded,
            token_ids=response_tuple,
            logprobs=response_logprobs,
            prompt_token_ids=prompt_ids,
            sampling_seed=actual_seed,
            finish_reason=finish_reason,
            metadata=self._generation_metadata(config, model_device, len(prompt_ids), len(response_ids)),
        )

    def score_response_logprobs(
        self,
        prompt_token_ids: tuple[int, ...] | list[int],
        response_token_ids: tuple[int, ...] | list[int],
    ) -> tuple[float, ...]:
        """Score the supplied response IDs without decoding or sampling them."""

        prompt = tuple(int(token) for token in prompt_token_ids)
        response = tuple(int(token) for token in response_token_ids)
        if not prompt:
            raise ValueError("prompt_token_ids must not be empty")
        if not response:
            return ()
        torch = self._torch
        model_device = next(self.model.parameters()).device
        full_ids = torch.tensor([prompt + response], dtype=torch.long, device=model_device)
        attention_mask = torch.ones_like(full_ids)
        with torch.no_grad():
            logits = self.model(input_ids=full_ids, attention_mask=attention_mask, use_cache=False).logits
        return gather_response_logprobs(logits, len(prompt), response)

    def tokenize_text(self, text: str) -> tuple[int, ...]:
        """Tokenize an environment-only string for a zero-loss observation span."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        encoded = self.processor(text=[text], return_tensors="pt")
        return tuple(int(token) for token in encoded["input_ids"][0].tolist())

    @staticmethod
    def _apply_repetition_penalty(logits: Any, response_ids: list[int], penalty: float) -> Any:
        if penalty == 1.0 or not response_ids:
            return logits
        import torch

        values = logits.clone()
        indices = list(dict.fromkeys(response_ids))
        selected = values[:, indices]
        values[:, indices] = torch.where(selected < 0, selected * penalty, selected / penalty)
        return values

    @staticmethod
    def _apply_top_k(logits: Any, top_k: int | None) -> Any:
        if top_k is None or top_k >= logits.shape[-1]:
            return logits
        values, _ = logits.topk(top_k, dim=-1)
        return logits.masked_fill(logits < values[:, [-1]], float("-inf"))

    @staticmethod
    def _apply_top_p(logits: Any, top_p: float) -> Any:
        if top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = logits.sort(descending=True, dim=-1)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        filtered = logits.clone()
        filtered.scatter_(1, sorted_indices, sorted_logits.masked_fill(remove, float("-inf")))
        return filtered

    def _generation_metadata(self, config: LLMGenerationConfig, model_device: Any, prompt_count: int, response_count: int) -> dict[str, Any]:
        return {
            "backend": "transformers",
            "model_path": self.model_path,
            "checkpoint": self.model_path,
            "device": str(model_device),
            "prompt_token_count": int(prompt_count),
            "response_token_count": int(response_count),
            "tokenizer_version": self.tokenizer_version,
            "template_version": self.template_version,
            "sampling_seed": config.sampling_seed,
            "sampling_config": {
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "repetition_penalty": config.repetition_penalty,
                "enable_thinking": config.enable_thinking,
            },
        }
