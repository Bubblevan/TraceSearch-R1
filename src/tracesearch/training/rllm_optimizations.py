"""Small, phase-scoped runtime optimizations for the real M1-C backend.

The optional rLLM/veRL stack is intentionally not imported at module import
time.  This keeps the native Windows/core test environment independent from
the Linux training environment.
"""

from __future__ import annotations

import functools
import inspect
from types import ModuleType
from typing import Any


def install_c0_post_batch_weight_sync_skip(backend_module: ModuleType, *, phase: str = "c0") -> str:
    """Skip only the redundant post-batch weight sync used by M1-C C0.

    C0 configures critic warmup so no actor optimizer step is executed.  The
    initial train-start sync is still left intact; only the later
    ``on_batch_end`` call is guarded.  C1/C2 never call this function.

    The guard wraps the pinned rLLM ``VerlBackend.on_batch_end`` method rather
    than editing the installed third-party package.  It temporarily replaces
    that hook's checkpoint-manager method with an async no-op, then restores
    it even when the original hook raises.
    """

    if phase != "c0":
        raise RuntimeError(f"C0 weight-sync optimization is phase-scoped; received phase={phase!r}")
    backend_type = getattr(backend_module, "VerlBackend", None)
    if backend_type is None:
        raise RuntimeError(
            "The pinned rLLM backend does not expose VerlBackend; "
            "refusing to apply the C0 weight-sync optimization."
        )

    original = getattr(backend_type, "on_batch_end", None)
    if original is None or not callable(original):
        raise RuntimeError(
            "The pinned rLLM VerlBackend does not expose on_batch_end; "
            "refusing to apply the C0 weight-sync optimization."
        )
    if getattr(original, "_tracesearch_c0_sync_guard", False):
        return "already_installed"

    @functools.wraps(original)
    async def guarded_on_batch_end(self: Any, *args: Any, **kwargs: Any) -> Any:
        trainer_config = getattr(self, "config", None)
        trainer = getattr(trainer_config, "trainer", None)
        warmup = getattr(trainer, "critic_warmup", None)
        global_step = getattr(self, "global_steps", None)
        if trainer is not None and (warmup is None or global_step is None or int(warmup) <= int(global_step)):
            return await _resolve_maybe_awaitable(original(self, *args, **kwargs))
        manager = getattr(self, "checkpoint_manager", None)
        update_weights = getattr(manager, "update_weights", None)
        if manager is None or update_weights is None:
            return await _resolve_maybe_awaitable(original(self, *args, **kwargs))

        async def skip_update_weights(*_args: Any, **_kwargs: Any) -> None:
            return None

        manager.update_weights = skip_update_weights
        try:
            return await _resolve_maybe_awaitable(original(self, *args, **kwargs))
        finally:
            manager.update_weights = update_weights

    guarded_on_batch_end._tracesearch_c0_sync_guard = True  # type: ignore[attr-defined]
    backend_type.on_batch_end = guarded_on_batch_end
    return "installed"


def install_c0_initial_weight_sync_skip(
    backend_module: ModuleType,
    *,
    backend_instance: Any | None = None,
    phase: str = "c0",
) -> str:
    """Skip C0's redundant train-start sync for a preloaded rollout engine.

    The SGLang hybrid server loads the exact same base checkpoint as the FSDP
    actor.  C0 performs no optimizer update, so copying CUDA tensors from the
    actor into the colocated server is unnecessary and can fail on a single
    WSL GPU while crossing the server's CUDA IPC boundary.  The wrapper keeps
    checkpoint loading and trainer-state initialization intact and suppresses
    only the ``checkpoint_manager.update_weights`` call.  SGLang's colocated
    C0 configuration keeps its cache resident, because veRL's generic
    ``wake_up`` path is not supported for SGLang HYBRID mode.
    """

    if phase != "c0":
        raise RuntimeError(f"C0 initial weight-sync optimization is phase-scoped; received phase={phase!r}")
    backend_type = getattr(backend_module, "VerlBackend", None)
    if backend_type is None:
        raise RuntimeError(
            "The pinned rLLM backend does not expose VerlBackend; "
            "refusing to apply the C0 initial weight-sync optimization."
        )

    target = backend_instance if backend_instance is not None else backend_type
    original = getattr(target, "on_train_start", None)
    if original is None or not callable(original):
        raise RuntimeError(
            "The pinned rLLM VerlBackend does not expose on_train_start; "
            "refusing to apply the C0 initial weight-sync optimization."
        )
    if getattr(original, "_tracesearch_c0_initial_sync_guard", False):
        return "already_installed"

    load_checkpoint = getattr(backend_module, "load_checkpoint", None)
    if not callable(load_checkpoint):
        raise RuntimeError(
            "The pinned rLLM backend does not expose load_checkpoint; "
            "refusing to bypass train-start weight sync without preserving "
            "checkpoint state initialization."
        )

    @functools.wraps(original)
    async def guarded_on_train_start(*args: Any, **kwargs: Any) -> Any:
        # Keep the stateful part of VerlBackend.on_train_start, but do not call
        # CheckpointEngineManager.update_weights for a preloaded C0 SGLang
        # server.  This avoids CUDA IPC deserialization across the colocated
        # FSDP/server processes on a single WSL GPU.
        del kwargs
        if backend_instance is None:
            self, trainer_state = args
        else:
            self = backend_instance
            (trainer_state,) = args
        self.global_steps = trainer_state.global_step
        self.global_steps = load_checkpoint(
            self.config,
            self.actor_rollout_wg,
            train_dataloader=trainer_state.train_dataloader,
        )
        trainer_state.global_step = self.global_steps
        trainer_state.epoch = (
            trainer_state.train_dataloader.epoch
            if trainer_state.train_dataloader is not None
            else 0
        )

    guarded_on_train_start._tracesearch_c0_initial_sync_guard = True  # type: ignore[attr-defined]
    if backend_instance is not None:
        backend_instance.on_train_start = guarded_on_train_start
    else:
        backend_type.on_train_start = guarded_on_train_start
    return "installed"


async def _resolve_maybe_awaitable(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
