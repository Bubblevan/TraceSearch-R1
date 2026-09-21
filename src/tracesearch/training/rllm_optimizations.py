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


def install_c0_post_batch_weight_sync_skip(backend_module: ModuleType) -> str:
    """Skip only the redundant post-batch weight sync used by M1-C C0.

    C0 configures critic warmup so no actor optimizer step is executed.  The
    initial train-start sync is still left intact; only the later
    ``on_batch_end`` call is guarded.  C1/C2 never call this function.

    The guard wraps the pinned rLLM ``VerlBackend.on_batch_end`` method rather
    than editing the installed third-party package.  It temporarily replaces
    that hook's checkpoint-manager method with an async no-op, then restores
    it even when the original hook raises.
    """

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


async def _resolve_maybe_awaitable(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
