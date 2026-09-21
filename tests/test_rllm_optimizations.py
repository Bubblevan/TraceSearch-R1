import asyncio
from types import SimpleNamespace

from tracesearch.training.rllm_optimizations import install_c0_post_batch_weight_sync_skip


def test_c0_guard_skips_batch_sync_and_restores_manager_method():
    class Manager:
        def __init__(self):
            self.calls = 0

        async def update_weights(self, step):
            self.calls += 1
            return step

    class Backend:
        async def on_batch_end(self, state):
            return await self.checkpoint_manager.update_weights(state)

    module = SimpleNamespace(VerlBackend=Backend)
    assert install_c0_post_batch_weight_sync_skip(module) == "installed"

    backend = Backend()
    manager = Manager()
    backend.checkpoint_manager = manager
    assert asyncio.run(backend.on_batch_end(1)) is None
    assert manager.calls == 0

    assert asyncio.run(manager.update_weights(2)) == 2
    assert manager.calls == 1


def test_c0_guard_is_idempotent():
    class Backend:
        async def on_batch_end(self):
            return None

    module = SimpleNamespace(VerlBackend=Backend)
    assert install_c0_post_batch_weight_sync_skip(module) == "installed"
    assert install_c0_post_batch_weight_sync_skip(module) == "already_installed"
