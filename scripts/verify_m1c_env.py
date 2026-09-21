"""Fail-fast import and provenance check for the Linux M1-C uv environment."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import platform
import sys


PACKAGES = (
    "torch",
    "transformers",
    "ray",
    "vllm",
    "verl",
    "rllm",
    "hydra-core",
    "omegaconf",
    "flash-attn",
)


def main() -> int:
    if platform.system() != "Linux":
        raise SystemExit("M1-C verification must run on Linux/WSL2")

    print(f"python={sys.executable}")
    print(f"python_version={platform.python_version()}")
    print(f"virtual_env={os.environ.get('VIRTUAL_ENV', '<not activated; explicit interpreter is fine>')}")

    for distribution in PACKAGES:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise SystemExit(f"missing distribution: {distribution}") from exc
        print(f"{distribution}=={version}")

    required_modules = (
        "rllm.trainer.config",
        "verl.utils.device",
        "vllm",
        "flash_attn",
    )
    for module_name in required_modules:
        if importlib.util.find_spec(module_name) is None:
            raise SystemExit(f"missing import: {module_name}")
        module = importlib.import_module(module_name)
        print(f"import {module_name} -> {getattr(module, '__file__', '<namespace>')}")

    import torch

    print(f"torch_cuda_version={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; stop before starting the M1-C backend")
    print(f"cuda_device={torch.cuda.get_device_name(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
