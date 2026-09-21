"""Runtime provenance helpers for optional accelerator extensions."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
from typing import Any


def _module_record(name: str) -> dict[str, Any]:
    record: dict[str, Any] = {"name": name}
    try:
        module = importlib.import_module(name)
        record["module_file"] = getattr(module, "__file__", None)
        record["module_origin"] = getattr(getattr(module, "__spec__", None), "origin", None)
        record["imported"] = True
        record["version"] = getattr(module, "__version__", None)
    except Exception as exc:
        record.update({"imported": False, "error": f"{type(exc).__name__}: {exc}"})
    return record


def flash_attn_provenance() -> dict[str, Any]:
    """Describe the imported external FlashAttention surface unambiguously."""

    result: dict[str, Any] = {
        "shadowing_check": "src/flash_attn is absent; import must resolve outside TraceSearch source",
        "modules": [_module_record(name) for name in ("flash_attn", "flash_attn_2_cuda")],
        "symbols": {},
        "distributions": {},
    }
    try:
        package = importlib.import_module("flash_attn")
        for name in ("flash_attn.bert_padding", "flash_attn.ops.triton.rotary"):
            module = importlib.import_module(name)
            result["symbols"][name] = {
                "module_file": getattr(module, "__file__", None),
                "unpad_input": getattr(getattr(module, "unpad_input", None), "__module__", None),
                "pad_input": getattr(getattr(module, "pad_input", None), "__module__", None),
                "apply_rotary": getattr(getattr(module, "apply_rotary", None), "__module__", None),
            }
        result["compiled_cuda_extension_loaded"] = importlib.util.find_spec("flash_attn_2_cuda") is not None
        result["package_file"] = getattr(package, "__file__", None)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["compiled_cuda_extension_loaded"] = False
    for dist_name in ("flash-attn", "flash-attn-4"):
        try:
            dist = metadata.distribution(dist_name)
        except metadata.PackageNotFoundError:
            continue
        direct_url = dist.read_text("direct_url.json")
        result["distributions"][dist_name] = {
            "version": dist.version,
            "dist_info": str(dist._path),
            "direct_url": json.loads(direct_url) if direct_url else None,
        }
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
