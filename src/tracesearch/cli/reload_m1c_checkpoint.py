"""Fresh-process integrity check for an M1-C actor checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = Path(args.output)
    result: dict[str, object] = {
        "checkpoint_path": args.checkpoint,
        "process": sys.executable,
        "loaded": False,
        "tokenizer_loaded": False,
        "adapter_loaded": False,
        "parser_succeeded": False,
        "trace_search_inference": False,
    }
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        actor = Path(args.checkpoint) / "actor"
        tokenizer = AutoTokenizer.from_pretrained(actor / "huggingface", local_files_only=True)
        result["tokenizer_loaded"] = True
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        adapter = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        state_path = actor / "model_world_size_1_rank_0.pt"
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        adapter_state = {key: value for key, value in state.items() if ".lora_" in key}
        missing, unexpected = adapter.load_state_dict(adapter_state, strict=False)
        if not adapter_state or unexpected:
            raise RuntimeError(f"adapter state load was incomplete: missing={len(missing)}, unexpected={len(unexpected)}")
        result["loaded"] = True
        result["adapter_loaded"] = True
        result["adapter_tensor_count"] = len(adapter_state)

        prompt = "Respond with exactly one TraceSearch action and no explanation: <answer>yes</answer>"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            generated = adapter.generate(**inputs, max_new_tokens=24, do_sample=False)
        text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        from tracesearch.agent.parser import parse_policy_output

        parsed = parse_policy_output(text)
        result["parser_succeeded"] = True
        result["parsed_action"] = parsed.action.kind.value
        result["parsed_text"] = text
        result["trace_search_inference"] = parsed.action.kind.value in {"search", "visit", "answer"}
        result["success"] = bool(result["loaded"] and result["adapter_loaded"] and result["parser_succeeded"] and result["trace_search_inference"])
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["success"] = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
