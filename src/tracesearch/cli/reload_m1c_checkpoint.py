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

        # Use the same chat-template boundary as the live policy.  A bare
        # completion prompt is not a valid probe for an instruct/chat model:
        # it can make the model continue the example or emit multiple actions,
        # producing a parser failure even though the adapter loaded correctly.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a text search agent. Emit exactly one action per turn using "
                    "<think>...</think> followed by exactly one of "
                    "<search>query</search>, <visit>doc_id</visit>, or "
                    "<answer>final answer</answer>. Do not emit code or other tags. "
                    "For factual questions, the first turn must search before answering."
                ),
            },
            {"role": "user", "content": "Find one relevant document and emit the first search action."},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(rendered, dict):
                inputs = rendered
            else:
                inputs = {"input_ids": rendered}
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        else:
            prompt = messages[-1]["content"]
            inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            generated = adapter.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        text = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        result["generated_text"] = text
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
