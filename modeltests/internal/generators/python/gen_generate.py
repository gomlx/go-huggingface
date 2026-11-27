#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "huggingface-hub",
#   "numpy",
# ]
# ///
"""Generate reference spec.json for HuggingFace generative LLMs (temperature = 0).

Exports tokenizer Encode/Decode ground truth, prefill top-K logits at the last prompt
position, and deterministic greedy (do_sample=False) completions.

Usage example:
    uv run modeltests/internal/generators/python/gen_generate.py \
        --model google/gemma-4-E4B-it \
        --chat-prefix "<bos><|turn>user\n" \
        --chat-suffix "<turn|>\n<|turn>model\n" \
        --stop-token "<turn|>" \
        --max-new-tokens 32 \
        --dtype bfloat16 \
        --tier medium --gated \
        --out modeltests/testdata/gemma4-e4b
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def default_ref_prompts_path() -> Path:
    return Path(__file__).resolve().parents[3] / "testdata" / "reference_prompts.json"


def default_ref_sentences_path() -> Path:
    return Path(__file__).resolve().parents[3] / "testdata" / "reference_sentences.json"


def resolve_revision(model_id: str, requested_rev: str | None) -> str:
    if requested_rev:
        return requested_rev
    try:
        snap = snapshot_download(model_id, local_files_only=True, allow_patterns=["config.json"])
        rev = Path(snap).name
        if len(rev) == 40:
            return rev
    except Exception:
        pass
    try:
        cfg_path = hf_hub_download(model_id, "config.json")
        rev = Path(cfg_path).parent.name
        if len(rev) == 40:
            return rev
    except Exception:
        pass
    return ""


def extract_special_tokens(tokenizer: Any) -> dict[str, int]:
    mapping = {
        "bos": "bos_token_id",
        "eos": "eos_token_id",
        "unk": "unk_token_id",
        "pad": "pad_token_id",
        "mask": "mask_token_id",
        "cls": "cls_token_id",
    }
    out: dict[str, int] = {}
    for key, attr in mapping.items():
        val = getattr(tokenizer, attr, None)
        if val is not None:
            out[key] = int(val)
    return out


def build_tokenizer_cases(tokenizer: Any, texts: list[str]) -> list[dict[str, Any]]:
    cases = []
    seen = set()
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=True)]
        decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        decoded_skip = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        cases.append({
            "text": text,
            "token_ids": ids,
            "decoded": decoded,
            "decoded_skip_special": decoded_skip,
        })
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate modeltests spec.json for generative LLMs (temp=0).")
    parser.add_argument("--model", required=True, help="HuggingFace model repository ID")
    parser.add_argument("--revision", default="", help="Optional commit SHA / branch")
    parser.add_argument("--out", required=True, help="Output directory under modeltests/testdata/<name>")
    parser.add_argument("--name", default="", help="Model slug name (defaults to basename of --out)")
    parser.add_argument("--tier", default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--tags", default="safetensors,generate", help="Comma-separated tags")
    parser.add_argument("--gated", action="store_true")
    parser.add_argument("--ref-prompts", default=str(default_ref_prompts_path()))
    parser.add_argument("--ref-sentences", default=str(default_ref_sentences_path()))
    parser.add_argument("--chat-prefix", default="", help="Prefix for non-raw prompts")
    parser.add_argument("--chat-suffix", default="", help="Suffix for non-raw prompts")
    parser.add_argument("--stop-token", action="append", default=[], help="Additional stop token strings")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--raw-only", action="store_true", help="Only run raw completion prompts (skip chat-templated prompts)")
    parser.add_argument("--logit-max-abs-diff", type=float, default=0.75)
    parser.add_argument("--min-prefix-tokens", type=int, default=0,
                        help="Min matching greedy prefix tokens (0 = default min(16, len))")
    parser.add_argument("--dump-layers", action="store_true", help="Dump hidden states to <out>/debug_layers/")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.name or out_dir.name

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    torch_dtype = dtype_map[args.dtype]

    with open(args.ref_prompts, "r", encoding="utf-8") as f:
        shared_prompts = json.load(f)["prompts"]
    with open(args.ref_sentences, "r", encoding="utf-8") as f:
        shared_sentences = json.load(f)

    revision = resolve_revision(args.model, args.revision or None)
    print(f"Loading {args.model} (rev={revision or 'main'}, device={device}, dtype={args.dtype})...", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision or None, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=revision or None,
        dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    stop_token_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            stop_token_ids.extend(int(x) for x in tokenizer.eos_token_id)
        else:
            stop_token_ids.append(int(tokenizer.eos_token_id))
    for st in args.stop_token:
        tid = tokenizer.convert_tokens_to_ids(st)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_token_ids.append(int(tid))
    stop_token_ids = sorted(set(stop_token_ids))

    dump_dir = (out_dir / "debug_layers") if args.dump_layers else None
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)

    gen_cases = []
    tok_texts: list[str] = []

    with torch.no_grad():
        for p_idx, entry in enumerate(shared_prompts):
            if args.raw_only and not entry.get("raw", True):
                continue
            prompt = entry["prompt"]
            if entry.get("raw", True) or (not args.chat_prefix and not args.chat_suffix):
                formatted = prompt
            else:
                formatted = args.chat_prefix + prompt + args.chat_suffix
            tok_texts.append(formatted)

            prompt_ids = [int(x) for x in tokenizer.encode(formatted, add_special_tokens=True)]
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            # 1. Prefill forward pass for top-K logits at final prompt position.
            outputs = model(input_ids=input_ids, output_hidden_states=(dump_dir is not None))
            last_logits = outputs.logits[0, -1, :].float().cpu()
            top_vals, top_idxs = torch.topk(last_logits, k=args.top_k)
            prefill_top_k = [
                {
                    "token_id": int(tid),
                    "text": tokenizer.decode([int(tid)], skip_special_tokens=False),
                    "logit": round(float(val), 5),
                }
                for val, tid in zip(top_vals.tolist(), top_idxs.tolist())
            ]

            if dump_dir is not None and outputs.hidden_states is not None:
                for layer_idx, hs in enumerate(outputs.hidden_states):
                    arr = hs[0].cpu().float().numpy().astype("<f4")
                    (dump_dir / f"prompt{p_idx}_layer{layer_idx}.f32").write_bytes(arr.tobytes())

            # 2. Deterministic greedy generation (temperature = 0 / do_sample = False).
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
            }
            if stop_token_ids:
                gen_kwargs["eos_token_id"] = stop_token_ids
            gen_out = model.generate(input_ids=input_ids, **gen_kwargs)
            new_ids = [int(x) for x in gen_out[0, len(prompt_ids):].tolist()]
            if new_ids and new_ids[-1] in stop_token_ids:
                new_ids = new_ids[:-1]
            greedy_text = tokenizer.decode(new_ids, skip_special_tokens=False)

            min_prefix = args.min_prefix_tokens if args.min_prefix_tokens > 0 else min(16, len(new_ids))
            gen_cases.append({
                "name": entry["name"],
                "prompt": prompt,
                "formatted_prompt": formatted,
                "prompt_token_ids": prompt_ids,
                "prefill_top_k": prefill_top_k,
                "greedy_token_ids": new_ids,
                "greedy_text": greedy_text,
                "tolerances": {
                    "logit_max_abs_diff": args.logit_max_abs_diff,
                    "logit_min_cosine_sim": 0.995,
                    "top1_margin_tolerance": 0.25,
                    "min_exact_prefix_tokens": min_prefix,
                    "kvcache_parity_tokens": min(8, len(new_ids)),
                },
            })
            print(f"  [{entry['name']}] top1={prefill_top_k[0]['text']!r} ({prefill_top_k[0]['logit']:.2f}), greedy={greedy_text!r}", file=sys.stderr)

    tok_texts.extend(shared_sentences.get("tokenizer_edge_cases", []))

    spec: dict[str, Any] = {
        "name": model_name,
        "family": "transformer",
        "format": "safetensors",
        "repo_id": args.model,
        "revision": revision,
        "size_tier": args.tier,
        "tags": [t.strip() for t in args.tags.split(",") if t.strip()],
        "gated": args.gated,
        "model_options": {
            "causal_mask": True,
        },
        "provenance": {
            "generator": "modeltests/internal/generators/python/gen_generate.py",
            "command": " ".join(sys.argv),
            "runtime_version": f"torch=={torch.__version__}, transformers=={transformers.__version__}",
            "dtype": args.dtype,
            "device": device,
            "generated_at": dt.date.today().isoformat(),
        },
        "tokenizer": {
            "special_tokens": extract_special_tokens(tokenizer),
            "cases": build_tokenizer_cases(tokenizer, tok_texts),
        },
        "variants": [
            {
                "name": "default",
                "generate": {
                    "chat_template_prefix": args.chat_prefix,
                    "chat_template_suffix": args.chat_suffix,
                    "stop_tokens": list(args.stop_token),
                    "max_new_tokens": args.max_new_tokens,
                    "cases": gen_cases,
                },
            }
        ],
    }

    spec_path = out_dir / "spec.json"
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {spec_path} ({len(gen_cases)} prompt cases)", file=sys.stderr)


if __name__ == "__main__":
    main()
