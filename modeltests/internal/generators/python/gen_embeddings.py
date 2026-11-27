#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers",
#   "sentence-transformers",
#   "huggingface-hub",
#   "numpy",
# ]
# ///
"""Generate reference spec.json and embeddings.f32 for HuggingFace embedding models.

Usage examples:
    uv run modeltests/internal/generators/python/gen_embeddings.py \
        --model BAAI/bge-small-en-v1.5 \
        --query-prefix "Represent this sentence for searching relevant passages: " \
        --out modeltests/testdata/baai-bge-small-en-v1.5

    uv run modeltests/internal/generators/python/gen_embeddings.py \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --out modeltests/testdata/sentence-transformers-all-minilm-l6-v2
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel, AutoTokenizer


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
        # Path is .../snapshots/<sha>/config.json
        rev = Path(cfg_path).parent.name
        if len(rev) == 40:
            return rev
    except Exception:
        pass
    return ""


def load_optional_json(model_id: str, filename: str, revision: str | None) -> Any:
    try:
        path = hf_hub_download(model_id, filename, revision=revision or None)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def pool_and_normalize(
    last_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling_cfg: dict[str, Any] | None,
    do_normalize: bool,
    is_causal: bool,
) -> torch.Tensor:
    # last_hidden: [B, S, H], attention_mask: [B, S]
    h = last_hidden.float()
    mask = attention_mask.to(h.device)

    mode = "mean"
    if pooling_cfg:
        if pooling_cfg.get("pooling_mode_cls_token"):
            mode = "cls"
        elif pooling_cfg.get("pooling_mode_lasttoken"):
            mode = "last"
        elif pooling_cfg.get("pooling_mode_mean_tokens"):
            mode = "mean"
        elif pooling_cfg.get("pooling_mode_max_tokens"):
            mode = "max"
    elif is_causal:
        mode = "last"

    if mode == "cls":
        pooled = h[:, 0, :]
    elif mode == "last":
        lengths = mask.sum(dim=1) - 1
        batch_idx = torch.arange(h.size(0), device=h.device)
        pooled = h[batch_idx, lengths, :]
    elif mode == "max":
        expanded = mask.unsqueeze(-1).bool()
        pooled = h.masked_fill(~expanded, -1e9).max(dim=1).values
    else:
        expanded = mask.unsqueeze(-1).float()
        summed = (h * expanded).sum(dim=1)
        counts = expanded.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counts

    if do_normalize:
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1, eps=1e-12)
    return pooled


def compute_embeddings_transformers(
    model_id: str,
    revision: str | None,
    sentences: list[str],
    device: str,
    torch_dtype: torch.dtype,
    dump_layers_dir: Path | None,
) -> tuple[np.ndarray, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision or None, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision or None,
        dtype=torch_dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    modules = load_optional_json(model_id, "modules.json", revision) or []
    pooling_cfg = load_optional_json(model_id, "1_Pooling/config.json", revision)
    do_normalize = any(m.get("type") == "sentence_transformers.models.Normalize" for m in modules)

    model_type = getattr(model.config, "model_type", "").lower()
    is_causal = model_type not in ("bert", "roberta", "xlm-roberta", "deberta", "deberta-v2", "distilbert", "albert", "electra")

    embs = []
    if dump_layers_dir is not None:
        dump_layers_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, text in enumerate(sentences):
            enc = tokenizer(text, return_tensors="pt", padding=False, truncation=True).to(device)
            outputs = model(**enc, output_hidden_states=(dump_layers_dir is not None))
            pooled = pool_and_normalize(
                outputs.last_hidden_state,
                enc["attention_mask"],
                pooling_cfg,
                do_normalize,
                is_causal,
            )
            embs.append(pooled[0].cpu().float().numpy())

            if dump_layers_dir is not None and outputs.hidden_states is not None:
                for layer_idx, hs in enumerate(outputs.hidden_states):
                    arr = hs[0].cpu().float().numpy().astype("<f4")
                    (dump_layers_dir / f"sent{idx}_layer{layer_idx}.f32").write_bytes(arr.tobytes())

    return np.stack(embs, axis=0).astype(np.float32), tokenizer


def compute_embeddings_st(
    model_id: str,
    revision: str | None,
    sentences: list[str],
    device: str,
    torch_dtype: torch.dtype,
) -> tuple[np.ndarray, Any]:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(
        model_id,
        revision=revision or None,
        device=device,
        trust_remote_code=True,
        model_kwargs={"dtype": torch_dtype},
    )
    embs = st.encode(sentences, convert_to_numpy=True, normalize_embeddings=False)
    return np.asarray(embs, dtype=np.float32), st.tokenizer


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


def cosine_matrix(embs: np.ndarray) -> list[list[float]]:
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = embs / norms
    sim = normed @ normed.T
    return [[round(float(x), 6) for x in row] for row in sim]


def write_f32(path: Path, arr: np.ndarray) -> None:
    flat = np.asarray(arr, dtype="<f4").ravel()
    path.write_bytes(flat.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate modeltests spec.json and embeddings.f32 via PyTorch.")
    parser.add_argument("--model", required=True, help="HuggingFace model repository ID")
    parser.add_argument("--revision", default="", help="Optional commit SHA / branch")
    parser.add_argument("--out", required=True, help="Output directory under modeltests/testdata/<name>")
    parser.add_argument("--name", default="", help="Model slug name (defaults to basename of --out)")
    parser.add_argument("--tier", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--tags", default="safetensors,embed", help="Comma-separated tags")
    parser.add_argument("--gated", action="store_true")
    parser.add_argument("--query-prefix", default="", help="Prefix prepended to shared query sentences")
    parser.add_argument("--doc-prefix", default="", help="Prefix prepended to shared document sentences")
    parser.add_argument("--extra-sentence", action="append", default=[], help="Extra sentence(s) to embed")
    parser.add_argument("--ref-sentences", default=str(default_ref_sentences_path()))
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--causal-mask", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--approx-gelu", default="false", choices=["auto", "true", "false"],
                        help="Set model_options.approximate_gelu (PyTorch BERT uses exact erf GELU = false)")
    parser.add_argument("--min-cosine", type=float, default=0.9995)
    parser.add_argument("--max-abs-diff", type=float, default=5e-3)
    parser.add_argument("--max-mean-abs-diff", type=float, default=1e-3)
    parser.add_argument("--use-st", action="store_true", help="Use sentence-transformers library if installed")
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

    with open(args.ref_sentences, "r", encoding="utf-8") as f:
        shared = json.load(f)

    sentences = (
        [args.query_prefix + q for q in shared["queries"]]
        + [args.doc_prefix + d for d in shared["documents"]]
        + list(args.extra_sentence)
    )
    tok_texts = list(sentences) + list(shared.get("tokenizer_edge_cases", []))

    revision = resolve_revision(args.model, args.revision or None)
    dump_dir = (out_dir / "debug_layers") if args.dump_layers else None

    print(f"Generating embeddings for {args.model} (rev={revision or 'main'}, device={device}, dtype={args.dtype})...", file=sys.stderr)

    used_lib = f"torch=={torch.__version__}, transformers=={transformers.__version__}"
    if args.use_st and dump_dir is None:
        try:
            import sentence_transformers

            embs, tokenizer = compute_embeddings_st(args.model, revision or None, sentences, device, torch_dtype)
            used_lib += f", sentence-transformers=={sentence_transformers.__version__}"
        except ImportError:
            print("sentence_transformers not installed; falling back to transformers AutoModel.", file=sys.stderr)
            embs, tokenizer = compute_embeddings_transformers(args.model, revision or None, sentences, device, torch_dtype, dump_dir)
    else:
        embs, tokenizer = compute_embeddings_transformers(args.model, revision or None, sentences, device, torch_dtype, dump_dir)

    emb_filename = "embeddings.f32"
    write_f32(out_dir / emb_filename, embs)

    model_options: dict[str, Any] = {}
    if args.causal_mask != "auto":
        model_options["causal_mask"] = args.causal_mask == "true"
    if args.approx_gelu != "auto":
        model_options["approximate_gelu"] = args.approx_gelu == "true"

    spec: dict[str, Any] = {
        "name": model_name,
        "family": "transformer",
        "format": "safetensors",
        "repo_id": args.model,
        "revision": revision,
        "size_tier": args.tier,
        "tags": [t.strip() for t in args.tags.split(",") if t.strip()],
        "gated": args.gated,
        "model_options": model_options,
        "provenance": {
            "generator": "modeltests/internal/generators/python/gen_embeddings.py",
            "command": " ".join(sys.argv),
            "runtime_version": used_lib,
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
                "embedding": {
                    "use_shared_sentences": True,
                    "query_prefix": args.query_prefix,
                    "document_prefix": args.doc_prefix,
                    "extra_sentences": list(args.extra_sentence),
                    "embedding_dim": int(embs.shape[1]),
                    "embeddings_file": emb_filename,
                    "similarity_matrix": cosine_matrix(embs),
                    "tolerances": {
                        "min_cosine_sim": args.min_cosine,
                        "max_abs_diff": args.max_abs_diff,
                        "max_mean_abs_diff": args.max_mean_abs_diff,
                        "similarity_max_delta": 0.01,
                    },
                },
            }
        ],
    }

    spec_path = out_dir / "spec.json"
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {spec_path} and {out_dir / emb_filename} ({embs.shape[0]}x{embs.shape[1]})", file=sys.stderr)


if __name__ == "__main__":
    main()
