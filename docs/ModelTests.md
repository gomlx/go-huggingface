# Full Model Tests (`modeltests`) — Architecture & Developer Guide

> **Status:** Implemented (2026-11-26)
> This document describes the architecture, conventions, tested model roster, and workflow for end-to-end model parity tests in `go-huggingface`. Keep it updated as harnesses, generators, and tested models evolve.

---

## 1. Goals & Philosophy

1. **Confidence across all supported models & backends**: Verify that every supported native GoMLX model in `go-huggingface` (SafeTensors, GGUF, TimeSeries, etc.) produces accurate tokenization, embeddings, logits, and/or deterministic generations (`temperature = 0`) across GoMLX backends (`go`, `xla`, `onnx`, `mlx`/`mps`) and backend configurations.
   - *(ONNX-converted models such as `KnightsAnalytics/all-MiniLM-L6-v2` belong in a parallel `modeltests/` package inside [`onnx-gomlx`](https://github.com/gomlx/onnx-gomlx), where the ONNX converter and `onnxruntime_go` live, following the same conventions and `spec.json` format.)*
2. **Test-Driven Model Bring-Up**: Whenever support for a new model or architecture is added, the developer starts by generating its reference golden files using an external runtime (PyTorch/HuggingFace or `llama.cpp` via `yzma`) and iterates until the unified `modeltests` suite passes.
3. **First-Class, Low-Boilerplate Structure**: All model tests live in a single top-level package [`modeltests/`](../modeltests/) inside the root `go-huggingface` module with one subdirectory per model under `testdata/`, eliminating copy-pasted test harnesses across `examples/`. (`examples/` keeps clean, minimal user demos.)
4. **Safe & Ergonomic Defaults**: Model tests are **disabled by default** (gated behind `-modeltests`) because they are slow, memory-intensive, and download large weights from HuggingFace Hub.
5. **Zero Dependency Bloat**: Reference generators written in Go (e.g., `yzma`/`llama.cpp`) live in their own `go.mod` sub-modules under `modeltests/internal/generators/`, keeping external CGo/FFI runtimes out of `go-huggingface`'s module graph.

---

## 2. Directory & Module Layout

```text
modeltests/
├── modeltests_test.go                 # Flags (-modeltests, -backends, -max_tier, ...), discovery, summary
├── runner.go                          # Shared types (ModelSpec, VariantSpec, metrics, binary .f32 I/O)
├── memory.go                          # Per-test peak RSS (Linux VmHWM reset + portable RSS sampler)
├── transformer_test.go                # Harness: HF SafeTensors & GGUF embedders + generative LLMs
├── timeseries_test.go                 # Harness: Time-series models (e.g., Timestral)
├── testdata/
│   ├── reference_sentences.json       # Shared canonical sentences (queries, docs, tokenizer edge cases)
│   ├── reference_prompts.json         # Shared low-entropy prompts for generative models (temp=0)
│   ├── baai-bge-small-en-v1.5/
│   │   ├── spec.json
│   │   └── embeddings.f32
│   ├── sentence-transformers-all-minilm-l6-v2/
│   │   ├── spec.json
│   │   └── embeddings.f32
│   ├── bge-base-en-v1.5-gguf/
│   │   ├── spec.json
│   │   └── embeddings_Q4_0.f32
│   ├── minilm-l6-v2-gguf/
│   │   ├── spec.json                  # Declares all 13 GGUF quantization variants
│   │   ├── embeddings_Q2_K.f32
│   │   ├── embeddings_Q3_K_S.f32
│   │   ├── ...
│   │   └── embeddings_F16.f32
│   ├── kalm-gemma3-12b/
│   │   ├── spec.json
│   │   └── embeddings.f32
│   ├── qwen3.5-4b/                    # Hybrid Gated DeltaNet + full attention generative LLM
│   │   └── spec.json
│   ├── gemma4-e4b/                    # Gemma 4 generative LLM
│   │   └── spec.json
│   └── timestral/
│       ├── spec.json                  # Covers 28M and 123M variants
│       ├── context.f32
│       ├── results_28m.f32
│       └── results_123m.f32
└── internal/
    └── generators/                    # Reference generators (not imported by library code)
        ├── python/                    # PyTorch + transformers + sentence-transformers (uv PEP 723)
        │   ├── gen_embeddings.py
        │   └── gen_generate.py
        └── yzma/                      # llama.cpp generator based on ../benchmarks/yzma_embedder
            ├── go.mod                 # Separate Go module (isolates hybridgroup/yzma)
            └── main.go
```

> **Clean Repository Boundary (`go-huggingface` vs `onnx-gomlx`):**
> `modeltests/` stays directly inside the root `go-huggingface/go.mod` with **no new dependencies**, so `go test ./...` at the repo root discovers `modeltests` and cleanly prints the `-modeltests` skip notice.
> ONNX-converted models (e.g., `KnightsAnalytics/all-MiniLM-L6-v2`) live in a parallel `onnx-gomlx/modeltests/` package following the exact same CLI flags, `spec.json` / `.f32` schema, and workflow—where `onnx-gomlx` already depends on `go-huggingface` and `onnxruntime_go`.

---

## 3. Running Model Tests

### 3.1 Opt-in Gating (`-modeltests`)

All tests in `modeltests` are **skipped by default**. When `-modeltests` is not provided, tests skip immediately with:

```text
Set -modeltests to enable testing model tests, which are slow, and may require downloading of large models to run.
```

(Optionally, setting environment variable `GOMLX_MODELTESTS=1` enables the same behavior when invoking `go test` across multiple packages where custom flags cannot be passed.)

### 3.2 CLI Flags

| Flag | Default | Description |
| :--- | :--- | :--- |
| `-modeltests` | `false` | Required opt-in flag to execute full model tests. |
| `-backends` | `""` (default backend) | Semicolon-separated list of GoMLX backend configs passed to `compute.NewWithConfig(...)`. When empty, uses `compute.New()` (honoring `GOMLX_BACKEND`). Example: `-backends="go:ops_sequential;go:quant_cached;xla:cpu;onnx:cpu"`. |
| `-max_tier` | `""` (all) | Limit models by resource tier: `small` (< 1 GB RAM), `medium` (1–8 GB RAM), `large` (> 8 GB RAM). |
| `-cached_only` | `false` | Skip models whose weight files are not already present in the local HuggingFace cache (`~/.cache/huggingface/hub/`). |
| `-tag` | `""` | Filter models by tag (e.g. `gguf`, `safetensors`, `embed`, `generate`, `timeseries`). |
| `-strict` | `false` | Fail instead of skipping when a gated model lacks `HF_TOKEN` or a download fails (useful for nightly CI). |
| `-debug_layers` | `false` | If local layer-dump files exist in `testdata/<model>/debug_layers/`, run layer-by-layer activation checks. |
| `-set` | `""` | GoMLX store hyperparameter overrides (via `commandline.CreateSettingsFlag`, e.g., `-set="quantized_matmul=false"`). |

### 3.3 Common Invocation Examples

```bash
# Run a single model on the default backend (GOMLX_BACKEND)
go test -v -p=1 ./modeltests -modeltests -run TestModels/baai-bge-small-en-v1.5

# Run a single GGUF quantization variant across multiple Go & XLA/ONNX backend configs
go test -v -p=1 ./modeltests -modeltests \
  -run TestModels/minilm-l6-v2-gguf/Q4_K_M \
  -backends="go:ops_sequential;go:quant_cached;xla:cpu;onnx:cpu"

# Run all 13 GGUF quantizations of all-MiniLM-L6-v2 on the Go backend
go test -v -p=1 ./modeltests -modeltests -run TestModels/minilm-l6-v2-gguf -backends="go"

# Run all small models that are already cached locally
go test -v -p=1 ./modeltests -modeltests -max_tier=small -cached_only
```

### 3.4 Subtest Naming Hierarchy

Every test follows a predictable subtest tree so `-run` filtering works naturally at any granularity:

```text
TestModels/<model-slug>/<variant>/<backend-config>/<check>
```

Examples:
- `TestModels/baai-bge-small-en-v1.5/default/go/tokenize`
- `TestModels/baai-bge-small-en-v1.5/default/go/variables`
- `TestModels/baai-bge-small-en-v1.5/default/go/embed_single`
- `TestModels/baai-bge-small-en-v1.5/default/go/embed_batch`
- `TestModels/minilm-l6-v2-gguf/Q4_K_M/go:quant_cached/embed_single`
- `TestModels/gemma4-e4b/default/xla:cuda/prefill_logits`
- `TestModels/gemma4-e4b/default/xla:cuda/kvcache_parity`
- `TestModels/gemma4-e4b/default/xla:cuda/greedy_generate`

---

## 4. Peak Memory Reporting & Sequential Execution

### 4.1 Sequential Execution

- Tests in `modeltests` **never call `t.Parallel()`**, ensuring models run strictly one at a time within the package so memory measurements do not overlap and GPU/RAM budgets are respected.
- Between every `(model, variant, backend)` run, the harness explicitly calls `store.Finalize()`, finalizes cached executors/tensors, finalizes the backend (when switching backend configs), and runs `runtime.GC()` + `debug.FreeOSMemory()`.
- **Important Go caveat on `-p=1`**: Go's `go test -p=1` flag controls package-level parallelism and is consumed by the `go` driver rather than passed into the test binary, so a test binary cannot directly inspect `-p`. However, running `go test ./modeltests -modeltests` only executes a single package anyway, and we can check `flag.Lookup("test.parallel")` (skipping or warning if parallel test execution is explicitly enabled) while documenting `-p=1` for multi-package invocations.

### 4.2 Per-Test Peak Memory Measurement

`syscall.Getrusage(RUSAGE_SELF).Maxrss` reports the **process-lifetime** high-water mark—it never resets between subtests, so running a 12B model followed by a 22M model would report 24 GB for both. To get accurate **per-model** peak memory:

1. **Linux (primary)**: Before each `(model, variant, backend)` subtest (after `runtime.GC()` + `debug.FreeOSMemory()`), reset the kernel's per-process high-water mark by writing `"5\n"` to `/proc/self/clear_refs`, then record baseline `VmRSS` from `/proc/self/status`. After the subtest finishes, read `VmHWM` (exact kernel-tracked peak RSS during that subtest) and `VmRSS` (post-run RSS).
2. **macOS / portable fallback**: If `/proc/self/clear_refs` is unavailable, run a lightweight background sampler (every ~20 ms) reading current process RSS (`mach_task_basic_info` on Darwin / `VmRSS` on Linux) to track the peak during the subtest.
3. **Reported metrics per `(model, variant, backend)`**:
   - **Peak RSS** (during this subtest) and **Delta RSS** (`Peak RSS - Baseline RSS`)
   - **Go Heap Peak** (`runtime.MemStats.HeapAlloc` / `Sys`)
   - **Process Max RSS** (`syscall.Getrusage`)
   - **Timings**: Weight load time, graph compile time, and inference latency / throughput (passages/s or tok/s).

> **Note on CGo/XLA/CUDA allocators**: C++ runtimes (XLA/PJRT, ORT) often retain freed memory pools across runs inside the same process. When exact isolated process RSS is needed across many models, run individual models via `-run TestModels/<model>` (or a future `-isolate` flag that re-execs the test binary per model).

---

## 5. Shared Reference Inputs & `spec.json` Schema

### 5.1 Shared Reference Corpora

To avoid duplicating test sentences across dozens of models—while still letting any model add custom inputs—canonical inputs live in two shared files under `modeltests/testdata/`:

1. **`testdata/reference_sentences.json`** (for embedders & tokenizer parity):
   - **Dolly (`databricks/databricks-dolly-15k`) & MS MARCO (`microsoft/ms_marco`) Q&A pairs**:
     - Queries:
       - `"Why can camels survive for long without water?"` (Dolly)
       - `"What individual has won the most Olympic gold medals in the history of the games?"` (Dolly)
       - `"are whiskers on cats used for balance"` (MS MARCO)
       - `"what is early bird check in southwest"` (MS MARCO)
     - Documents: the corresponding Dolly answers and MS MARCO selected passages.
   - **Multilingual `"Why is the sky blue?"` queries**:
     - English: `"Why is the sky blue?"`
     - German: `"Warum ist der Himmel blau?"`
     - French: `"Pourquoi le ciel est-il bleu ?"`
     - Chinese: `"为什么天空是蓝色的？"`
     - Japanese: `"なぜ空は青いのですか？"`
   - **Tokenizer edge-case sentences** (tested for tokenization parity across all models):
     - Punctuation & contractions: `"Don't panic! Price: $19.99 (20% off) — see https://example.com/a_b?x=1."`
     - Unicode, accents & multilingual: `"Café naïve résumé — Munich, Zürich, 東京, 京都."`
     - Numbers, whitespace & symbols: `"  Leading   and\tmultiple\nspaces  1234567890  a+b=c  "`
     - Single-word boundary case: `"Hello"`

2. **`testdata/reference_prompts.json`** (for generative LLMs, `temperature = 0`):
   - Prompts from Dolly & MS MARCO (`dolly_camels`, `dolly_vermeer`, `msmarco_whiskers`), `"Why is the sky blue?"` in English, German, French, Chinese, and Japanese (`sky_en`, `sky_de`, `sky_fr`, `sky_zh`, `sky_ja`), plus a raw Python code-completion prompt (`code_fib`) that exercises multi-space indentation.

Models can:
- Reference the shared corpus (`"use_shared_sentences": true` / `"use_shared_prompts": true`), automatically applying model-specific query/document prefixes or chat templates on top.
- Add model-specific entries via `"extra_sentences"` / `"extra_prompts"`.
- Fully override inputs via `"sentences"` / `"prompts"` when needed (e.g., domain-specific models or non-English models).

### 5.2 Model Specification (`testdata/<model-slug>/spec.json`)

Each model subdirectory contains a `spec.json`:

```json
{
  "name": "minilm-l6-v2-gguf",
  "family": "transformer",
  "format": "gguf",
  "repo_id": "second-state/All-MiniLM-L6-v2-Embedding-GGUF",
  "revision": "<40-char-commit-sha>",
  "size_tier": "small",
  "tags": ["gguf", "bert", "embed"],
  "gated": false,
  "model_options": {
    "causal_mask": false,
    "approximate_gelu": true,
    "force_float32": false,
    "ignored_checkpoint_prefixes": []
  },
  "provenance": {
    "generator": "modeltests/internal/generators/yzma",
    "command": "go run . -model second-state/All-MiniLM-L6-v2-Embedding-GGUF -all_quants -out ../../testdata/minilm-l6-v2-gguf",
    "runtime_version": "llama.cpp bXXXX",
    "dtype": "fp32",
    "device": "cpu",
    "generated_at": "2026-11-26"
  },
  "tokenizer": {
    "special_tokens": {
      "pad": 0,
      "unk": 100,
      "cls": 101,
      "sep": 102,
      "mask": 103
    },
    "cases": [
      {
        "text": "Why is the sky blue?",
        "token_ids": [101, 2339, 2003, 1996, 3712, 2630, 1029, 102],
        "decoded": "[CLS] why is the sky blue? [SEP]",
        "decoded_skip_special": "why is the sky blue?"
      }
    ]
  },
  "variants": [
    {
      "name": "Q4_K_M",
      "gguf_file": "all-MiniLM-L6-v2-Q4_K_M.gguf",
      "embedding": {
        "use_shared_sentences": true,
        "embedding_dim": 384,
        "embeddings_file": "embeddings_Q4_K_M.f32",
        "similarity_matrix": [[1.0, 0.12, 0.85, 0.09], ...],
        "tolerances": {
          "min_cosine_sim": 0.9995,
          "max_abs_diff": 0.005,
          "max_mean_abs_diff": 0.001
        }
      }
    }
  ]
}
```

For **generative LLM** variants, each variant includes a `"generate"` section alongside `"tokenizer"`:

```json
{
  "generate": {
    "chat_template_prefix": "<bos><|turn>user\n",
    "chat_template_suffix": "<turn|>\n<|turn>model\n",
    "max_new_tokens": 32,
    "cases": [
      {
        "prompt": "Why is the sky blue?",
        "formatted_prompt": "<|turn>user\nWhy is the sky blue?<turn|>\n<|turn>model\n",
        "prompt_token_ids": [2, ...],
        "prefill_top_k": [
          { "token_id": 818, "text": "The", "logit": 21.4375 }
        ],
        "greedy_token_ids": [818, ...],
        "greedy_text": "The sky appears blue because...",
        "tolerances": {
          "logit_max_abs_diff": 0.75,
          "logit_min_cosine_sim": 0.995,
          "min_exact_prefix_tokens": 16
        }
      }
    ]
  }
}
```

---

## 6. Test Checks by Model Family

### 6.1 Common Checks (All Transformer Models)
1. **Tokenizer Parity (`tokenize`)**:
   - Verifies special token IDs (`BOS`, `EOS`, `PAD`, `UNK`, `CLS`, `SEP`, `MASK`) when present in `spec.json`.
   - Verifies exact `tokenizer.Encode(text) == expected.token_ids`.
   - Verifies `tokenizer.Decode(expected.token_ids)` against reference decoded strings (checking both raw decode and special-token handling where applicable).
2. **Graph & Weight Hygiene (`variables`)**:
   - Captures all variables loaded into `model.Store` before graph construction, builds the graph, and asserts **zero unloaded variables** were created.
   - Optionally reports any checkpoint tensors that were ignored (validating against `ignored_checkpoint_prefixes`, e.g., `model.visual.*`, `mtp.*`).

### 6.2 Sentence Embedders (`embed_single`, `embed_batch`)
1. **Single-sentence execution**: Runs each input individually (unpadded) and validates against the reference `.f32` vector.
2. **Batched padded execution**: Pads all sentences in a batch with `padID` + passes `seqLen` / attention mask, validating that every non-padded row still matches the reference vector within tolerance (catches masking/padding/RoPE bugs).
3. **Metrics**: Reports and asserts cosine similarity, max absolute difference, mean absolute difference, and (when provided) pairwise cosine similarity matrix error.

### 6.3 Generative LLMs (`prefill_logits`, `kvcache_parity`, `greedy_generate`)
Greedy decoding (`temperature = 0`) can diverge across runtimes/dtypes if two candidate tokens are nearly tied. Therefore generative tests use a **two-tier verification**:
1. **Prefill / Teacher-Forced Top-K Logits (`prefill_logits`)**: Runs a single forward pass on the prompt token IDs and checks the top-K (e.g. top-16) reference token logits (cosine similarity, max abs diff, and top-1 agreement when margin > tolerance). Fast, deterministic, and isolates math errors from decoding drift.
2. **Internal KV-Cache Parity (`kvcache_parity`)**: Runs a short greedy rollout (e.g., 16 tokens) with both naive (no KV cache) and KV-cached generation inside GoMLX and asserts **exact token parity + tight logit agreement**. Directly catches KV-cache, sliding-window, RoPE offset, and Gated DeltaNet state bugs with zero cross-framework noise.
3. **Greedy Generation (`greedy_generate`)**: Runs `temperature = 0` generation for `max_new_tokens` on low-entropy prompts and compares generated token IDs and decoded text against the reference (`min_exact_prefix_tokens` / exact match).

### 6.4 Time-Series Models (`timeseries_test.go`)
- Loads binary `.f32` context and expected forecast tensors (e.g., Timestral 28M & 123M), validates `max_abs_diff < 1e-4` across backends, and supports `-force_f32` / backend capability handling automatically.

### 6.5 Optional Layer-by-Layer Debug Check (`-debug_layers`)
- During model bring-up, generators can be invoked with `--dump-layers` to write intermediate activations into `testdata/<model>/debug_layers/` (which is `.gitignore`d so huge activation dumps are never committed).
- Passing `-debug_layers` to `go test` automatically runs `AllLayers` and pinpoints the exact first layer where GoMLX diverges from the reference runtime.

---

## 7. Reference Generators (`modeltests/internal/generators/`)

All golden files **must** be produced by an independent reference runtime (never by `go-huggingface` itself, including tokenization!).

### 7.1 Python Generator (`internal/generators/python/`)
- Uses [PEP 723 inline script metadata](https://peps.python.org/pep-0723/) so scripts run reproducibly with `uv run` and zero manual virtualenv setup:
  - `gen_embeddings.py` — PyTorch + `sentence-transformers` + `transformers` (exports `spec.json` + `.f32` + tokenizer `Encode`/`Decode` ground truth + optional `--dump-layers`).
  - `gen_generate.py` — PyTorch + `transformers` (exports `spec.json` with tokenizer `Encode`/`Decode`, prefill top-K logits, and `do_sample=False` greedy completions + optional `--dump-layers`).

### 7.2 Go + `yzma` / `llama.cpp` Generator (`internal/generators/yzma/`)
- Lives in its own `go.mod` (`module github.com/gomlx/go-huggingface/modeltests/internal/generators/yzma`), so `github.com/hybridgroup/yzma` never enters `go-huggingface`'s dependency graph.
- Built on the battle-tested `llama.cpp` loader, pooling, and GGUF resolution logic from `../benchmarks/yzma_embedder/main.go`.
- **Key difference from `yzma_embedder`**: Uses `llama.cpp`'s own `llama.Tokenize` and `llama.Detokenize` (via `llama.ModelGetVocab`) to produce the reference token IDs and decoded strings—ensuring `go-huggingface`'s embedded GGUF tokenizer is tested against true `llama.cpp` output!
- Supports `-all_quants` to iterate over every `.gguf` file in a multi-quantization HuggingFace repo and emit all variants + `.f32` files into one `testdata/<model>/` directory in a single run.
- Extensible to GGUF generative LLMs (`-task=generate`) using `llama.Decode` + `llama.GetLogitsIth` for top-K logits and greedy generation.

---

## 8. Initial Target Model Roster

### `go-huggingface/modeltests`

| Model Directory | HuggingFace Repo | Format | Family / Tasks | Variants | Tier | Reference Generator |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baai-bge-small-en-v1.5` | `BAAI/bge-small-en-v1.5` | SafeTensors | Transformer / Tokenize + Embed | Default (FP32) | `small` | Python (`sentence-transformers`) |
| `sentence-transformers-all-minilm-l6-v2` | `sentence-transformers/all-MiniLM-L6-v2` | SafeTensors | Transformer / Tokenize + Embed | Default (FP32) | `small` | Python (`sentence-transformers`) |
| `bge-base-en-v1.5-gguf` | `ChristianAzinn/bge-base-en-v1.5-gguf` | GGUF | Transformer / Tokenize + Embed | `Q4_0` | `small` | Go `yzma` (`llama.cpp`) |
| `minilm-l6-v2-gguf` | `second-state/All-MiniLM-L6-v2-Embedding-GGUF` | GGUF | Transformer / Tokenize + Embed | **All 13 supported quants**: `Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_0`, `Q4_K_S`, `Q4_K_M`, `Q5_0`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0`, `F16` | `small` | Go `yzma` (`llama.cpp`) |
| `kalm-gemma3-12b` | `tencent/KaLM-Embedding-Gemma3-12B-2511` | SafeTensors | Transformer / Tokenize + Embed | Default (BF16) | `large` | Python (`sentence-transformers`) |
| `qwen3.5-4b` | `Qwen/Qwen3.5-4B` | SafeTensors | Transformer / Tokenize + Generate | Default (BF16) | `medium` | Python (`transformers`) |
| `gemma4-e4b` | `google/gemma-4-E4B-it` | SafeTensors | Transformer / Tokenize + Generate | Default (BF16, gated) | `medium` | Python (`transformers`) |
| `timestral` | `mistralai/Timestral-28M` & `123M` | SafeTensors | TimeSeries / Forecast | `28m`, `123m` | `small` | Python (PyTorch) |

### Known Issues Surfaced by the Initial Test Run

Running the initial suite uncovered three real issues — two in `tokenizers/hftokenizer` (now fixed) and one remaining in `compute/gobackend`:
1. **`compute/gobackend` Float16 (`minilm-l6-v2-gguf/F16`) [Open]**: Unpadded `seqLen = 32` diverges on `go` / `go:ops_sequential` (`cos = 0.787`), while `seqLen = 4, 9, 36` on `go` and all lengths on `xla:cpu` pass (`cos = 1.00000`).
2. ~~**`tokenizers/hftokenizer` ByteLevel BPE multi-space pre-tokenization (`qwen3.5-4b`)**~~ **[Fixed]**: Emulated `\s+(?!\S)` lookahead in `Split` pre-tokenizer and honored `ByteLevel` `use_regex: false` (`byteLevelEncodeNoSplit`).
3. ~~**`tokenizers/hftokenizer` `SpecialTokenID` lookups (`kalm-gemma3-12b`, `qwen3.5-4b`)**~~ **[Fixed]**: `resolveSpecialTokens` now searches `added_tokens` in addition to `model.vocab` for config-defined special tokens (`pad_token`, etc.) and falls back to non-special `<mask>`/`[MASK]` entries in `added_tokens`.

### Parallel `onnx-gomlx/modeltests` (Sibling Repository)

| Model Directory | HuggingFace Repo | Format | Tasks | Tier | Reference Generator |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `knightsanalytics-all-minilm-l6-v2` | `KnightsAnalytics/all-MiniLM-L6-v2` | ONNX | Tokenize + Embed | `small` | Go `onnxruntime_go` / Python ORT |

---

## 9. How to Add Support for a New Model (TDD Workflow)

Follow this workflow whenever adding a new model or architecture to `go-huggingface`:

### Step 1: Generate the Reference Golden Files
Run the appropriate reference generator targeting a new directory under `modeltests/testdata/<model-slug>/`:

```bash
# Example A: HuggingFace SafeTensors embedder
uv run modeltests/internal/generators/python/gen_embeddings.py \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --out modeltests/testdata/sentence-transformers-all-minilm-l6-v2

# Example B: HuggingFace generative LLM (with optional local layer dump for debugging)
uv run modeltests/internal/generators/python/gen_generate.py \
  --model Qwen/Qwen3.5-0.8B \
  --out modeltests/testdata/qwen3.5-0.8b \
  --dump-layers

# Example C: GGUF model via llama.cpp (yzma)
(cd modeltests/internal/generators/yzma && \
  go run . -model ChristianAzinn/bge-base-en-v1.5-gguf -quant Q4_0 \
  -out ../../../testdata/bge-base-en-v1.5-gguf)
```

Verify that `spec.json` includes:
- Pinned `revision` commit hash
- Complete `provenance` metadata
- Tokenizer `Encode` + `Decode` test cases from the reference runtime
- Appropriate `size_tier`, `tags`, and `tolerances`

### Step 2: Run the Targeted Model Test
```bash
go test -v -p=1 ./modeltests -modeltests -run TestModels/<model-slug>
```

### Step 3: Iterate Using Layered Diagnostics
Fix issues from the bottom up:
1. **`tokenize` fails**: Fix `tokenizers/hftokenizer` (normalizer, pre-tokenizer, model, post-processor, or decoder) or `models/gguf` tokenizer extraction.
2. **`variables` fails**: Fix weight name mapping / config parsing in `models/transformer` (`config.go`, `model.go`, `gguf.go`).
3. **`embed_*` or `prefill_logits` fails**: Re-run with `-debug_layers` to identify the exact first layer where activations diverge.
4. **`kvcache_parity` fails**: Inspect KV-cache, sliding-window, RoPE position offset, or linear-attention recurrent state updates.
5. **`greedy_generate` fails**: Check chat templates, BOS/EOS stop tokens, logit soft-capping, or tied/untied `lm_head` weights.

### Step 4: Test Across Backends
```bash
go test -v -p=1 ./modeltests -modeltests \
  -run TestModels/<model-slug> \
  -backends="go;xla:cpu;onnx:cpu"
```

### Step 5: Clean Up & Document
- Ensure any local `debug_layers/` artifacts are excluded by `.gitignore` and **not** committed.
- Add the new model to the Supported Models table in this document (or regenerate it once automated).
- Add an entry to [`docs/CHANGELOG.md`](./CHANGELOG.md).

---

## 10. Implementation Status & Roadmap

1. **Phase 1 — Core Harness & Shared Corpora (Done)**:
   - Created `modeltests/` with `-modeltests`, `-backends`, `-max_tier`, `-cached_only`, `-tag`, `-strict`, `-debug_layers`, `-set`.
   - Implemented `memory.go` / `memory_linux.go` (Linux `VmHWM` reset via `/proc/self/clear_refs` + portable RSS sampler) and end-of-run summary table.
   - Authored `testdata/reference_sentences.json` and `testdata/reference_prompts.json`.
   - Implemented `transformer_test.go` (tokenize, variables, embed single/batch, prefill top-K logits, KV-cache parity, greedy generation) and `timeseries_test.go`.
2. **Phase 2 — Reference Generators (Done)**:
   - Created `modeltests/internal/generators/python/` (`gen_embeddings.py`, `gen_generate.py`).
   - Created `modeltests/internal/generators/yzma/` (separate `go.mod`, adapted from `../benchmarks/yzma_embedder/`, using `llama.Tokenize`/`llama.Detokenize` and supporting `-all_quants`).
3. **Phase 3 — Populate Initial Model Roster (Done)**:
   - Generated specs & golden `.f32` files for `baai-bge-small-en-v1.5`, `sentence-transformers-all-minilm-l6-v2`, `bge-base-en-v1.5-gguf`, `minilm-l6-v2-gguf` (all 13 quantizations), `kalm-gemma3-12b`, `qwen3.5-4b`, `gemma4-e4b`, and `timestral` (28M & 123M).
4. **Phase 4 — Clean Up Redundant Example Tests (Done)**:
   - Removed redundant parity tests and golden `.txt`/`.f32` test files from `examples/BAAI-bge-small-en-v1.5`, `examples/minilm-l6-v2-gguf`, `examples/kalmgemma3`, and `examples/timestral` while preserving helper packages and runnable demos (`gemma4-e4bit/demo`, `dolly/browse`, `msmarco/benchmark_embed`, `timestral/main.go`).
   - Removed superseded `internal/testing/` and legacy `internal/py/{embed_sentence,tokenize_bge}.py`.
5. **Phase 5 — Parallel `onnx-gomlx/modeltests/` (Next)**:
   - Create a matching `modeltests/` suite inside `onnx-gomlx` (using the same `-modeltests`, `-backends`, `spec.json`, and `.f32` conventions), starting with `KnightsAnalytics/all-MiniLM-L6-v2` and other ONNX models.
