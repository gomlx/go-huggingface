# GGUF Transformer Models Plan (`go-huggingface`, `gomlx`, `compute`)

Date: 2026-11-12

## Goal

Add end-to-end support for GGUF-based Transformer models in `go-huggingface` (backed by `FusedQuantizedDense` and `QuantizedEmbeddingLookup` in `compute` / `gomlx`), targeting `second-state/All-MiniLM-L6-v2-Embedding-GGUF` as our primary validation and benchmark model:

```bash
GOEXPERIMENT=simd GOMLX_BACKEND="go" go run github.com/gomlx/go-huggingface/examples/msmarco/benchmark_embed/ \
  -bucket=8192 -model second-state/All-MiniLM-L6-v2-Embedding-GGUF \
  -limit 1000 -num_embedders=1 -warmup -dynamic=true
```

---

## 1. Investigation Findings

### 1.1 Anatomy of `second-state/All-MiniLM-L6-v2-Embedding-GGUF`

Inspecting the repository metadata and the GGUF v3 headers of all 13 `.gguf` variants revealed several structural differences compared to standard Hugging Face `safetensors` / ONNX repositories (such as `sentence-transformers/all-MiniLM-L6-v2`):

1. **Files in repository**:
   - `config.json` (standard `BertModel` configuration: 6 layers, `hidden_size=384`, `intermediate_size=1536`, 12 attention heads, `vocab_size=30522`, `max_position_embeddings=512`, `hidden_act="gelu"`).
   - 13 `.gguf` weight files: `Q2_K`, `Q3_K_L`, `Q3_K_M`, `Q3_K_S`, `Q4_0`, `Q4_K_M`, `Q4_K_S`, `Q5_0`, `Q5_K_M`, `Q5_K_S`, `Q6_K`, `Q8_0`, and `ggml-model-f16`.
2. **Missing Hugging Face auxiliary files (must be synthesized from GGUF metadata)**:
   - **No `tokenizer.json`, `tokenizer_config.json`, or `vocab.txt`**: All 30,522 WordPiece tokens, scores, token types, and special token IDs (`[PAD]=0`, `[UNK]=100`, `[CLS]=101`, `[SEP]=102`, `[MASK]=103`) are embedded in GGUF `tokenizer.ggml.*` metadata.
   - **No `modules.json`, `1_Pooling/config.json`, or `config_sentence_transformers.json`**: Pooling is specified via GGUF metadata `bert.pooling_type = 1` (`LLAMA_POOLING_TYPE_MEAN`), and sentence-transformers / `llama.cpp` (`--embd-normalize 2`) apply L2 normalization after pooling.
3. **Multi-GGUF file selection**:
   - `models/gguf.Model.Load()` previously picked `files[:1]` alphabetically, selecting `all-MiniLM-L6-v2-Q2_K.gguf` ("not recommended; significant quality loss").
   - When multiple `.gguf` files exist in a repository and none is explicitly specified, we will default to community preference order starting with **`Q4_K_M`**, and expose `-gguf_file` / `Model.WithGGUFFile(...)` for explicit selection.

### 1.2 Mixed Quantization Types Across the 13 GGUF Files

Each `.gguf` file contains 101 tensors (`token_embd.weight`, `position_embd.weight`, `token_types.weight`, `token_embd_norm.{weight,bias}`, and `blk.{0..5}.*`). Because `hidden_size = 384` is not divisible by 256 (`QK_K = 256`), `llama-quantize` only uses 256-value K-quants on `ffn_down.weight` (`K = 1536`) and falls back to 32-value legacy block formats (`Q5_0`, `Q5_1`, `IQ4_NL`, `Q8_0`) on `K = 384` matrices:

| GGUF Variant(s) | Quantized Tensor Types Present | Notes |
| :--- | :--- | :--- |
| `Q4_K_M.gguf` (21.7 MB) | `Q5_0` (28), `Q4_K` (4), `Q6_K` (2), `Q8_0` (3) | Default selection; requires `GGMLQ5_0` |
| `Q4_K_S.gguf` (20.6 MB) | `Q5_0` (26), `Q5_1` (4), `Q4_K` (5), `Q5_K` (1), `Q8_0` (1) | Requires `GGMLQ5_0`, `GGMLQ5_1` |
| `Q5_K_M.gguf` (23.7 MB) | `Q5_1` (28), `Q5_K` (4), `Q6_K` (2), `Q8_0` (3) | Requires `GGMLQ5_1` |
| `Q5_K_S.gguf`, `Q5_0.gguf` | `Q5_1` / `Q5_0` + `Q5_K` + `Q8_0` | Requires `GGMLQ5_0`, `GGMLQ5_1` |
| `Q3_K_M.gguf`, `Q3_K_L.gguf` | `IQ4_NL` (18), `Q5_0` / `Q5_1` (12), `Q4_K` / `Q5_K` (6), `Q8_0` (1) | Requires `GGMLQ5_0`, `GGMLQ5_1` |
| `Q8_0.gguf` (36.6 MB) | `Q8_0` (37) | Already in `compute.GGMLQuantType` |
| `Q4_0.gguf` (20.9 MB) | `Q4_0` (36), `Q8_0` (1) | Already in `compute.GGMLQuantType` |
| `Q6_K.gguf` (27.4 MB) | `Q6_K` (6), `Q8_0` (31) | Already in `compute.GGMLQuantType` |
| `Q2_K.gguf`, `Q3_K_S.gguf` | `IQ4_NL` (30), `Q3_K` (6), `Q8_0` (1) | Already in `compute.GGMLQuantType` |
| `ggml-model-f16.gguf` (46.0 MB) | `F16` (38), `F32` (63) | Float weights (`FusedDense`) |

Adding **`GGMLQ4_1`**, **`GGMLQ5_0`**, and **`GGMLQ5_1`** to `compute` and `gomlx` makes **all 13 GGUF variants** run 100% natively through `FusedQuantizedDense` and `QuantizedEmbeddingLookup` without host-side weight expansion.

### 1.3 GELU Behavior in `llama.cpp` vs PyTorch / Hugging Face

- **GGUF / `llama.cpp`**: GGUF BERT files do not encode an activation key in metadata; `llama.cpp` (`src/models/bert.cpp` + `src/llama-graph.cpp`) hardcodes `LLM_FFN_GELU` -> `ggml_gelu()`, which evaluates the **`tanh`-approximate GELU**:
  $$\operatorname{GELU}_{\text{approx}}(x) = 0.5 x \left(1 + \tanh\left(\sqrt{\tfrac{2}{\pi}}(x + 0.044715 x^3)\right)\right)$$
  (whereas `LLM_FFN_GELU_ERF` -> `ggml_gelu_erf()` is only used by specific non-BERT architectures).
- **PyTorch / Hugging Face `transformers`**: `"hidden_act": "gelu"` in `BertConfig` evaluates the **exact `erf` GELU**:
  $$\operatorname{GELU}_{\text{exact}}(x) = 0.5 x \left(1 + \operatorname{erf}\left(\tfrac{x}{\sqrt{2}}\right)\right)$$
- **GoMLX / `go-huggingface` design**:
  - Both `ActivationGelu` (exact `erf`) and `ActivationGeluApproximate` (`tanh`) are already supported in `compute` (`FusedDense` and `FusedQuantizedDense`) and `gomlx/ml/zoo/transformer` (`Model.WithApproximateGelu(bool)`).
  - `mltransformer.Model` defaults `ApproximateGelu = true`, which **matches `llama.cpp`'s `ggml_gelu` implementation** and is faster on CPU SIMD.
  - We will expose `Model.WithApproximateGelu(bool)` on `go-huggingface/models/transformer.Model` (defaulting to `true`, with `WithApproximateGelu(false)` switching to exact `erf` GELU when comparing against PyTorch / `safetensors` / ONNX).

### 1.4 Additional Subtleties Found

1. **Mixed Float DTypes (`position_embd.weight` is `F16` in all 13 files)**:
   - Even in quantized GGUFs, `position_embd.weight` is stored as `Float16[512, 384]`, while `QuantizedGather` and `FusedQuantizedDense` (`QuantGGML`) produce `Float32` activations and norms/biases/`token_types.weight` are `Float32`.
   - Both `pos.Learned.PreEncode` / `layers.Embedding` and the GGUF loader should handle preloaded `Float16` lookup tables cleanly when compute activations are `Float32`.
2. **Dynamic Shapes (`-dynamic=true`) Bug in `nn.QuantizedGather`**:
   - `gomlx/ml/nn/quantized_gather.go` currently passes `useDecomposedAlternateVJP = true` to `InternalFusedOpCaller`, causing `ggml.EmbeddingLookupDecomposed` to run unconditionally at graph-build time and panic on `Reshape` when batch/sequence dimensions are `shapes.DynamicDim` (`-1`).
   - Because `QuantizedEmbeddingLookup` takes a `Uint8` table and integer `indices` (non-differentiable), `useDecomposedAlternateVJP` must be `false` (matching `nn.QuantizedDense`).

---

## 2. Implementation Plan

### Phase 1: Add `GGMLQ4_1`, `GGMLQ5_0`, and `GGMLQ5_1` in `compute`

1. **`compute/ops_fused.go`**:
   - Append `GGMLQ4_1`, `GGMLQ5_0`, and `GGMLQ5_1` to `GGMLQuantType` (preserving existing `0..8` enum values):
     - `GGMLQ4_1`: `ValuesPerBlock() = 32`, `BytesPerBlock() = 20`, `String() = "Q4_1"` (`fp16 d` [2B] + `fp16 m` [2B] + `qs` [16B]).
     - `GGMLQ5_0`: `ValuesPerBlock() = 32`, `BytesPerBlock() = 22`, `String() = "Q5_0"` (`fp16 d` [2B] + `uint32 qh` [4B] + `qs` [16B]).
     - `GGMLQ5_1`: `ValuesPerBlock() = 32`, `BytesPerBlock() = 24`, `String() = "Q5_1"` (`fp16 d` [2B] + `fp16 m` [2B] + `uint32 qh` [4B] + `qs` [16B]).
   - Because `ValuesPerBlock() == 32`, `shapeinference.validateGGMLActivationPairing` automatically pairs all three with `GGMLQ8_0` activations.
2. **`compute/internal/gobackend/fusedops/quantizeddense_ggml.go`**:
   - Add row dequantizers (`dequantQ4_1Row`, `dequantQ5_0Row`, `dequantQ5_1Row`) to `ggmlDequantFunc` (used by `QuantizedEmbeddingLookup`, `FusedQuantizedDense` Mode A float, and `FusedQuantizedDenseVJP`):
     - `Q4_1`: $y_j = d \cdot (qs_j \mathbin{\&} \text{0xF}) + m$, $y_{j+16} = d \cdot (qs_j \gg 4) + m$.
     - `Q5_0`: 5th bits from `qh = uint32_le(block[2:6])` (`xh0 = ((qh >> j) << 4) & 0x10`, `xh1 = (qh >> (j+12)) & 0x10`), signed $q = \text{int32}((nibble \mid xh)) - 16$, $y = d \cdot q$.
     - `Q5_1`: same 5-bit unpacking (`0..31` unsigned), $y = d \cdot q + m$.
   - Add paired integer dot-product kernels (`vecDotQ4_1_Q8_0`, `vecDotQ5_0_Q8_0`, `vecDotQ5_1_Q8_0`) to `ggmlVecDotFunc` (used by `FusedQuantizedDense` Mode B1 pre-quantized and Mode B2 `QuantizedMatmuls: true`):
     - Symmetric (`Q5_0`): $\text{acc} \mathrel{+}= (d_w \cdot d_x) \sum_{k=0}^{31} q_{w,k} q_{x,k}$.
     - Asymmetric (`Q4_1`, `Q5_1`): accumulate integer dot $\text{sumi} = \sum_{k=0}^{31} q_{w,k} q_{x,k}$ and integer activation sum $\text{sumx} = \sum_{k=0}^{31} q_{x,k}$ per 32-element block, then $\text{acc} \mathrel{+}= (d_w \cdot d_x)\,\text{sumi} + (m_w \cdot d_x)\,\text{sumx}$.
3. **`compute/support/backendtest/quantized_dense_ggml.go` & `shapeinference/quantized_dense_test.go`**:
   - Extend reference block builders and test suites to cover `GGMLQ4_1`, `GGMLQ5_0`, and `GGMLQ5_1` across Mode A, Mode B1, Mode B2 (`QuantizedMatmuls: true`), VJP, and `QuantizedEmbeddingLookup`.

---

### Phase 2: Core Quantized Layer Hooks & Dynamic-Shape Fix in `gomlx`

1. **`gomlx/ml/ggml/dequant.go`**:
   - Add graph-level decomposed `dequantQ4_1`, `dequantQ5_0`, and `dequantQ5_1` (`CanDecompose == true`) using `Slice`, `Bitcast`, `BitwiseAnd`, and `BitwiseShiftRightLogical` so non-fused backends (e.g. XLA) also get automatic fallback for 32-element GGML types.
2. **`gomlx/ml/nn/quantized_gather.go`**:
   - Fix `-dynamic=true` panic by checking `FusionEnv` and `backend.Capabilities().Operations[compute.OpTypeQuantizedEmbeddingLookup]`, and passing `useDecomposedAlternateVJP = false` to `InternalFusedOpCaller`.
3. **`gomlx/ml/model/variable.go`**:
   - Add optional `*Quantization` metadata on `model.Variable` (`Scheme`, `GGMLType`, optional `Scale`/`ZeroPoint` `*Variable`, `BlockAxis`, `BlockSize`), preserved across `CloneToStore`, with `Variable.WithQuantization(...)` and `Variable.Quantization()`.
4. **`gomlx/ml/layers` & `gomlx/ml/zoo/transformer`**:
   - **`layers.NewDense` (`*layers.DenseBuilder`) & `fnn.New`**:
     - Check `scope.GetVariable("weights")` first. When `weightsVar != nil && weightsVar.Quantization() != nil`, dispatch to `nn.QuantizedDense` with `WeightLayout: layout`, `WeightQuantization`, optional fused `WithActivation(act)`, and `QuantizedMatmuls: model.GetParamOr(scope, ParamQuantizedMatmuls, true)`.
     - Note: `QuantizedMatmuls: true` (`XQuantization: nil`) invokes `gobackend` Mode B2, which quantizes `x` into pooled `Q8_0` / `Q8_K` scratch buffers inside `FusedQuantizedDense` without graph-level intermediate allocations.
   - **`layers.Embedding` & `pos.Learned.PreEncode`**:
     - In `layers.Embedding`: when `scope.GetVariable("embeddings")` has `QuantGGML` quantization, call `nn.QuantizedGather` and cast output to `dtype` if needed; if preloaded float dtype differs from `dtype`, cast output cleanly.
     - In `pos.Learned.PreEncode`: reuse an existing `pos_embed/embeddings` variable even if its float dtype differs from `x.DType()` (e.g. `Float16` in GGUF while `x` is `Float32`), casting after `Gather`.
   - **`mltransformer.Model`**:
     - Use `m.dense` in `LogitsFromEmbeddings` when weights are transposed or quantized.
     - Fuse `m.Activation` into `ff1` in `forwardLayerStandard` when dropout is unused/inference so `FusedQuantizedDense` / `FusedDense` fuse GELU in-place.

---

### Phase 3: Embedded GGUF Tokenizer in `go-huggingface`

Respecting the strict modularity boundary (`tokenizers` **must never** import `gomlx` or `compute`):

1. **`tokenizers/hftokenizer`**:
   - Refactor `NewFromContent` to delegate to `NewFromStruct(config *api.Config, tj *TokenizerJSON) (*Tokenizer, error)`.
   - Add pure-Go `NewFromGGUF(params GGUFTokenizerParams) (*Tokenizer, error)` taking plain Go types (`Model`, `Pre`, `Tokens`, `Scores`, `TokenTypes`, `Merges`, special token IDs, `AddBosToken`, `AddEosToken`, `DoLowerCase`) and synthesizing the `TokenizerJSON` pipeline for:
     - **`"bert"` (WordPiece)**: `BertNormalizer` (`CleanText: true`, `HandleChineseChars: true`, `Lowercase: doLowerCase`), `BertPreTokenizer`, `WordPiece` (`ContinuingSubwordPrefix: "##"`, `MaxInputCharsPerWord: 100`), `AddedTokens` for `token_type == 3` control tokens, and `[CLS] $A [SEP]` `TemplateProcessing`.
     - **`"gpt2"` (Byte-level BPE)**: `BPE` with `Tokens` + `Merges` and `ByteLevel` pre-tokenizer/decoder.
     - **`"llama"` (SentencePiece / Unigram)**: `Unigram` with `Tokens` + `Scores` and `Metaspace` (`▁`) pre-tokenizer/decoder.
2. **`models/gguf` & `models/transformer`**:
   - Add `gguf.File.TokenizerParams()` extracting `tokenizer.ggml.*` metadata (including GGUF's historical `tokenizer.ggml.seperator_token_id` spelling alongside `separator_token_id`).
   - Update `transformer.Model.GetTokenizer()` to fall back to constructing the tokenizer from the GGUF header when `tokenizer.json` / `tokenizer_config.json` are absent in `m.Repo`.

---

### Phase 4: GGUF Model Discovery, Config Synthesis & Weight Loading in `go-huggingface`

1. **`models/gguf/tensor_type.go` & `models/gguf/model.go`**:
   - Map `TensorTypeQ4_1 -> compute.GGMLQ4_1`, `TensorTypeQ5_0 -> compute.GGMLQ5_0`, `TensorTypeQ5_1 -> compute.GGMLQ5_1` in `ToGGMLQuantType()`.
   - Add deterministic GGUF file selection (`SelectGGUFFile(files []string) string`) when multiple `.gguf` files exist in a repo and no explicit file was set:
     - Preference order: **`Q4_K_M`** -> `Q4_K_S` -> `Q5_K_M` -> `Q8_0` -> `Q6_K` -> `Q5_0` -> `Q4_0` -> `Q3_K_M` -> `Q3_K_L` -> `Q3_K_S` -> `Q2_K` -> `F16` -> `BF16` -> `F32` -> first `.gguf`.
2. **`models/transformer` (`model.go`, `transformer.go`, new `gguf.go`)**:
   - Add options on `transformer.Model`:
     - `WithGGUFFile(filename string) *Model` (selects a specific `.gguf` file in multi-quant repos).
     - `WithApproximateGelu(approx bool) *Model` (defaults to `true`, matching `llama.cpp`'s `ggml_gelu`; set `false` for exact `erf` GELU matching PyTorch/ONNX).
   - Support loading `transformer.Model` both from GGUF repos **with** `config.json` (like `second-state/All-MiniLM-L6-v2-Embedding-GGUF`) and **pure-GGUF repos without `config.json`** by reading `<arch>.block_count`, `<arch>.embedding_length`, `<arch>.feed_forward_length`, `<arch>.attention.*`, `<arch>.context_length`, `<arch>.rope.*`, and `<arch>.pooling_type` (`1`=mean, `2`=cls, `3`=last).
   - For GGUF embedding models (`pooling_type > 0`) when `modules.json` is absent, synthesize `[Transformer, Pooling, Normalize]` so output embeddings are L2-normalized.
   - Use `dtypes.Float32` activation compute dtype in `CreateGoMLXModel` when backed by GGUF.
   - Implement `loadStoreGGUF` + architecture-aware `mapGGUFTensorName`:
     - Map `token_embd`, `position_embd`, `token_types`, `token_embd_norm`, `output_norm`, `output`, and `blk.{N}.*` tensors for both **BERT encoder** (`attn`, `norm1`, `ff1`, `ff2`, `norm2`) and **Gemma/LLaMA/Qwen decoder** (`input_norm`, `self_attn`, `pre_feedforward_norm`, `mlp/{gate_proj,up_proj,down_proj}`, `final_norm`) architectures.
     - Load 2D quantized tensors with `ToGGMLQuantType() == true` via `ReadTensorRaw` (`Uint8[N, bytesPerRow]`) + `Variable.WithQuantization(&model.Quantization{Scheme: compute.QuantGGML, GGMLType: ggmlType})`.
     - Load float tensors (converting `F16` 1D/embedding tables to `Float32` when compute dtype is `Float32`) and any unrecognized quant formats via `ReadTensor`.

---

### Phase 5: `benchmark_embed` CLI Integration & Verification

1. **`examples/msmarco/benchmark_embed/main.go`**:
   - Detect GGUF repositories (`repoHasGGUF(repo)`) alongside `repoHasSafetensors(repo)` in `createEmbedExec` so GGUF models run through `hfModel.LoadStore` + `hfModel.SentenceEmbeddingGraph` with `DonateStore()` and `-dynamic=true`.
   - Add `-gguf_file` flag (documented with the `Q4_K_M`-first default selection order) and optional `-approx_gelu` flag (default `true`).
   - Log the selected `.gguf` file and total loaded variable bytes.
2. **Tests & Verification**:
   - **`compute` tests**: `go test ./shapeinference/... ./internal/gobackend/...` verifying `GGMLQ4_1`, `GGMLQ5_0`, `GGMLQ5_1` (with and without `GOEXPERIMENT=simd`).
   - **`gomlx` tests**: `go test ./ml/ggml/... ./ml/nn/... ./ml/layers/... ./ml/zoo/transformer/...` verifying decomposed `Q4_1`/`Q5_0`/`Q5_1`, `QuantizedGather` with dynamic shapes, and quantized `layers.DenseWithLayout` / `layers.Embedding`.
   - **`go-huggingface` tests & `llama.cpp` (`yzma`) ground-truth validation**:
     - **Standalone `yzma` reference generator (outside `go-huggingface`, no dependency added to `go-huggingface/go.mod`)**:
       - Build a temporary standalone Go module using [`github.com/hybridgroup/yzma`](https://github.com/hybridgroup/yzma) (`pkg/llama`, purego/FFI wrapper over `llama.cpp`) in a scratch/benchmark directory outside `go-huggingface`.
       - Run `all-MiniLM-L6-v2-Q4_K_M.gguf` through `llama.cpp` (`PoolingTypeMean` + L2 normalization) on a reference prompt/passage set to export exact `llama.cpp` token IDs and materialized `float32` embeddings (and keep the standalone harness available for future throughput comparisons against `llama.cpp`).
     - **Committed materialized test in `go-huggingface` (`examples/minilm-l6-v2-gguf/`)**:
       - Mirroring `examples/BAAI-bge-small-en-v1.5/`, check in the materialized `llama.cpp` reference embeddings (`similarity_embeddings.txt`) and token IDs alongside `minilm_test.go` using `internal/testing.ReadPythonEmbeddingsList` and `ValidateEmbeddingTensor` — zero `yzma` dependency in `go-huggingface`.
     - **Unit tests**:
       - `hftokenizer.NewFromGGUF` exact token ID parity against both `llama.cpp` (`yzma`) and `sentence-transformers/all-MiniLM-L6-v2` `tokenizer.json`.
       - Synthetic GGUF transformer unit tests (`-short`-safe) covering `LoadStore` and `SentenceEmbeddingGraph`.
   - **Target Benchmark Execution**:
     ```bash
     GOEXPERIMENT=simd GOMLX_BACKEND="go" go run github.com/gomlx/go-huggingface/examples/msmarco/benchmark_embed/ \
       -bucket=8192 -model second-state/All-MiniLM-L6-v2-Embedding-GGUF \
       -limit 1000 -num_embedders=1 -warmup -dynamic=true
     ```
   - Update `docs/CHANGELOG.md` and `.agents/AGENTS.md` across touched repos without committing.
