# Half-Precision Transformer Support Plan (`compute`, `gomlx`, `go-huggingface`)

Date: 2026-11-15

## Goal

Preserve half-precision (`Float16` and `BFloat16`) weights natively when loading half-precision GGUF models (and support native half-precision transformer execution end-to-end on CPU/accelerators), while keeping full `Float32` precision where required for numerical stability (matmul accumulation, softmax/attention reductions, normalization parameters/reductions, and L2 normalization).

Primary end-to-end validation target:

```bash
GOEXPERIMENT=simd GOMLX_BACKEND=go:quantized_cache go run github.com/gomlx/go-huggingface/examples/msmarco/benchmark_embed/ \
  -bucket=0 -model ChristianAzinn/bge-base-en-v1.5-gguf -limit 100 \
  -num_embedders=32 -warmup -dynamic=true -gguf_file=bge-base-en-v1.5_fp16.gguf
```

Quantized GGUF models (e.g., `Q4_0`, `Q4_K_M`, `Q8_0`) must continue to convert any remaining unquantized `F16`/`BF16` tensors (such as `position_embd.weight`) to `Float32`, and backends without half-precision support must still be able to opt into forcing `Float32`.

---

## 1. Investigation Findings

1. **GGUF tensor layout in native F16/BF16 models**:
   - Inspecting `ChristianAzinn/bge-base-en-v1.5-gguf/bge-base-en-v1.5_fp16.gguf` (`general.file_type = 1`, 197 tensors):
     - **74 `F16` tensors**: `token_embd.weight`, `position_embd.weight`, `blk.*.attn_{q,k,v,output}.weight`, `blk.*.ffn_{up,down}.weight`.
     - **123 `F32` tensors**: `token_types.weight`, all `*_norm.{weight,bias}`, all `attn_*.bias`, `ffn_*.bias`.
   - `second-state/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-ggml-model-f16.gguf` follows the same split (38 `F16` + 63 `F32`).
   - `models/gguf/reader.go` (`ReadTensor`) already loads `F16`/`BF16` directly as `dtypes.Float16`/`dtypes.BFloat16`; `models/transformer/gguf.go` (`loadStoreGGUF`, `loadSplitGGUFQKV`) and `models/transformer/transformer.go` (`CreateGoMLXModel`) unconditionally convert everything to `Float32`.
2. **`gomlx` mixed-dtype variable handling**:
   - `layers.NewDense`, `layers.Embedding`, and `pos.Learned` already check existing store variable dtypes and call `ConvertDType`.
   - `norm.LayerNorm` and `norm.RMSNorm` (plus `layer_scalar` in `ml/zoo/transformer/transformer.go`) unconditionally call `VariableWithShape(..., shapes.Make(x.DType(), ...))`, which panics on loaded `Float32` norm parameters when `x` is `Float16`/`BFloat16`.
   - `norm.RMSNorm` currently performs `Square`, `ReduceMean`, and `Sqrt` directly in `x.DType()` without upcasting 16-bit inputs to `Float32`.
3. **Half-precision `Float16` underflow/overflow hazards in `sentence.go`**:
   - `normalizeEmbeddings` computes `ReduceSum(Square(x), -1)` and `eps = 1e-12` in `x.DType()`. In IEEE `Float16`, `1e-12` underflows to `0.0` (min subnormal is `~5.96e-8`), and `ReduceSum(Square(x))` risks `Float16` overflow (`> 65504`).
4. **Backend support (`compute/internal/gobackend`)**:
   - `FusedDense` (including `RHSConstantData` caching and `Float32` bias), `FusedAttentionQKVProjection`, `FusedActivation`, `DotGeneral` (`Float32` accumulation), `Gather`, `Where`, `DynamicIota`, `UnaryOps`, `BinaryOps`, and `Reduce` already support `Float16`/`BFloat16`.
   - `FusedSoftmax` and `FusedScaledDotProductAttention` currently return `ErrNotImplemented` for `Float16`/`BFloat16`, causing fallback to decomposed `Float16` graph ops (allocating `[B, H, S, S]` score matrices and losing SIMD/Flash fusion and `Float32` softmax/attention accumulation).

---

## 2. Bottom-Up Implementation Plan

### Stage 1: `compute` — Half-Precision `FusedSoftmax` & `FusedScaledDotProductAttention`

1. **Interface Documentation (`compute/ops_fused.go`)**:
   - Explicitly document half-precision (`Float16` and `BFloat16`) semantics on `FusedSoftmax` and `FusedScaledDotProductAttention`:
     - Inputs/outputs stay in the caller's dtype (`Float32`, `Float64`, `Float16`, or `BFloat16`), preserving memory footprint.
     - Internal exponentials, reductions, `Q@K^T`, `P@V`, and softmax normalization for `Float16`/`BFloat16` may accumulate in `Float32` (preventing half-precision overflow/underflow) before rounding back to the input/output dtype.
     - For `FusedScaledDotProductAttention`: additive `Mask` / `Bias` (when non-nil and not `Bool`) use the query float dtype (or `ErrNotImplemented` if an unsupported dtype combination is requested).
     - Note other backends: `compute-onnx` already supports `Float16`/`BFloat16` for both ops; `go-xla` returns `ErrNotImplemented` for `FusedSoftmax` and supports accelerator-specific dtypes in `FusedScaledDotProductAttention` (returning `ErrNotImplemented` otherwise); `go-darwinml` supports MLX/MPS float dtypes or returns `ErrNotImplemented`.
2. **`FusedSoftmax` in `compute/internal/gobackend/fusedops` (`softmax.go`, `softmax_simd.go`)**:
   - Non-SIMD generic path (`softmax.go`): accept `dtypes.Float16` and `dtypes.BFloat16`, converting element values via `.Float32()` / `SetFloat32(...)` and performing max, `fastmath.Exp32`, sum, and normalization in `float32`.
   - SIMD path (`softmax_simd.go`, `innerSize == 1`): add fast `Float16`/`BFloat16` row processors converting vectors to `float32` via `float16.ToFloat32SIMD` / `bfloat16.ToFloat32SIMD`, accumulating max, vectorized `exp` + sum, and normalization in `float32` inside a per-worker reusable scratch buffer, and packing back with `FromFloat32SIMD` (decoding from memory once per row, encoding once per row).
3. **`FusedScaledDotProductAttention` in `compute/internal/gobackend/fusedops` (`sdpa.go`)**:
   - Update `buildSDPANode` and `execFusedScaledDotProductAttention` to accept `dtypes.Float16` and `dtypes.BFloat16` (supporting `Bool` masks and `Float16`/`BFloat16` additive masks/biases).
   - For each worker head/block task, unpack/convert `Q`, `K`, `V` (and optional additive `Mask`/`Bias` slices) into per-worker pooled `[]float32` scratch buffers (using vectorized SIMD `ToFloat32SIMD` conversions when `GOEXPERIMENT=simd` is active), execute the existing high-speed `float32` arch/SIMD SDPA kernel (`GetSDPAArchDispatcher()` / `sdpaGeneric[float32]`), and pack the `float32` output rows back into `Float16`/`BFloat16`.
4. **Tests & Micro-Benchmarks**:
   - Extend `compute/support/backendtest/fusedops.go` to test `FusedSoftmax` and `FusedScaledDotProductAttention` on `Float16` and `BFloat16` (1D/2D/trailing & non-trailing axis softmax, causal/mask/bias/GQA SDPA).
   - Add micro-benchmarks in `compute/internal/gobackend/fusedops` (and/or `backendtest/benchmarks.go`) comparing `Float32` vs `Float16` vs `BFloat16` across representative transformer shapes, and record results in Section 3 below.

---

### Stage 2: `gomlx` — Variable DType & Normalization Precision (`ml/layers/norm`, `ml/zoo/transformer`)

1. **`norm.LayerNorm` (`gomlx/ml/layers/norm/layernorm.go`)**:
   - Add `WithDType(dtype dtypes.DType) *LayerNormBuilder` (and `WithNormalizationDType(dtype dtypes.DType)` for programmatic override alongside `ParamLayerNormNormalizationDType`).
   - Behavior of `WithDType`:
     - By default (`InvalidDType`), variables (`"gain"`, `"offset"`) must match `x.DType()` (panicking on mismatch via `VariableWithShape`), catching unexpected dtype mismatches.
     - If `WithDType(d)` is explicitly set (e.g., `dtypes.Float32`), `"gain"`/`"offset"` are created with `d` and any existing variable in `scope.Store()` **must** match `d` (panicking otherwise).
     - In `Done()`, cast `gamma` and `beta` to `normalizationDType` (already `Float32` by default for 16-bit `x`), preserving full `Float32` precision when stored as `Float32`.
2. **`norm.RMSNorm` (`gomlx/ml/layers/norm/rmsnorm.go`)**:
   - Add `WithDType(dtype dtypes.DType) *RMSNormBuilder`, `WithNormalizationDType(dtype dtypes.DType) *RMSNormBuilder`, and `ParamRMSNormNormalizationDType = "rms_norm_dtype"`.
   - Promote 16-bit inputs (`Float16`, `BFloat16`) to `normalizationDType = dtypes.Float32` by default for `Square`, `ReduceMean`, `Sqrt`, division, and scaling, then cast back to `x.DType()` — matching `LayerNorm`.
   - Apply the same strict-by-default variable-dtype semantics as `LayerNorm` for `"scale"`: require `x.DType()` by default, or enforce an explicitly requested `WithDType(d)`.
3. **Transformer `WithNormDType` (`gomlx/ml/zoo/transformer/transformer.go`)**:
   - Add `Model.NormDType` / `Model.WithNormDType(dtype dtypes.DType)` and `ParamNormDType`. By default (`InvalidDType`), normalization layers require `m.DType` (catching unexpected mismatches, as does `layer_scalar`); setting `WithNormDType(dtypes.Float32)` explicitly opts into `Float32` normalization weights for a half-precision transformer.
   - Leave the existing heterogeneous-dtype `ConvertDType` support in `layers.NewDense` and `layers.Embedding` as is for now and document why (supporting pre-trained models/GGUFs with mixed parameter dtypes, to be tightened later if desired).
4. **Tests**:
   - Unit tests in `gomlx/ml/layers/norm` and `gomlx/ml/zoo/transformer` covering strict default dtype enforcement (panic on unexpected mismatch), explicit `WithDType` / `WithNormDType` mismatch validation, explicit `Float32` norm params + `Float16`/`BFloat16` activations, and `RMSNorm` 16-bit upcast stability.

---

### Stage 3: `go-huggingface` — Native Half-Precision GGUF & Sentence Embedding Stability

1. **Primary GGUF DType Detection & Force-F32 Option (`models/transformer`)**:
   - Add `ggufDTypes() (modelDType, normDType dtypes.DType)` in `models/transformer/gguf.go`:
     - If `m.ForceFloat32()` or any tensor in the GGUF is quantized (`ToGGMLQuantType()` succeeds), return `(dtypes.Float32, dtypes.Float32)` (keeping quantized weights via `QuantGGML` and promoting small remaining `F16`/`BF16` tensors like `position_embd.weight` to `Float32`).
     - Otherwise inspect mapped tensors by parameter count for primary 2D non-norm weights (`Float16` / `BFloat16` / `Float32`) and mapped norm weights (`gain` / `offset` / `scale`), resolving `normDType` separately (promoting mixed norm dtypes to `Float32` if needed).
   - Default to keeping half-precision models in their detected half-precision dtype (`tm.WithDType(modelDType)` in `CreateGoMLXModel`), and explicitly configure `tm.WithNormDType(normDType)` when `normDType != modelDType` (e.g., `Float32` normalization weights in llama.cpp `F16`/`BF16` exports).
   - Add `WithForceFloat32(force bool) *Model` / `ForceFloat32() bool` on `transformer.Model` so callers targeting backends without half-precision support can force `Float32` weights & execution for both GGUF and Safetensors models.
2. **Preserve Half-Precision Weights During Loading (`loadStoreGGUF`, `loadSplitGGUFQKV`)**:
   - In `loadStoreGGUF`: only call `ensureFloat32HostTensor` when `targetModelDType == dtypes.Float32` (quantized GGUFs, F32 GGUFs, or `forceFloat32 == true`), `Float64`, or mixed norm weights targeting `Float32`. Otherwise keep `Float16`/`BFloat16` weights and `Float32` norm weights natively in the store.
   - In `loadSplitGGUFQKV`: split unquantized fused QKV weights and biases natively via `splitTensorFirstAxis3` (`float32`, `float16.Float16`, `bfloat16.BFloat16`) unless `forceFloat32` requires `Float32`.
3. **Half-Precision Pooling & L2 Normalization (`models/transformer/sentence.go`)**:
   - Update `normalizeEmbeddings` and mean pooling reductions in `ApplySentencePooling` (`PoolingModeMeanTokens`) to promote `Float16`/`BFloat16` inputs to `Float32` before `Square`/`ReduceSum`/`Sqrt`/`eps`/`Div`/`MaskedReduceMean`, then cast back to the original dtype, avoiding `Float16` `1e-12` underflow and overflow.
4. **Tests & End-to-End Benchmarks**:
   - Added `TestGGUFHalfPrecisionEndToEnd` in `models/transformer/gguf_test.go` (synthetic `F16` and `BF16` GGUFs with `F32` norms, split QKV, and `WithForceFloat32(true)`) and `TestSentenceHalfPrecisionPoolingAndNormalize` in `models/transformer/sentence_test.go` (verifying large-magnitude non-overflow and zero-vector `1e-12` epsilon stability).
   - Benchmarked `ChristianAzinn/bge-base-en-v1.5-gguf` (`bge-base-en-v1.5_fp16.gguf`) on `go:quantized_cache` (`-limit 100 -bucket=0 -num_embedders=32 -warmup -dynamic=true`).

---

## 3. Stage 1 Micro-Benchmark Results: `FusedSoftmax`, `FusedScaledDotProductAttention`, and `FusedDense` (`Float32` vs `Float16` vs `BFloat16`)

Measured on `linux/amd64`, **AMD Ryzen 9 9950X3D 16-Core Processor** (`parallelism=32`), with `GOEXPERIMENT=simd` (`AVX512` + `AVX2` active, `-benchtime=500ms`):

```bash
cd /home/janpf/Projects/gomlx/compute
GOEXPERIMENT=simd go test ./gobackend -run='^$' -bench='BenchmarkCompliance/(Softmax|ScaledDotProductAttention)/' -benchtime=500ms
GOEXPERIMENT=simd go test ./gobackend -run='^$' -bench='BenchmarkCompliance/Dense/(all-MiniLM|bge-base|large)/[^/]*/(Float32|Float16|BFloat16)/Fused/None' -benchtime=500ms
```

### 3.1. `Softmax` (`FusedSoftmax` vs `Decomposed`)

| Shape (`axis`) | `Float32` Fused | `Float32` Decomp | `Float16` Fused (vs F32) | `Float16` Decomp (Fused Speedup) | `BFloat16` Fused (vs F32) | `BFloat16` Decomp (Fused Speedup) |
|---|---:|---:|---:|---:|---:|---:|
| `[8, 64]` (`axis=1`) | 1.23 µs | 2.23 µs | 1.54 µs (1.25×) | 4.45 µs (2.90×) | 1.62 µs (1.31×) | 2.92 µs (1.81×) |
| `[32, 128]` (`axis=1`) | 4.32 µs | 6.10 µs | 4.63 µs (1.07×) | 25.86 µs (5.58×) | 4.87 µs (1.13×) | 12.35 µs (2.54×) |
| `[64, 512]` (`axis=1`) | 31.51 µs | 37.88 µs | **29.36 µs (0.93×)** | 201.73 µs (6.87×) | 32.30 µs (1.02×) | 97.18 µs (3.01×) |
| `[8, 16, 64]` (`axis=2`) | 10.36 µs | 14.77 µs | 10.86 µs (1.05×) | 49.57 µs (4.57×) | 11.67 µs (1.13×) | 23.97 µs (2.05×) |
| `[12, 128, 128]` (`axis=2`) | 128.81 µs | 249.18 µs | **109.12 µs (0.85×)** | 986.27 µs (9.04×) | **110.59 µs (0.86×)** | 559.57 µs (5.06×) |
| `[4, 8, 32, 128]` (`axis=3`) | 103.29 µs | 148.62 µs | **93.04 µs (0.90×)** | 727.92 µs (7.82×) | **94.23 µs (0.91×)** | 380.77 µs (4.04×) |

Key takeaways for `FusedSoftmax`:
- Handwritten Plan 9 F16C `VCVTPH2PS`/`VCVTPS2PH` assembly (`internal/gobackend/avx512`, `internal/gobackend/avx2`) and chunked in-L1 `Float32` softmax cut `Float16` `[64, 512]` from `42.92 µs` down to **`29.36 µs`** (beating `Float32`'s `31.51 µs`), and `[12, 128, 128]` to **`109.12 µs`** (**9.04× faster** than decomposed `Float16`, and **15% faster** than `Float32`).

### 3.2. `FusedScaledDotProductAttention`

| Shape & Config | `Float32` | `Float16` (vs `Float32`) | `BFloat16` (vs `Float32`) |
|---|---:|---:|---:|
| `B=1, H=12, S=128, D=64` (`BHSD`) | 142.66 µs | **118.88 µs (0.83×)** | **127.42 µs (0.89×)** |
| `B=1, H=12, S=128, D=64` (`BSHD`) | 131.41 µs | **129.30 µs (0.98×)** | 135.96 µs (1.03×) |
| `B=8, H=12, S=128, D=64` (`BHSD`) | 406.28 µs | **361.33 µs (0.89×)** | **373.22 µs (0.92×)** |
| `B=1, H=6, S=384, D=64` (`BHSD`) | 218.45 µs | 225.11 µs (1.03×) | 238.47 µs (1.09×) |
| `B=4, H=8, KV=2, S=256, D=128` (`Causal`, `BHSD`) | 481.68 µs | 704.67 µs (1.46×) | 714.21 µs (1.48×) |

Key takeaways for `FusedScaledDotProductAttention`:
- Using reusable per-worker L1/L2 `Float32` scratch buffers in `sdpaMultiHeadHalf` (avoiding global `qF32`/`kF32`/`vF32`/`outF32` DRAM allocations and extra `ParallelizeChunks` barriers) made non-causal `Float16` and `BFloat16` encoder attention **faster than `Float32`** on `B=1, H=12, S=128, D=64` (**`118.9 µs` vs `142.7 µs`**) and `B=8, H=12, S=128, D=64` (**`361.3 µs` vs `406.3 µs`**, down from `709.3 µs`!).
- `OpTypeFusedScaledDotProductAttention` is enabled in `gobackend` ([`capabilities.go`](../../compute/internal/gobackend/capabilities.go) and [`sdpa.go`](../../compute/internal/gobackend/fusedops/sdpa.go)), so `Float16`/`BFloat16` attention ops now stay fused end-to-end instead of falling back to decomposed graph ops. Backends without half-precision fused support continue to return `compute.ErrNotImplemented` so GoMLX falls back cleanly.

### 3.3. `FusedDense` (`BiasAdd`, `Activation=None`, `M×K×N`)

| Shape | `Float32` | `Float16` (vs `Float32`) | `BFloat16` (vs `Float32`) |
|---|---:|---:|---:|
| `all-MiniLM` `416×384×1152` | 423.12 µs | **378.10 µs (0.89×)** | **400.05 µs (0.95×)** |
| `bge-base` `128×768×768` | 271.98 µs | **234.47 µs (0.86×)** | **190.35 µs (0.70×)** |
| `bge-base` `512×768×768` | 483.64 µs | **471.59 µs (0.98×)** | **468.58 µs (0.97×)** |
| `bge-base` `128×768×3072` | 824.57 µs | **572.93 µs (0.69×)** | **598.40 µs (0.73×)** |
| `large` `1024×1024×1024` | 1,068.83 µs | **1,065.29 µs (1.00×)** | 1,150.64 µs (1.08×) |

Key takeaways for `FusedDense`:
- Previously, `execFusedDenseHalf` allocated an unpooled `make([]float32, totalOut)` on every call and downcast outputs via a single-threaded scalar loop. Pooling the `Float32` accumulator buffer via `backend.GetBuffer` and converting outputs with `gobackend.ParallelConvertSlice` + Plan 9 F16C / SIMD converters unlocked the larger `Float16`/`BFloat16` matmul kernel blocks (`30%` faster on `bge-base` `128×768×3072` FFN projections and `14%–30%` faster on `128×768×768`).
- `MultiRHS` (`FusedAttentionQKVProjection`) natively supports `Float16` and `BFloat16` with `Float32` accumulation on `amd64` (`avx512` and `avx2`), with the same buffer/downcast optimization applied in `attentionqkv.go` (`arm64` NEON falls back cleanly to 3 `MatMul` calls).

---

## 4. Stage 3 End-to-End MS-MARCO Benchmark (`bge-base-en-v1.5_fp16.gguf` on `go:quantized_cache`)

Measured on `linux/amd64`, **AMD Ryzen 9 9950X3D 16-Core Processor** (`GOEXPERIMENT=simd`, `GOMLX_BACKEND=go:quantized_cache`, `AVX512` + `AVX2` active):

```bash
GOEXPERIMENT=simd GOMLX_BACKEND=go:quantized_cache go run github.com/gomlx/go-huggingface/examples/msmarco/benchmark_embed/ \
  -bucket=0 -model ChristianAzinn/bge-base-en-v1.5-gguf -limit 100 \
  -num_embedders=32 -warmup -dynamic=true -gguf_file=bge-base-en-v1.5_fp16.gguf
```

| Mode | Store Load Time | Warmup (1K queries) | Benchmark (1K passages, 76.8K tokens) | Throughput | Max RSS Mem |
|---|---:|---:|---:|---:|---:|
| Native `Float16` (`-force_f32=false`) | 75.7 ms | 995.4 ms | 5.7 s | **177 items/s (13.6K tokens/s)** | **894.4 MiB** |
| Forced `Float32` (`-force_f32=true`) | 331.8 ms | 1.5 s | 10.0 s | 100 items/s (7.7K tokens/s) | 1.2 GiB |

Native `Float16` execution on the `go` backend runs **1.77× faster** than upcasting to `Float32`, loads weights **4.4× faster**, and reduces peak RSS memory from 1.2 GiB to 894.4 MiB.
