# `go-huggingface` Changelog

## Unreleased

- 2026/11/08:
  - `models/timestral`, `examples/timestral`:
    - Added `Model.WithForceFloat32(forceFloat32 bool) *Model` and `Model.ForceFloat32() bool` to use `dtypes.Float32` instead of `dtypes.Float64` for internal Welford running statistics (`ComputeCausalStatistics` and `AdvancingBlockFrames`).
    - Automatically detect backends that do not support `dtypes.Float64` (such as Apple Metal / `mps`), force `Float32` statistics, and log a `klog.Warningf` suggesting `WithForceFloat32(true)`.
    - Added `-force_float32` flag to `./examples/timestral`.
  - `internal/downloader`:
    - Added an explanatory hint on HTTP 401 (`StatusUnauthorized`) responses ("Hugging Face API suggests either an invalid username/password or a non-existent/private repository -- check that the model name is correct, or HF_TOKEN").
  - `hub`:
    - Default `Repo.authToken` in `hub.New` to the `HF_TOKEN` environment variable when set (can still be overridden or cleared via `Repo.WithAuth`).
  - `examples/msmarco/benchmark_embed`:
    - Added `-test_backend` flag to compare embeddings between the default backend (`GOMLX_BACKEND`) and a secondary backend under test.
    - Refactored reporting into `SpeedStats` / `ReportSpeed` and `DiffStats` / `ReportDifferences`, displaying a `Metric | Mean | Max` comparison table covering element-wise absolute difference (MAE), element-wise root mean square error (RMSE), passage Euclidean distance ($L_2$), passage relative $L_2$, and passage cosine distance ($1 - \cos(u, v)$) across non-padded sentences.
    - Increased `model.Exec` compilation cache limit to 64 (`SetMaxCache(64)`).

- 2026/11/06:
  - `internal/downloader`:
    - Validated downloaded byte count against `resp.ContentLength`. When `downloadedBytes != contentLength` or upon `io.ErrUnexpectedEOF`, `Manager.Download` now aborts with a descriptive error including both the downloaded byte count and expected `Content-Length`, ensuring incomplete `.part` files are removed rather than promoted to final files.

- 2026/11/05:
  - `models/gguf` (#18):
    - Added `TensorType.ToGGMLQuantType()` to map GGUF quantization formats (`Q4_0`, `Q8_0`, `IQ4_NL`, `Q2_K`–`Q6_K`) to `compute.GGMLQuantType`.
    - Added `TensorInfo.GoMLXRawShape()`, `Reader.ReadTensorRaw(backend, name)`, and `Model.ReadTensorRaw(backend, name)` to load un-dequantized tensors as `dtypes.Uint8` GoMLX tensors in native GGML block layout (`[..., bytesPerRow]`), plus `Reader.ReadTensorBytes(name)` and `Model.ReadTensorBytes(name)` to read raw `[]byte` slices.
    - Renamed `Model.GetTensor` to `Model.ReadTensor` (keeping `GetTensor` as a deprecated alias).
    - Added multi-file GGUF support (`Model.LoadAll()`, `Model.LoadFiles()`, `Model.ListGGUFFiles()`, `Model.ExtraFiles()`, `Model.GetTensorInfo()`) for multimodal repositories (such as LLaVA) that split weights across multiple `.gguf` files.
  - `tokenizers/api`, `tokenizers/hftokenizer` (#18):
    - Updated `api.TokensDecoder` (with `.String() string` and custom `UnmarshalJSON`) to unmarshal special token fields in `tokenizer_config.json` that may be either plain strings or `AddedToken` JSON objects (`{"content": "...", "lstrip": ..., ...}`), preserving the extra token attributes. **Note (minor API change)**: `Config.ClsToken`, `UnkToken`, `SepToken`, `MaskToken`, `BosToken`, `EosToken`, and `PadToken` are now `api.TokensDecoder` instead of `string` (use `.String()` or `.Content` to get the token string).

- 2026/11/02:
  - `examples/msmarco/benchmark_embed`:
    - Added support for ONNX models via `github.com/gomlx/onnx-gomlx` when `.safetensors` files are not available in the model repository (e.g. `KnightsAnalytics/all-MiniLM-L6-v2`).
    - Added `CPU Time` and `~Cores` rows to the benchmark report table showing total CPU time (user + system), CPU time per passage (`cpu_time / passages`), and average core utilization (`cpu_time / wall_time`).
  - `models/transformer`:
    - Added `Model.SentenceEmbeddingFromLastHiddenState(lastHiddenState, seqLen)` to apply post-Transformer sentence embedding modules (pooling and L2 normalization) to an existing hidden-state tensor.
    - Default `PoolingConfig` to mean pooling when `sentence_transformers.models.Pooling` is declared in `modules.json` without a `1_Pooling/config.json` file.
  - `models/timestral`:
    - Fused attention `_gate` projection (`Dense` + `Sigmoid`) into a single `nn.Dense(..., activation.TypeSigmoid)` (`FusedDense` with `ActivationSigmoid`).
    - Swapped `w12` `[gate; up]` halves to `[up; gate]` at weight-load time and replaced manual `Slice` + `Silu` + `Mul` in `FusedSwiGLU` with `activation.SwiGLU(w12)`.
  - `examples/timestral`: Format elapsed durations using `humanize.Duration()`.

- 2026/11/01:
  - `models/safetensors`, `models/timestral`, `models/transformer`, `models/sam2`: Added `Model.WithMMap(useMMap bool) *Model`, `Model.UseMMap() bool`, `safetensors.DefaultUseMMap()`, and `safetensors.IterTensorsFromRepoWithMMap(...)` to optionally read tensors via standard file I/O (`os.File.ReadAt`) instead of `mmap`. When `useMMap` is `false`, tensor bytes are read directly from the OS page cache into the destination tensor buffer (for host and shared-buffer backends), avoiding mapping weight files into the process's virtual address space and inflating Resident Set Size (RSS). Also fixed a missing `reader.Close()` in `safetensors.Model.GetTensorFromFile`.
  - `examples/timestral`: Added `-mmap` boolean flag to toggle `mmap` vs. standard file I/O when loading safetensors weights.

- 2026/10/31:
  - `models/timestral`: Added native GoMLX support for Mistral AI's Timestral time-series foundation models (`mistralai/Timestral-5M`, `mistralai/Timestral-28M`, `mistralai/Timestral-123M`, `mistralai/Timestral-209M`, and any compatible `*hub.Repo`), including causal Welford per-patch scaling with `asinh`, Decoupled Horizon Forecasting (DHF) two-stream time/variate attention masks, partial Sin/Cos RoPE, monotone softplus quantile heads, advancing block frame refinement, block-autoregressive decoding, `clampsym` symmetric averaging, and numerical/categorical covariates (`TargetEncode`).
  - `examples/timestral`: Added CLI tool and PyTorch reference accuracy test for native GoMLX Timestral forecasting.

- 2026/10/30:
  - `hub`: Resolve symlinks via `filepath.EvalSymlinks` in `Repo.Save` / `Repo.SaveCtx` when `linkOnly=true` so hard links point to the underlying blob files rather than copying relative snapshot symlinks.

- 2026/10/12:
  - `models/transformer`: Support saving and reloading fine-tuned models with GoMLX checkpoints and embedded filesystems.
    - Added `Model.SaveConfigs(dir)` and `Model.SaveToCheckpoint(checkpointDir)` to export model and tokenizer configurations into a standalone directory or into the `hf/` sub-directory of a GoMLX checkpoint.
    - Added `LoadModelFromCheckpoint(checkpointDir)` to reload model configurations directly from a checkpoint directory without needing to re-download base weights.
    - Added `LoadModelFromFS(fsys, subDir)` to load model configurations directly from in-memory and embedded filesystems (`embed.FS`, `*zip.Reader`, etc.).
    - Updated `LoadStoreFiltered` to check if a variable is already loaded in `model.Store` (e.g. from a checkpoint), preventing fine-tuned weights from being overwritten by base model safetensors and finalizing unused tensors to avoid memory leaks.

- 2026/10/10:
  - `tokenizers/hftokenizer`: Fix token byte spans drifting when using Replace normalizer (fixes #67).
    - Modularized normalization logic into `normalizer.go` and renamed `WithSpan` to `WithOffsets` to clarify the normalized-to-original byte mapping.
    - Added `NormalizationType` enum with string conversion functions.
    - Added `applyReplaceNormalizerWithOffsets` to accurately track byte offsets during string/regex replacement.
    - Propagated byte offsets into `wordWithOffset` and updated subword tokenizers (`BPE`, `WordPiece`, `Unigram`) to compute token spans via `spanForSubword`.
    - Split test suite: moved normalization tests to `normalizer_test.go`, pre-tokenization tests to `pretokenizer_test.go`, and post-processing tests to `postprocessor_test.go`.
  - `models/transformer`: Fix `LoadModel` leaving causal mask off by default (fixes #65). `LoadModel` now automatically detects whether to use a causal mask based on the model configuration (true for decoders like Gemma, LLaMA, or KaLM; false for bidirectional encoders like BERT, falling back to true if unknown). Added `Model.UseCausalMask()`, package-level documentation, and updated/harmonized documentation and comments across the codebase.

- 2026/09/15:
  - Cache and offline reliability improvements in `hub`:
    - Avoid re-downloading model revision info when it is already cached by default. Documented update polling via `DownloadInfo(true)`.
    - Short-circuit revision lookup when `revision` is already a 40-character commit hash.
    - Removed unused `revisionHashRefreshed` field.
    - Fixed `LockedDownload` to preserve existing cached files until a forced re-download successfully completes and atomically replaces them, avoiding cache deletion on network failure (fixes #59).
  - Replaced `Semaphore` with `FIFOSemaphore` in `internal/downloader`: added support for `context.Context` cancellation during semaphore acquisition and ensured strict FIFO ordering (fixes #60).
  - Fixed error reporting in `internal/downloader`: correctly wrap `readErr` instead of nil when an error occurs while reading the response body (fixes #61).

- 2026/08/28:
  - Unified sentence pooling in `models/transformer/sentence.go` to use `DimensionSpecFor`, `DimensionSize`, `DynamicBroadcastInDim`, and `DynamicIota` without conditional shape branching.

- 2026/07/30: 
  - Added local-directory mode to `hub`: `hub.NewLocal(dir)` / `Repo.WithLocalDir(dir)` let a `*hub.Repo` read
    files straight from a plain local directory (e.g. a `git clone` or `huggingface-cli download --local-dir`
    output) instead of downloading from HuggingFace Hub. No network access is made in this mode; every consumer
    of `*hub.Repo` (`models/transformer`, `models/safetensors`, `models/gguf`, `models/sam2`, `tokenizers/*`)
    works unchanged.
  - Added embedded-filesystem mode to `hub`: `hub.NewEmbed(fsys, subDir)` / `Repo.WithEmbedFS(fsys, subDir)` let a `*hub.Repo`
    read files directly from a Go `fs.FS` (e.g. `//go:embed` binaries) without requiring network access.
  - Added `Repo.IsEmbed()` and `Repo.IsLocal()` helper methods.
  - Added `Repo.Open(name)` and `Repo.ReadFile(name)` stream reading methods that work across all repository modes
    (remote, local, and embedded) without requiring disk file extraction in embedded mode.
  - Added `Repo.FetchFiles()` and `Repo.FetchFile()` methods to pre-download and cache repository files upfront without returning OS disk paths.
  - Added `Repo.Save(dirPath, linkOnly)` and `Repo.SaveCtx(ctx, dirPath, linkOnly)` methods to save/copy/hard-link
    the files of a repository to a local directory.
  - Added `Repo.DeleteCache()`.
  - Added flags `-local`, `-save`, `-link_only` and `-delete_cache` to CLI tool `cmd/hubinfo`.
  
## v0.4.0: Updated to GoMLX v0.28.0, added SAM2 and Gemma4; add CLI tool `cmd/hubinfo`.

- #46, #47: Expanded the HuggingFace repository metadata retrieval to support detailed file/LFS size information and adds a new command-line tool `cmd/hubinfo` to display this metadata in the terminal.
  - Added `hub.Repo.IterFileInfos()` to iterate over files with complete `FileInfo` records without triggering file downloads.
  - **New CLI Tool (`cmd/hubinfo`):**
- Improve support for large datasets, on-demand downloads, and CLI styling`
- Refreshed README.md, with fixes and updates to new features. Including fixing of the accompanying `go-huggingface.ipynb` notebook.
- Added SAM2 and Gemma4 models support.

## v0.3.5 Added `datasets`, `tokenizers/bucket`, significant improvements on `transformer` and `safetensors`.

- Added `.github` with continuous integration (CI) workflow: only for Linux/amd64 because it should work the
  same on other platforms.
- Package `tokenizers`: **API change!**
  - Updated `api.Tokenizer` interface: added `EncodeWithAnnotations`, `VocabSize()`, `Normalize()` and `Config` methods,
    and cleaned up the API.
- Package `tokenizers/hftokenizer`:
  - Added support to `AddBosToken` and `AddEosToken`.
- Package `tokenizers/sentencepiece`:
  - Added post-processing.
  - Spans only generated for `EncodeWithAnnotations`.
  - Added support to `AddBosToken` and `AddEosToken`.
- Package `tokenizers/bucket`
  - Added `bucket` package for streaming tokenization of sentences into buckets (or batches) of discrete sizes, 
    to minimize padding.
  - Added "Two-Bits Bucketing" strategy.
- Package `datasets`:
  - Added `datasets` package for downloading and iterating over parquet files of datasets from the HuggingFace Hub.
  - Added `cmd/generate_dataset_structs` for generating Go structs for dataset records.
  - Added `ParquetFixListSchema` for fixing list schema parsed from Go struct (a bug? in parquet-go where it hard-codes
    the group/element node names in lists).
  - Added `CreateParquetReader` for creating a parquet reader for the given dataset, config and split.
    Less convenient than an iterator, but allows for random access.
- Package `transformer`
  - Renamed main method to `AllLayers`: it returns both the final hidden state and all layer outputs; 
    it added RoPE positional embeddings support; added support for scaling factor.
  - Updates to modified GoMLX transformer API.
  - Added support for configured task prompts (via `task_prompts.json`):
    - `QueryPrompt` builds the full query prompt, based on a task code.
    - `RegisteredPromptTasks` returns a list of all task codes for which prompts are registered.
    - `GetTaskPrompt` returns the prompt string for the given task code.
  - `LoadContext` now accepts an optional (nil-lable) backend and loads the variables directly into the backend.
  - Fixed mask support for batches with padding.
- Package `safetensors`:
  - `IterTensors` and `IterTensorsFromRepo` now take an optional (nil-lable) backend for reading tensors directly into
    a backend.
  - `IterTensors` and `IterTensorsFromRepo` implemented with `github.com/edsrzf/mmap-go` (to avoid one kernel buffer copy to normal memory).
  - `IterTensorsFromRepo` uses now a parallelized
    pipeline (including parallelizing transfers to the GPU, to ensure XLA will optimize them in a queue) for loading model throughput.
  - Loading of KaLM-Gemma3 12B model (with 22 GB of weights) now takes ~1.5s where previously it took ~8.5s (and slightly faster than in Python using ~1.7s).
- Examples (`/examples/...`):
  - Added [Tecent's KaLM-Gemma3 12B model](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511) sentence embedder example.
  - Added [MSMARCO Dataset](https://huggingface.co/datasets/microsoft/ms_marco) example.
  - Added [BAAI (Beijing Academy of Artificial Intelligence) BGE Small Sentence Embedder (English) v1.5](https://github.com/gomlx/go-huggingface/tree/main/examples/BAAI-bge-small-en-v1.5): a small and very performatic sentence embedder (BERT based).
  - Added [MSMARCO `benchmark_embed`](https://github.com/gomlx/go-huggingface/tree/main/examples/msmarco/benchmark_embed/) 
    command-line benchmark of sentence embeddings, that also serves as an example of how to use the library.

## v0.3.4

- Added `models/transformer` package for loading HuggingFace transformer models (Experimental).
  (Tested with tencent/KaLM-Embedding-Gemma3-12B-2511)

## v0.3.3

- Added `PostProcessor` support and `EncodeWithOptions` to the tokenizer API.
- Split on added tokens before pre-tokenization in `hftokenizer`.

## v0.3.2

- Added `safetensors` support.
- Added support for GGUF file format.
- Tokenizer API improvements.

## v0.3.1

- Fixed go.mod/go.sum.

## v0.3.0

- Bumped the version of GoMLX in tests and documentation.
- Bumped version of dependencies: including github.com/daulet/tokenizers, which requires a fresh download of the 
  corresponding c++ library libtokenizers.a.

## v0.2.2

* Fixed file truncation issues during download.

## v0.2.1

* Forcefully refresh (download) the revision's hash at least once before using.

## v0.2.0

* Add Windows support by moving to the cross-platform flock: see PR #6, thanks to @mrmichaeladavis

## v0.1.2

* If verbosity is 0, it won't print progress.
* Added support for custom end-points. Default being "https://huggingface.co" or the environment variable
  `$HF_ENDPOINT` if defined.

## v0.1.1

* Fixed URL resolution of non-model repos.
* Fixed sentencepiece Tokenizer and tokenizer API string methods (using `enumer`).
* Added dataset example. 
* Added usage with Rust tokenizer.
* Improved README.md
* Added SentencePiece proto support – to be used in future conversion of SentencePiece models.
* Improved documentation.

## v0.1.0

* package `hub`: inspect and download files from arbitrary repos. Very functional.
* package `tokenizers`:
	* Interfaces, types and constants.
	* Gemma tokenizer implementation.
	* Not any other tokenizer implemented yet.
* Examples in `README.md`.
