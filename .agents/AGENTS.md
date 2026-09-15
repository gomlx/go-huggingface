# Developer Summary for AI Agents (`go-huggingface`)

Welcome to the `go-huggingface` codebase. This document is a comprehensive guide for AI developer agents modifying, extending, or debugging this repository.

---

## 1. Project Overview & Architectural Philosophy

[`go-huggingface`](file:///home/janpf/Projects/gomlx/go-huggingface) provides Go-native tools to interact with [HuggingFace](https://huggingface.co):
- **Hub Client (`hub`)**: Download and cache model, dataset, or generic repository files, inspect repo metadata, or work with local and embedded files.
- **Tokenizers (`tokenizers`)**: Pure Go implementations of HuggingFace tokenizers (WordPiece, BPE, Unigram) reading directly from `tokenizer.json`, plus SentencePiece model support.
- **Streaming Bucketing (`tokenizers/bucket`)**: Grouping sentences by length into discrete buckets (Power-of-2, Two-Bits) with padding minimization and latency limits.
- **Model Formats & Execution (`models/`)**: 
  - Safetensors parsing and zero-copy memory-mapped loading into GoMLX backends (`models/safetensors`).
  - GGUF binary format parsing and pure-Go dequantization (`models/gguf`).
  - HuggingFace transformer config translation to GoMLX computation graphs (`models/transformer`).
  - Segment Anything 2 (SAM2) image backbone and mask decoder (`models/sam2`).
- **Parquet Datasets (`datasets`)**: On-demand downloading, schema inspection, and stream-reading of Parquet-backed HuggingFace datasets with automatic Go struct generation.

### Modularity & Dependency Decoupling
The packages are strictly decoupled:
- `hub` has **no** dependency on GoMLX or Parquet.
- `tokenizers` has **no** dependency on GoMLX, Parquet, or external CGo/Python libraries (unless using optional third-party wrappers like `daulet/tokenizers`).
- GoMLX (`github.com/gomlx/gomlx` and `github.com/gomlx/compute`) is only imported by `models/*` and example packages.

---

## 2. Package Directory Map

```
go-huggingface/
├── cmd/
│   ├── dataset_download/        # CLI tool to inspect, download, list, and delete dataset files
│   ├── generate_dataset_structs/ # CLI tool to generate Go structs from Parquet dataset schemas
│   └── hubinfo/                 # CLI tool to inspect metadata, save local copies, and manage cache
├── datasets/                    # HuggingFace datasets client & Parquet scanning
├── docs/
│   └── CHANGELOG.md             # Project change log (MANDATORY UPDATE ON EVERY FIX)
├── examples/
│   ├── BAAI-bge-small-en-v1.5/  # Example & test suite for BERT-based sentence embeddings
│   ├── gemma4-e4bit/            # Gemma 4 E4B model loading example
│   ├── kalmgemma3/              # KaLM-Gemma3 12B sentence embedding example & tests
│   └── msmarco/                 # MS MARCO dataset reader and embedding benchmark
├── hub/                         # Core HuggingFace Hub client (remote, local, and embed modes)
├── internal/
│   ├── downloader/              # Parallel download manager with FIFOSemaphore & atomic file locking
│   ├── files/                   # File utilities (tilde expansion, flock-based file locking)
│   ├── py/                      # Python reference scripts for generating golden test tensors
│   └── testing/                 # Shared test validation utilities (tensor comparison tolerances)
├── models/
│   ├── gguf/                    # GGUF parser, metadata extraction & k-quant dequantization
│   ├── image/                   # Image preprocessing graph operations (resize, normalize)
│   ├── safetensors/             # Safetensors reader, mmap integration & tensor iterators
│   ├── sam2/                    # Segment Anything 2 (SAM2) model architecture & Segmenter API
│   └── transformer/             # Transformer graph builder for GoMLX
└── tokenizers/
    ├── api/                     # Tokenizer interface, TokenSpan, AnnotatedEncoding, EncodeOptions
    ├── bucket/                  # Sentence length bucketing (TwoBitBucket, ByPower, ByPowerBudget)
    ├── hftokenizer/             # Native Go tokenizer engine parsing tokenizer.json (BPE/WordPiece/Unigram)
    └── sentencepiece/           # SentencePiece processor wrapper (via eliben/go-sentencepiece)
```

---

## 3. Public Packages Deep-Dive

### 3.1. `hub`
Manages downloading and access to HuggingFace repositories.
- **Three Repository Modes**:
  1. **Remote (`hub.New(id)`)**: Downloads from HuggingFace Hub into the local cache (`~/.cache/huggingface/hub/` or `$XDG_CACHE_HOME/huggingface/hub/`).
  2. **Local (`hub.NewLocal(dir)`)**: Reads directly from a local folder on disk without any network requests. Useful for offline environments, `git clone` checkouts, or Docker images.
  3. **Embedded (`hub.NewEmbed(fsys, subDir)`)**: Reads directly from an `fs.FS` (e.g., `//go:embed`). Operates fully in-memory.
- **Preferred File Access APIs**:
  - Use `repo.Open(fileName)` (returns `fs.File`) or `repo.ReadFile(fileName)` (returns `[]byte`). These stream data directly from memory for embedded repositories without disk writes.
  - Avoid `repo.DownloadFile(s)` unless an OS disk path string is strictly necessary: for embedded repos, `DownloadFile` forces temporary extraction into `os.TempDir()`.
  - Use `repo.FetchFiles(...)` to pre-download/cache files without requesting their disk paths.
- **Local Copying & Cache Management**:
  - `repo.Save(dirPath, linkOnly)`: Exports a remote repo to a local directory (supporting hard links via `linkOnly` on the same filesystem).
  - `repo.DeleteCache()`: Deletes downloaded cache for a remote repo.

### 3.2. `tokenizers`
- Implements the [`api.Tokenizer`](file:///home/janpf/Projects/gomlx/go-huggingface/tokenizers/api/api.go) interface:
  - `Encode(text) []int`: Encodes text to token IDs.
  - `EncodeWithAnnotations(text) AnnotatedEncoding`: Returns token IDs along with byte spans (`TokenSpan`) and special token masks.
  - `Decode([]int) string`: Reconstructs text from token IDs.
  - `SpecialTokenID(SpecialToken) (int, error)`: Standardized lookup for `[BOS]`, `[EOS]`, `[PAD]`, `[UNK]`, `[MASK]`, `[CLS]`.
- **`hftokenizer`**:
  - Pure Go implementation that unmarshals `tokenizer.json`.
  - Handles normalizers (BERT normalizer, lowercase, strip accents, replace), pre-tokenizers (whitespace, ByteLevel, Metaspace, BertPreTokenizer), models (BPE with byte fallback, WordPiece with `##` prefix, Unigram with Viterbi search), and post-processors (TemplateProcessing, BertProcessing, RobertaProcessing).
- **`tokenizers/bucket`**:
  - Asynchronously batches streamed sentences into discrete length buckets to minimize zero-padding overhead in neural network inputs.
  - Strategies:
    - `ByTwoBitBucket(batchSize, minSentenceLength)`: Buckets using values representable by 2 bits (e.g., 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64...). Restricts padding overhead to ~20% while providing binary/hardware-friendly shapes.
    - `ByPower(batchSize, minSentenceLength, base)` / `ByPowerBudget(tokenBudget, minSentenceLength, base)`.
  - Supports `WithMaxDelay(...)` to flush partial batches after a latency threshold in real-time streaming services.

### 3.3. `models/safetensors`
- Reads Safetensors files (single file or sharded models via `model.safetensors.index.json`).
- Uses `edsrzf/mmap-go` for direct memory-mapping.
- `IterTensorsFromRepo(repo)` streams tensors across shards with parallelized transfers directly into a `compute.Backend` device without redundant user-space copies.

### 3.4. `models/gguf`
- Parses GGUF v2/v3 files (magic `GGUF`, metadata key-values, tensor info headers).
- Features pure-Go dequantizers for standard and k-quant block formats:
  - `Q8_0`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`
  - `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`
- Reads half-precision FP16 scales using `compute/dtypes/float16`.

### 3.5. `models/transformer`
- Maps HuggingFace model architectures to GoMLX computation graphs (`github.com/gomlx/gomlx/ml/zoo/transformer`).
- Parses HuggingFace config stack:
  - `config.json` (mandatory base config)
  - `config_sentence_transformers.json` (sentence transformer pipeline settings)
  - `modules.json` (execution pipeline: Transformer -> Pooling -> Normalize)
  - `1_Pooling/config.json` (pooling strategy: mean, CLS, max)
- Supports RoPE positional embeddings, layer normalization, RMSNorm, multi-head/grouped-query attention, and task prompts (`task_prompts.json`).

### 3.6. `models/sam2`
- Implements Facebook's Segment Anything 2 (SAM2) image architecture in GoMLX:
  - Hiera vision backbone with multi-scale feature pyramids.
  - Two-way attention mask decoder and prompt encoder (points and bounding boxes).
  - High-level [`Segmenter`](file:///home/janpf/Projects/gomlx/go-huggingface/models/sam2/segmenter.go) struct for interactive image segmentation with point and box prompts.

### 3.7. `datasets`
- Connects to HuggingFace datasets server (`datasets-server.huggingface.co`) or raw repository files.
- Inspects splits, configs, and column features.
- Iterates over remote Parquet datasets on-demand (`IterParquetFromDataset[T]`), downloading files one by one to avoid exhausting disk space.
- Codegen: [`GenerateGoStructFromParquet`](file:///home/janpf/Projects/gomlx/go-huggingface/datasets/codegen.go) automatically generates Go struct definitions with `json` and `parquet` struct tags matching a Parquet schema.

---

## 4. Internal Packages Scan

- **`internal/downloader`**:
  - [`FIFOSemaphore`](file:///home/janpf/Projects/gomlx/go-huggingface/internal/downloader/semaphore.go): Concurrency limiter with strict FIFO queuing to prevent starvation during massive parallel downloads. Supports context cancellation while queued and dynamic capacity resizing.
  - [`LockedDownload`](file:///home/janpf/Projects/gomlx/go-huggingface/internal/downloader/locked.go): Downloads into temporary `.part` files and atomically renames them to destination. Coordinates parallel processes using `.lock` files. Preserves existing cache on download failure.
- **`internal/files`**:
  - `ExecOnFileLock`: Inter-process file locking using `github.com/gofrs/flock` with jittered polling.
  - `ReplaceTildeInDir`: Expands `~` and `~user` paths into system home directories.
- **`internal/testing`**:
  - `ValidateEmbeddingTensor`: Compares GoMLX tensor embeddings with PyTorch golden files using relative difference metrics.
  - `ReadPythonEmbeddingsList`: Parses floating-point golden values dumped by Python scripts.
- **`internal/py`**:
  - Python scripts (`embed_sentence.py`, `tokenize_bge.py`) used to produce ground-truth test outputs from official HuggingFace/PyTorch libraries.

---

## 5. Non-Obvious Observations & Technical Subtleties

When modifying or extending this repo, keep these specific architectural nuances in mind:

1. **Python Cache Compatibility**:
   - The remote cache directory uses the exact same directory layout and naming conventions as Python's `huggingface_hub` (`~/.cache/huggingface/hub/models--<owner>--<repo>`).
   - Files are stored as content-addressed blobs under `blobs/<etag>` and symlinked relatively into `snapshots/<commitHash>/<path>`.
   - Relative symlinks are required so that caches can be copied or relocated without breaking symlink destinations.

2. **Commit Hash Short-Circuiting**:
   - If `repo.WithRevision(...)` is provided with a 40-character git hex commit hash, `hub.Repo` skips downloading `_info_.json` during revision resolution and immediately uses the given commit hash.

3. **Local Directory Mode Skips Tooling Directories**:
   - In `hub.NewLocal(dir)`, file scanning automatically ignores `.git`, `.cache`, `.huggingface`, and `.ipynb_checkpoints` directories at any depth. Its synthetic commit hash is set to `"local"`.

4. **Embedded Repositories and Disk Extractions**:
   - `hub.NewEmbed` reads in-memory without disk overhead when using `Open`, `ReadFile`, or `FetchFiles`.
   - However, calling `DownloadFile` or `DownloadFiles` on an embedded repo forces file extraction into `os.TempDir()/go-huggingface-embed/<id>/` because the caller explicitly requested filesystem paths.

5. **TokenSpan Offsets Are Byte Positions**:
   - In `api.TokenSpan`, `Start` and `End` are **byte offsets** (not UTF-8 rune offsets). This enables fast zero-allocation slicing of Go strings: `originalText[span.Start:span.End]`. Any normalizer that changes string lengths must carefully map intermediate spans back to original byte positions.

6. **Tokenizer JSON Format Polymorphism**:
   - `tokenizer.json` files have polymorphic representations across HuggingFace models:
     - Vocab can be an object `{"token": id}` (WordPiece, BPE) OR a 2D array `[["token", score], ...]` (Unigram, where the integer token ID is the array index and the float is log probability).
     - Merges can be a string list `["token1 token2", ...]` OR a nested array `[["token1", "token2"], ...]`.
   - `hftokenizer.Model.UnmarshalJSON` explicitly handles both formats.

7. **Transformer Causal Mask Defaults**:
   - `transformer.Model` defaults causal masking to **true**.
   - **Crucial**: Many sentence embedding models (such as KaLM-Gemma3) are bidirectional and trained *without* causal masks. For sentence embedders, always verify whether `hfModel.WithCausalMask(false)` must be explicitly called.

8. **Parquet 3-Level List Schema Compatibility (`ParquetFixListSchema`)**:
   - The Go `parquet-go` package expects repeated lists to follow a rigid standard schema with node names `"list"` and `"element"`.
   - Many HuggingFace parquet files use custom group/element node names for list fields.
   - `datasets.ParquetFixListSchema[T]` inspects the actual Parquet file schema and dynamically rewrites the Go struct's schema tree to match the file's naming conventions, preventing unmarshaling errors.

9. **Dynamic XLA Pooling in Sentence Transformers**:
   - Sentence pooling in `models/transformer/sentence.go` utilizes dynamic XLA operations (`DynamicBroadcastInDim`, `DynamicIota`, etc.) to support variable sequence lengths and padded batches without triggering recompilation or conditional graph branching.

10. **Tools Configured in `go.mod`**:
    - The repository declares development tools via `tool` directives in `go.mod` (Go 1.24+ standard):
      ```go
      tool github.com/gomlx/go-huggingface/cmd/generate_dataset_structs
      tool github.com/gomlx/go-huggingface/cmd/dataset_download
      ```

---

## 6. Testing & Tooling Guidelines

- **Running Tests**:
  - Run fast/offline unit tests (same as CI):
    ```bash
    go test -short ./...
    ```
  - Running full tests (requires cached models/datasets or active network connection):
    ```bash
    go test ./...
    ```
- **Using CLI Tools**:
  - Inspect model info:
    ```bash
    go run ./cmd/hubinfo Qwen/Qwen3-0.6B
    ```
  - Generate dataset struct:
    ```bash
    go run ./cmd/generate_dataset_structs -dataset HuggingFaceFW/fineweb
    ```
  - Download dataset partitions:
    ```bash
    go run ./cmd/dataset_download -config default -split train microsoft/ms_marco
    ```

---

## 7. CRITICAL: CHANGELOG Note on Every Fix

> [!IMPORTANT]
> **MANDATORY FOR ALL AGENTS AND DEVELOPERS**:
> Every time a bug is fixed, a feature is added, or an architectural improvement is made, you **MUST** add a bullet point describing the change to [`docs/CHANGELOG.md`](file:///home/janpf/Projects/gomlx/go-huggingface/docs/CHANGELOG.md).
> 
> - Place the entry under the `## Unreleased` section.
> - Include today's date formatted as `- YYYY/MM/DD:` (or add to today's date list if it already exists).
> - Specify the affected package(s) and concisely summarize the change, rationale, and any issue/PR reference if applicable.
