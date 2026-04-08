# `go-huggingface` Changelog

## Next

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
