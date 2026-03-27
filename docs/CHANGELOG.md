# `go-huggingface` Changelog

## Next

- Package `datasets`:
  - Added `datasets` package for downloading and iterating over parquet files of datasets from the HuggingFace Hub.
  - Added `cmd/generate_dataset_structs` for generating Go structs for dataset records.
  - Added `ParquetFixListSchema` for fixing list schema parsed from Go struct (a bug? in parquet-go where it hard-codes
    the group/element node names in lists).
- Package `tokenizers/bucket`
  - Added `bucket` package for streaming tokenization of sentences into buckets (or batches) of discrete sizes, 
    to minimize padding.

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
