# [Databricks Dolly 15k Dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k)

An open-source (CC-BY-SA-3.0) dataset of ~15,000 human-generated instruction-following records across diverse categories
(brainstorming, classification, closed QA, generation, information extraction, open QA, and summarization).

This package contains a pre-generated Go struct schema (`gen_schema.go`, generated via `//go:generate`) for reading the
Parquet dataset using `github.com/gomlx/go-huggingface/datasets`.

For more information see:

* https://pkg.go.dev/github.com/gomlx/go-huggingface/datasets : The `datasets` package documentation.
* https://huggingface.co/datasets/databricks/databricks-dolly-15k : The dataset page on HuggingFace Hub.

## `browse`

A small CLI in [`browse/`](./browse) that downloads (and caches) the dataset, prints sample records, and optionally
generates responses with a HuggingFace model using GoMLX's `ml/zoo/transformer/generate`:

```bash
# Browse example #3 (default -n=1):
go run ./examples/dolly/browse -ex=3

# Generate responses for 3 examples using a GoMLX model:
go run ./examples/dolly/browse -n=3 -model=google/gemma-4-E2B-it -prompt_tokens=1024 -gen_tokens=128

# Generate responses using a local Ollama server:
go run ./examples/dolly/browse -n=3 -ollama=gemma4:e2b-it-bf16

# Run both and compare generated outputs side-by-side:
go run ./examples/dolly/browse -n=3 -model=google/gemma-4-E2B-it -ollama=gemma4:e2b-it-bf16
```
