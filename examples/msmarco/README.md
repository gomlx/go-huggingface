# [MSMARCO Dataset](https://huggingface.co/datasets/microsoft/ms_marco)

Used to evaluating ranking and retrieval models.

This package contains a pre-generated schema (and the `//go:generate` line that does that) and some
tests to demo it.

For more information see:

* The [`msmacro_test.go`](https://github.com/gomlx/go-huggingface/blob/main/examples/msmarco/msmarco_test.go) file,
  it serves as an example of how to use the `datasets` package.
* https://pkg.go.dev/github.com/gomlx/go-huggingface/datasets : The `datasets` package documentation.
* https://huggingface.co/datasets/microsoft/ms_marco : The dataset page on HuggingFace Hub.
* https://microsoft.github.io/msmarco/ : Microsoft's MS MARCO page.

## `benchmark_embed`

It includes this small binary that will attempt to embed all the MSMARCO passages using the given model,
and print out the speed. It simply discards the embeddings.

It's provides 3 functionalities:

- Check that a model works.
- Benchmark it in any supporting backend.
- Example for someone who needs to do something similar.

It only embeds the passages, because generally the "document" doesn't require any instruction, while
queries usually require instructions, and there is not a general way to specify them.
