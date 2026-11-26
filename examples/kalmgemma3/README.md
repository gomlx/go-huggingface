# [Tecent's KaLM-Gemma3 12B model](https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511) 

This package defines the `Repository` ID, helper loaders, and task prompt utilities for `tencent/KaLM-Embedding-Gemma3-12B-2511`.

Full parity tests live in [`../../modeltests/`](../../modeltests/) (see [`../../docs/ModelTests.md`](../../docs/ModelTests.md)):

```sh
go test -v -p=1 ./modeltests -modeltests -run TestModels/kalm-gemma3-12b
```

See KaLM embedding site [1] and HuggingFace model repo [2]. You'll find
links to the papers in either these locations.

* [1] https://kalm-embedding.github.io/
* [2] https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511
