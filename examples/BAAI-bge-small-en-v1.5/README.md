# BAAI (Beijing Academy of Artificial Intelligence) BGE Small Sentence Embedder (English) v1.5

[1] https://huggingface.co/BAAI/bge-small-en-v1.5

## Testing

Full parity tests for `BAAI/bge-small-en-v1.5` live in [`../../modeltests/`](../../modeltests/) (see [`../../docs/ModelTests.md`](../../docs/ModelTests.md)):

```sh
go test -v -p=1 ./modeltests -modeltests -run TestModels/baai-bge-small-en-v1.5
```
