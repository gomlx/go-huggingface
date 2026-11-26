# All-MiniLM-L6-v2-Embedding-GGUF Example & Parity Test

[1] https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF

Demonstrates loading and executing a GGUF-quantized BERT sentence-embedding model (`all-MiniLM-L6-v2-Q4_K_M.gguf`) in GoMLX using:
- Embedded GGUF WordPiece tokenizer metadata (`tokenizer.ggml.*`)
- Native quantized GGML weights (`Q4_K` / `Q6_K`) via `nn.QuantizedDense` and `nn.QuantizedGather`
- Mean pooling + L2 normalization synthesized from GGUF `bert.pooling_type = 1`

## Testing

Full parity tests for all 13 GGUF quantization variants of `second-state/All-MiniLM-L6-v2-Embedding-GGUF` against `llama.cpp` live in [`../../modeltests/`](../../modeltests/) (see [`../../docs/ModelTests.md`](../../docs/ModelTests.md)):

```sh
GOEXPERIMENT=simd go test -v -p=1 ./modeltests -modeltests -run TestModels/minilm-l6-v2-gguf/Q4_K_M
```
