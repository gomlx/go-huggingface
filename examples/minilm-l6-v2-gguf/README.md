# All-MiniLM-L6-v2-Embedding-GGUF Example & Parity Test

[1] https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF

Demonstrates loading and executing a GGUF-quantized BERT sentence-embedding model (`all-MiniLM-L6-v2-Q4_K_M.gguf`) in GoMLX using:
- Embedded GGUF WordPiece tokenizer metadata (`tokenizer.ggml.*`)
- Native quantized GGML weights (`Q4_K` / `Q6_K`) via `nn.QuantizedDense` and `nn.QuantizedGather`
- Mean pooling + L2 normalization synthesized from GGUF `bert.pooling_type = 1`

## Testing

`similarity_embeddings.txt` contains materialized `float32` reference embeddings produced by `llama.cpp` (via [`hybridgroup/yzma`](https://github.com/hybridgroup/yzma), CPU `NGpuLayers=0`, `PoolingTypeMean`, L2-normalized) on `all-MiniLM-L6-v2-Q4_K_M.gguf`.

Run the parity test with:

```sh
GOEXPERIMENT=simd GOMLX_BACKEND="go" go test -v ./examples/minilm-l6-v2-gguf/...
```
