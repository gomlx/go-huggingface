# Timestral Example & Benchmark (`examples/timestral`)

This example demonstrates running Mistral AI's **Timestral** time-series foundation models (`mistralai/Timestral-5M`, `mistralai/Timestral-28M`, `mistralai/Timestral-123M`, `mistralai/Timestral-209M`, or any compatible Hugging Face repo ID or local checkpoint directory) natively in GoMLX using [`github.com/gomlx/go-huggingface/models/timestral`](../../models/timestral).

It loads `config.json` and `model.safetensors` directly via `go-huggingface/hub` and `models/timestral`, uses the embedded 12,096-step 5-minute series (`context.f32`) with a missing-value (`NaN`) gap, and compiles one `(batch=2, contextSteps)` shape with weights baked as graph constants.

## Usage

### Run with PyTorch Reference Comparison (`-compare`)

```bash
GOEXPERIMENT=simd go run ./examples/timestral \
    -model mistralai/Timestral-28M \
    -compare ./examples/timestral/model-28m_test_results.f32 \
    -backend go \
    -runs 5
```

For the 123M model:

```bash
GOEXPERIMENT=simd go run ./examples/timestral \
    -model mistralai/Timestral-123M \
    -compare ./examples/timestral/model-123m_test_results.f32 \
    -backend go \
    -runs 5
```

### Run Standalone

```bash
GOEXPERIMENT=simd go run ./examples/timestral \
    -model mistralai/Timestral-28M \
    -context-steps 12096 \
    -horizon-steps 864 \
    -backend go \
    -runs 5
```

## Running Tests

To run the numerical accuracy tests for both `Timestral-28M` and `Timestral-123M` against the local PyTorch reference files (`model-28m_test_results.f32` and `model-123m_test_results.f32`, verifying `max abs diff < 1e-4`):

```bash
go test -v ./examples/timestral/...
```
