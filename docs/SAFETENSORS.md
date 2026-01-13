# Safetensor Loading

This document describes the safetensor loading functionality in the go-huggingface library.

## Overview

The library now supports loading and parsing `.safetensors` files from HuggingFace models, with:
- Pure Go implementation (no cgo required)
- Memory-mapped streaming for large models
- Automatic sharded model support
- GoMLX tensor library integration
- Full dtype support (F32, F64, I32, I64, F16, BF16, etc.)

## Quick Start

### Basic Usage

```go
repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")

// List all tensors in a file
names, err := repo.ListSafetensors("model.safetensors")

// Load raw tensor data
data, meta, err := repo.LoadSafetensorRaw("model.safetensors", "embeddings.word_embeddings.weight")
fmt.Printf("Dtype: %s, Shape: %v, Size: %d bytes\n", meta.Dtype, meta.Shape, len(data))
```

### GoMLX Integration

```go
repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")

// Load a tensor directly as GoMLX tensor
tensor, err := repo.LoadSafetensorForGoMLX("model.safetensors", "embeddings.word_embeddings.weight")
if err != nil {
    panic(err)
}

// Use with GoMLX
// graph.ConstTensor(g, tensor)
```

### Sharded Models

Large models are often split across multiple files. The library automatically detects and handles these:

```go
repo := hub.New("google/gemma-2-2b-it").WithAuth(os.Getenv("HF_TOKEN"))

// Load sharded model index
sharded, err := repo.LoadShardedModel("model.safetensors.index.json")
if err != nil {
    panic(err)
}

// List all tensors across all shards
tensorNames := sharded.ListTensors()

// Load a specific tensor (automatically finds the right shard)
data, meta, err := sharded.LoadTensor("model.layers.0.self_attn.q_proj.weight")
```

### Iterating Over All Tensors

```go
repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")

// Iterate over all tensors (automatically handles sharded models)
for tensorData, err := range repo.IterAllTensors() {
    if err != nil {
        panic(err)
    }
    fmt.Printf("Tensor: %s, Shape: %v, Dtype: %s\n", 
        tensorData.Name, tensorData.Meta.Shape, tensorData.Meta.Dtype)
}

// Iterate with GoMLX conversion
for name, tensor, err := range repo.IterAllTensorsForGoMLX() {
    if err != nil {
        panic(err)
    }
    fmt.Printf("GoMLX Tensor: %s, Shape: %v\n", name, tensor.Shape())
}

// Iterate with Gorgonia conversion
for name, tensor, err := range repo.IterAllTensorsForGorgonia() {
    if err != nil {
        panic(err)
    }
    fmt.Printf("Gorgonia Tensor: %s, Shape: %v\n", name, tensor.Shape())
}
```

### Streaming Access

F   panic(err)
}
defer reader.Close()

// Get metadata without loading data
meta := reader.Metadata()
fmt.Printf("Tensor size: %d bytes\n", reader.Len())

// Read specific portion of tensor
buf := make([]byte, 1024)
n, err := reader.ReadAt(buf, 0) // Read first 1KB
```

## API Reference

### Core Types

#### `TensorMetadata`
```go
type TensorMetadata struct {
    Name        string      // Tensor name
    Dtype       string      // Data type: F32, F64, I32, I64, F16, BF16, etc.
    Shape       []int       // Tensor dimensions
    DataOffsets [2]int64    // [start, end] byte offsets in file
}
```

#### `SafetensorHeader`
```go
type SafetensorHeader struct {
    Tensors  map[string]*TensorMetadata  // Tensor name -> metadata
    Metadata map[string]interface{}      // Optional metadata
}
```

#### `ShardedModel`
```go
type ShardedModel struct {
    IndexFile string
    Index     *ShardedModelIndex
}
```

### Repo Methods

#### Basic Operations
- `ListSafetensors(filename string) ([]string, error)`
- `GetSafetensorMetadata(filename, tensorName string) (*TensorMetadata, error)`
- `LoadSafetensorRaw(filename, tensorName string) ([]byte, *TensorMetadata, error)`

#### Streaming
- `LoadSafetensorStreaming(filename, tensorName string) (*SafetensorReader, error)`

#### Sharded Models
- `DetectShardedModel() (indexFile string, isSharded bool, error)`
- `LoadShardedModel(indexFilename string) (*ShardedModel, error)`

#### Iteration
- `IterSafetensors() func(yield func(SafetensorFileInfo, error) bool)`
- `IterAllTensors() func(yield func(TensorData, error) bool)`

#### GoMLX Bridge
- `LoadSafetensorForGoMLX(filename, tensorName string) (*tensors.Tensor, error)`
- `LoadSafetensorForGoMLXStreaming(filename, tensorName string) (*tensors.Tensor, error)`
- `IterAllTensorsForGoMLX() func(yield func(name string, tensor *tensors.Tensor, err error) bool)`

#### Gorgonia Bridge
- `LoadSafetensorForGorgonia(filename, tensorName string) (*tensor.Dense, error)`
- `LoadSafetensorForGorgoniaZeroCopy(filename, tensorName string) (*GorgoniaTensorWithCleanup, error)`
- `IterAllTensorsForGorgonia() func(yield func(name string, tensor *tensor.Dense, err error) bool)`

### ShardedModel Methods

- `ListTensors() []string`
- `GetTensorLocation(tensorName string) (filename string, error)`
- `LoadTensor(tensorName string) ([]byte, *TensorMetadata, error)`
- `LoadTensorStreaming(tensorName string) (*SafetensorReader, error)`
- `GetTensorMetadata(tensorName string) (*TensorMetadata, error)`
- `LoadTensorForGoMLX(tensorName string) (*tensors.Tensor, error)`
- `LoadTensorForGorgonia(tensorName string) (*tensor.Dense, error)`
- `LoadTensorForGorgoniaZeroCopy(tensorName string) (*GorgoniaTensorWithCleanup, error)`

## Supported Data Types

The following safetensor dtypes are supported:


## Performance Tips

1. **Use streaming for large models**: `LoadSafetensorStreaming()` uses memory-mapped files instead of loading everything into RAM.

2. **Iterate efficiently**: The iterator methods (`IterAllTensors()`, etc.) download and process files lazily.

3. **Sharded model detection is automatic**: When using `IterAllTensors()`, sharded models are detected and handled automatically.

4. **Reuse repo instances**: The `Repo` object manages caching and downloads - reuse it for multiple operations.

## Architecture

### File Format

Safetensors format:
```
[8 bytes: header size as little-endian u64]
[header_size bytes: JSON header with metadata]
[remaining bytes: tensor data]
```

### Cache Integration

Safetensor files are downloaded using the existing HuggingFace cache structure:
```
~/.cache/huggingface/hub/
├── models--owner--model/
│   ├── blobs/
│   │   └── {etag}  # Actual file content
│   └── snapshots/
│       └── {commit}/
│           └── model.safetensors → ../../blobs/{etag}
```

This ensures compatibility with Python's `huggingface_hub` library.

### Memory Management

- **Raw loading**: Reads entire tensor into memory as `[]byte`
- **Streaming**: Uses `golang.org/x/exp/mmap` for memory-mapped access
- **GoMLX tensors**: Converts raw bytes to typed GoMLX tensors using `tensors.FromFlatDataAndDimensions()`

## Examples

### Example 1: Load Model Embeddings

```go
package main

import (
    "fmt"
    "github.com/gomlx/go-huggingface/hub"
)

func main() {
    repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
    
    // Load word embeddings
    data, meta, err := repo.LoadSafetensorRaw("model.safetensors", "embeddings.word_embeddings.weight")
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Loaded embeddings: %s\n", meta.Dtype)
    fmt.Printf("Shape: %v\n", meta.Shape)
    fmt.Printf("Size: %d bytes\n", len(data))
}
```

### Example 2: Convert All Layers to GoMLX

```go
package main

import (
    "fmt"
    "strings"
    "github.com/gomlx/go-huggingface/hub"
)

func main() {
    repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
    
    for name, tensor, err := range repo.IterAllTensorsForGoMLX() {
        if err != nil {
            panic(err)
        }
        
        // Only process layer weights
        if strings.Contains(name, ".weight") {
            fmt.Printf("%s: %v\n", name, tensor.Shape())
        }
    }
}
```

### Example 3: Working with Large Sharded Models

```go
package main

import (
    "fmt"
    "os"
    "github.com/gomlx/go-huggingface/hub"
)

func main() {
    token := os.Getenv("HF_TOKEN")
    repo := hub.New("google/gemma-2-2b-it").WithAuth(token)
    
    // Auto-detect if sharded
    indexFile, isSharded, err := repo.DetectShardedModel()
    if err != nil {
        panic(err)
    }
    
    if isSharded {
        fmt.Printf("Model is sharded, index file: %s\n", indexFile)
        
        sharded, err := repo.LoadShardedModel(indexFile)
        if err != nil {
            panic(err)
        }
        
        // Load first layer's attention weights
        tensor, err := sharded.LoadTensorForGoMLX("model.layers.0.self_attn.q_proj.weight")
        if err != nil {
            panic(err)
        }
        
        fmt.Printf("Loaded tensor shape: %v\n", tensor.Shape())
    }
}
```

## Testing

Run tests:
```bash
go test ./hub/...
```

## Implementation Notes

1. **Pure Go**: No cgo dependencies, fully cross-platform
2. **Lazy loading**: Files are downloaded and parsed on-demand
3. **Cross-process safe**: Uses file locking for concurrent access
4. **Memory efficient**: Streaming options available
5. **Type safe**: Full Go type system integration with GoMLX

## Future Enhancements

Potential future improvements:
- Custom tensor library bridges (user-provided converters)
- Parallel shard loading for faster initialization
- Tensor slicing without full load
- Write support for creating safetensor files
- Direct ONNX model conversion
