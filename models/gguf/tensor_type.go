// Package gguf provides a reader for GGUF (GGML Universal Format) model files,
// with support for loading tensor data as GoMLX tensors, including on-the-fly
// dequantization of quantized weight formats.
//
// Example loading from a local file:
//
//	model, err := gguf.NewFromFile("/path/to/model.gguf")
//	if err != nil {
//		panic(err)
//	}
//	for tn, err := range model.IterTensors() {
//		if err != nil {
//			panic(err)
//		}
//		fmt.Printf("- Tensor %s: shape=%s\n", tn.Name, tn.Tensor.Shape())
//	}
//
// Example loading from HuggingFace:
//
//	repo := hub.New(modelID).WithAuth(hfAuthToken)
//	for tn, err := range gguf.IterTensorsFromRepo(repo) {
//		if err != nil {
//			panic(err)
//		}
//		fmt.Printf("- Tensor %s: shape=%s\n", tn.Name, tn.Tensor.Shape())
//	}
package gguf

import (
	"fmt"
	"slices"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// TensorType represents the data type or quantization format of a tensor in a GGUF file.
type TensorType uint32

const (
	TensorTypeF32  TensorType = 0
	TensorTypeF16  TensorType = 1
	TensorTypeQ4_0 TensorType = 2
	TensorTypeQ4_1 TensorType = 3
	// 4, 5 are unused/removed types.
	TensorTypeQ5_0 TensorType = 6
	TensorTypeQ5_1 TensorType = 7
	TensorTypeQ8_0 TensorType = 8
	TensorTypeQ8_1 TensorType = 9
	TensorTypeQ2_K TensorType = 10
	TensorTypeQ3_K TensorType = 11
	TensorTypeQ4_K TensorType = 12
	TensorTypeQ5_K TensorType = 13
	TensorTypeQ6_K TensorType = 14
	TensorTypeQ8_K TensorType = 15
	TensorTypeIQ2_XXS TensorType = 16
	TensorTypeIQ2_XS  TensorType = 17
	TensorTypeIQ3_XXS TensorType = 18
	TensorTypeIQ1_S   TensorType = 19
	TensorTypeIQ4_NL  TensorType = 20
	TensorTypeIQ3_S   TensorType = 21
	TensorTypeIQ2_S   TensorType = 22
	TensorTypeIQ4_XS  TensorType = 23
	TensorTypeI8  TensorType = 24
	TensorTypeI16 TensorType = 25
	TensorTypeI32 TensorType = 26
	TensorTypeI64 TensorType = 27
	TensorTypeF64 TensorType = 28
	TensorTypeIQ1_M TensorType = 29
	TensorTypeBF16  TensorType = 30
	// 31-33 are unused.
	TensorTypeTQ1_0 TensorType = 34
	TensorTypeTQ2_0 TensorType = 35
	// 36-38 are unused.
	TensorTypeMXFP4 TensorType = 39
)

// String returns a human-readable name for the tensor type.
func (t TensorType) String() string {
	switch t {
	case TensorTypeF32:
		return "F32"
	case TensorTypeF16:
		return "F16"
	case TensorTypeQ4_0:
		return "Q4_0"
	case TensorTypeQ4_1:
		return "Q4_1"
	case TensorTypeQ5_0:
		return "Q5_0"
	case TensorTypeQ5_1:
		return "Q5_1"
	case TensorTypeQ8_0:
		return "Q8_0"
	case TensorTypeQ8_1:
		return "Q8_1"
	case TensorTypeQ2_K:
		return "Q2_K"
	case TensorTypeQ3_K:
		return "Q3_K"
	case TensorTypeQ4_K:
		return "Q4_K"
	case TensorTypeQ5_K:
		return "Q5_K"
	case TensorTypeQ6_K:
		return "Q6_K"
	case TensorTypeQ8_K:
		return "Q8_K"
	case TensorTypeIQ2_XXS:
		return "IQ2_XXS"
	case TensorTypeIQ2_XS:
		return "IQ2_XS"
	case TensorTypeIQ3_XXS:
		return "IQ3_XXS"
	case TensorTypeIQ1_S:
		return "IQ1_S"
	case TensorTypeIQ4_NL:
		return "IQ4_NL"
	case TensorTypeIQ3_S:
		return "IQ3_S"
	case TensorTypeIQ2_S:
		return "IQ2_S"
	case TensorTypeIQ4_XS:
		return "IQ4_XS"
	case TensorTypeI8:
		return "I8"
	case TensorTypeI16:
		return "I16"
	case TensorTypeI32:
		return "I32"
	case TensorTypeI64:
		return "I64"
	case TensorTypeF64:
		return "F64"
	case TensorTypeIQ1_M:
		return "IQ1_M"
	case TensorTypeBF16:
		return "BF16"
	case TensorTypeTQ1_0:
		return "TQ1_0"
	case TensorTypeTQ2_0:
		return "TQ2_0"
	case TensorTypeMXFP4:
		return "MXFP4"
	default:
		return fmt.Sprintf("unknown(%d)", t)
	}
}

// BlockSize returns the number of elements per quantization block.
// Native types have a block size of 1.
// Legacy quantized types (Q4_0, Q8_0, etc.) have a block size of 32.
// K-quant types (Q2_K, Q4_K, etc.) have a block size of 256.
func (t TensorType) BlockSize() int {
	switch t {
	case TensorTypeF32, TensorTypeF16, TensorTypeBF16, TensorTypeF64,
		TensorTypeI8, TensorTypeI16, TensorTypeI32, TensorTypeI64:
		return 1
	case TensorTypeQ4_0, TensorTypeQ4_1, TensorTypeQ5_0, TensorTypeQ5_1,
		TensorTypeQ8_0, TensorTypeQ8_1, TensorTypeIQ4_NL:
		return 32
	case TensorTypeQ2_K, TensorTypeQ3_K, TensorTypeQ4_K, TensorTypeQ5_K,
		TensorTypeQ6_K, TensorTypeQ8_K,
		TensorTypeIQ2_XXS, TensorTypeIQ2_XS, TensorTypeIQ3_XXS,
		TensorTypeIQ1_S, TensorTypeIQ3_S, TensorTypeIQ2_S, TensorTypeIQ4_XS,
		TensorTypeIQ1_M, TensorTypeTQ1_0, TensorTypeTQ2_0:
		return 256
	case TensorTypeMXFP4:
		return 32
	default:
		return 0
	}
}

// TypeSize returns the number of bytes per quantization block.
// For native types with block size 1, this is the element size in bytes.
func (t TensorType) TypeSize() int {
	switch t {
	case TensorTypeF32:
		return 4
	case TensorTypeF16:
		return 2
	case TensorTypeBF16:
		return 2
	case TensorTypeF64:
		return 8
	case TensorTypeI8:
		return 1
	case TensorTypeI16:
		return 2
	case TensorTypeI32:
		return 4
	case TensorTypeI64:
		return 8
	// Legacy quants (block size = 32):
	case TensorTypeQ4_0:
		return 2 + 32/2 // f16 scale + 16 bytes of nibbles = 18
	case TensorTypeQ4_1:
		return 2 + 2 + 32/2 // f16 scale + f16 min + 16 bytes = 20
	case TensorTypeQ5_0:
		return 2 + 4 + 32/2 // f16 scale + 4 bytes high bits + 16 bytes = 22
	case TensorTypeQ5_1:
		return 2 + 2 + 4 + 32/2 // f16 scale + f16 min + 4 bytes high bits + 16 bytes = 24
	case TensorTypeQ8_0:
		return 2 + 32 // f16 scale + 32 int8 values = 34
	case TensorTypeQ8_1:
		return 2 + 2 + 32 // f16 d + f16 s + 32 int8 values = 36
		// Correction: Q8_1 is f16 d + f16 s + 32 int8 = 36
	// K-quants (block size = 256):
	case TensorTypeQ2_K:
		return 256/4 + 256/16 + 2 + 2 // 64 + 16 + 2 + 2 = 84
	case TensorTypeQ3_K:
		return 256/4 + 256/8 + 12 + 2 // 64 + 32 + 12 + 2 = 110
	case TensorTypeQ4_K:
		return 2 + 2 + 12 + 256/2 // 2 + 2 + 12 + 128 = 144
	case TensorTypeQ5_K:
		return 2 + 2 + 12 + 256/2 + 256/8 // 2 + 2 + 12 + 128 + 32 = 176
	case TensorTypeQ6_K:
		return 256/2 + 256/4 + 256/16 + 2 // 128 + 64 + 16 + 2 = 210
	case TensorTypeQ8_K:
		return 4 + 256 + 256/16*2 // f32 d + 256 int8 + 16 f16 scales = 4+256+32 = 292
	case TensorTypeIQ4_NL:
		return 2 + 32/2 // same as Q4_0 layout = 18
	default:
		return 0
	}
}

// IsQuantized returns true if the tensor type requires dequantization
// to be used as standard floating-point data.
func (t TensorType) IsQuantized() bool {
	switch t {
	case TensorTypeF32, TensorTypeF16, TensorTypeBF16, TensorTypeF64,
		TensorTypeI8, TensorTypeI16, TensorTypeI32, TensorTypeI64:
		return false
	default:
		return true
	}
}

// GoMLXDType returns the GoMLX dtype for native (non-quantized) tensor types.
// For quantized types, returns dtypes.Float32 (the dequantization output type).
func (t TensorType) GoMLXDType() dtypes.DType {
	switch t {
	case TensorTypeF32:
		return dtypes.Float32
	case TensorTypeF16:
		return dtypes.Float16
	case TensorTypeBF16:
		return dtypes.BFloat16
	case TensorTypeF64:
		return dtypes.Float64
	case TensorTypeI8:
		return dtypes.Int8
	case TensorTypeI16:
		return dtypes.Int16
	case TensorTypeI32:
		return dtypes.Int32
	case TensorTypeI64:
		return dtypes.Int64
	default:
		// Quantized types dequantize to Float32.
		return dtypes.Float32
	}
}

// TensorInfo holds parsed information about a single tensor in a GGUF file.
type TensorInfo struct {
	Name   string
	Shape  []uint64   // Dimensions in GGUF native order (innermost first).
	Type   TensorType
	Offset uint64     // Byte offset within the tensor data section.
}

// NumElements returns the total number of elements in the tensor.
func (ti *TensorInfo) NumElements() uint64 {
	if len(ti.Shape) == 0 {
		return 0
	}
	n := uint64(1)
	for _, d := range ti.Shape {
		n *= d
	}
	return n
}

// NumBytes returns the total number of bytes the tensor data occupies in the file.
func (ti *TensorInfo) NumBytes() int64 {
	bs := ti.Type.BlockSize()
	ts := ti.Type.TypeSize()
	if bs == 0 || ts == 0 {
		return 0
	}
	nElements := ti.NumElements()
	nBlocks := nElements / uint64(bs)
	return int64(nBlocks) * int64(ts)
}

// GoMLXShape returns the GoMLX dtype and dimensions for this tensor.
// GGUF stores dimensions innermost-first; this reverses them to the
// outermost-first convention used by GoMLX and HuggingFace.
func (ti *TensorInfo) GoMLXShape() (dtypes.DType, []int) {
	dtype := ti.Type.GoMLXDType()
	dims := make([]int, len(ti.Shape))
	for i, d := range ti.Shape {
		dims[i] = int(d)
	}
	slices.Reverse(dims)
	return dtype, dims
}
