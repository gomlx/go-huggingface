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
	"math"
	"math/bits"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/pkg/errors"
)

// TensorType represents the data type or quantization format of a tensor in a GGUF file.
type TensorType uint32

const (
	TensorTypeF32  TensorType = 0
	TensorTypeF16  TensorType = 1
	TensorTypeQ4_0 TensorType = 2
	TensorTypeQ4_1 TensorType = 3
	// 4, 5 are unused/removed types.
	TensorTypeQ5_0    TensorType = 6
	TensorTypeQ5_1    TensorType = 7
	TensorTypeQ8_0    TensorType = 8
	TensorTypeQ8_1    TensorType = 9
	TensorTypeQ2_K    TensorType = 10
	TensorTypeQ3_K    TensorType = 11
	TensorTypeQ4_K    TensorType = 12
	TensorTypeQ5_K    TensorType = 13
	TensorTypeQ6_K    TensorType = 14
	TensorTypeQ8_K    TensorType = 15
	TensorTypeIQ2_XXS TensorType = 16
	TensorTypeIQ2_XS  TensorType = 17
	TensorTypeIQ3_XXS TensorType = 18
	TensorTypeIQ1_S   TensorType = 19
	TensorTypeIQ4_NL  TensorType = 20
	TensorTypeIQ3_S   TensorType = 21
	TensorTypeIQ2_S   TensorType = 22
	TensorTypeIQ4_XS  TensorType = 23
	TensorTypeI8      TensorType = 24
	TensorTypeI16     TensorType = 25
	TensorTypeI32     TensorType = 26
	TensorTypeI64     TensorType = 27
	TensorTypeF64     TensorType = 28
	TensorTypeIQ1_M   TensorType = 29
	TensorTypeBF16    TensorType = 30
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

// ToGGMLQuantType maps a quantized TensorType to the corresponding GoMLX compute.GGMLQuantType.
// Returns the GGMLQuantType and true if the type is a supported GGML quantization format,
// or (0, false) for non-quantized or unsupported types.
func (t TensorType) ToGGMLQuantType() (compute.GGMLQuantType, bool) {
	switch t {
	case TensorTypeQ4_0:
		return compute.GGMLQ4_0, true
	case TensorTypeQ4_1:
		return compute.GGMLQ4_1, true
	case TensorTypeQ5_0:
		return compute.GGMLQ5_0, true
	case TensorTypeQ5_1:
		return compute.GGMLQ5_1, true
	case TensorTypeQ8_0:
		return compute.GGMLQ8_0, true
	case TensorTypeIQ4_NL:
		return compute.GGMLIQ4NL, true
	case TensorTypeQ2_K:
		return compute.GGMLQ2_K, true
	case TensorTypeQ3_K:
		return compute.GGMLQ3_K, true
	case TensorTypeQ4_K:
		return compute.GGMLQ4_K, true
	case TensorTypeQ5_K:
		return compute.GGMLQ5_K, true
	case TensorTypeQ6_K:
		return compute.GGMLQ6_K, true
	case TensorTypeQ8_K:
		return compute.GGMLQ8_K, true
	default:
		return 0, false
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
	Shape  []uint64 // Dimensions in GGUF native order (innermost first).
	Type   TensorType
	Offset uint64 // Byte offset within the tensor data section.
}

// NumElements returns the total number of elements in the tensor.
// Returns 0 if the shape is empty or the product overflows uint64.
func (ti *TensorInfo) NumElements() uint64 {
	if len(ti.Shape) == 0 {
		return 0
	}
	n := uint64(1)
	for _, d := range ti.Shape {
		hi, prod := bits.Mul64(n, d)
		if hi != 0 {
			return 0
		}
		n = prod
	}
	return n
}

// checkedNumElements validates dimensions and returns the total element count,
// rejecting empty/excessive ranks, dimensions exceeding math.MaxInt, and product overflow.
func (ti *TensorInfo) checkedNumElements() (uint64, error) {
	if len(ti.Shape) == 0 {
		return 0, errors.Errorf("tensor has 0 dimensions")
	}
	if len(ti.Shape) > maxTensorDims {
		return 0, errors.Errorf("tensor has %d dimensions, exceeds limit %d", len(ti.Shape), maxTensorDims)
	}
	n := uint64(1)
	for i, d := range ti.Shape {
		if d > uint64(math.MaxInt) {
			return 0, errors.Errorf("dimension %d (%d) exceeds max int (%d)", i, d, math.MaxInt)
		}
		hi, prod := bits.Mul64(n, d)
		if hi != 0 || prod > uint64(math.MaxInt) {
			return 0, errors.Errorf("tensor shape %v element count overflows int", ti.Shape)
		}
		n = prod
	}
	return n, nil
}

// checkedNumBytes computes the serialized byte size of the tensor with full overflow
// and block-alignment validation.
func (ti *TensorInfo) checkedNumBytes() (int64, error) {
	bs := ti.Type.BlockSize()
	ts := ti.Type.TypeSize()
	if bs == 0 || ts == 0 {
		return 0, errors.Errorf("unsupported tensor type %s (%d)", ti.Type, ti.Type)
	}
	nElements, err := ti.checkedNumElements()
	if err != nil {
		return 0, err
	}
	if bs > 1 && ti.Shape[0]%uint64(bs) != 0 {
		return 0, errors.Errorf("innermost dimension %d is not divisible by block size %d for type %s", ti.Shape[0], bs, ti.Type)
	}
	// Ensure GoMLXRawShape's innermost byte dimension cannot overflow int even if an outer dim is 0.
	if ti.Shape[0]/uint64(bs) > uint64(math.MaxInt)/uint64(ts) {
		return 0, errors.Errorf("innermost raw byte dimension overflows int for shape %v and type %s", ti.Shape, ti.Type)
	}
	nBlocks := nElements / uint64(bs)
	hi, nb := bits.Mul64(nBlocks, uint64(ts))
	if hi != 0 || nb > math.MaxInt64 {
		return 0, errors.Errorf("tensor byte size overflows int64 for shape %v and type %s", ti.Shape, ti.Type)
	}
	return int64(nb), nil
}

// validateTensorShapeAndLimits validates a tensor's dimensions, alignment, and
// allocatable byte limits without requiring file size context.
func validateTensorShapeAndLimits(ti TensorInfo) error {
	nElements, err := ti.checkedNumElements()
	if err != nil {
		return err
	}
	limit := uint64(maxAllocBytes())

	// Check dequantized / GoMLX output tensor byte size.
	outElemSize := uint64(ti.Type.GoMLXDType().Size())
	hi, outBytes := bits.Mul64(nElements, outElemSize)
	if hi != 0 || outBytes > limit {
		return errors.Errorf("tensor output size for shape %v (%s) exceeds limit %d bytes", ti.Shape, ti.Type, limit)
	}

	bs := ti.Type.BlockSize()
	ts := ti.Type.TypeSize()
	if bs > 0 && ts > 0 {
		rawBytes, err := ti.checkedNumBytes()
		if err != nil {
			return err
		}
		if uint64(rawBytes) > limit {
			return errors.Errorf("tensor raw size %d bytes exceeds limit %d bytes", rawBytes, limit)
		}
	}
	return nil
}

// validateTensorInFile validates a tensor's dimensions, byte size, and offset against
// the file's data section [dataOffset, fileSize]. If requireSupportedType is true,
// tensor types without known BlockSize/TypeSize are rejected.
func validateTensorInFile(ti TensorInfo, dataOffset, fileSize int64, requireSupportedType bool) (tensorOffset int64, rawBytes int64, err error) {
	if err := validateTensorShapeAndLimits(ti); err != nil {
		return 0, 0, err
	}
	if dataOffset < 0 || fileSize < 0 || dataOffset > fileSize {
		return 0, 0, errors.Errorf("data section offset %d out of range for %d-byte file", dataOffset, fileSize)
	}

	dataSectionSize := uint64(fileSize - dataOffset)
	if ti.Offset > dataSectionSize {
		return 0, 0, errors.Errorf("tensor offset %d exceeds %d-byte data section", ti.Offset, dataSectionSize)
	}
	tensorOffset = dataOffset + int64(ti.Offset)
	remaining := dataSectionSize - ti.Offset

	bs := ti.Type.BlockSize()
	ts := ti.Type.TypeSize()
	if bs == 0 || ts == 0 {
		if requireSupportedType {
			return 0, 0, errors.Errorf("unsupported tensor type %s (%d)", ti.Type, ti.Type)
		}
		// Even for an unknown/unimplemented type, any non-empty tensor must require at least 1 bit/element.
		nElements, err := ti.checkedNumElements()
		if err != nil {
			return 0, 0, err
		}
		minBytes := (nElements + 7) / 8
		if minBytes > remaining {
			return 0, 0, errors.Errorf("tensor requires at least %d bytes at offset %d, exceeding %d-byte data section", minBytes, ti.Offset, dataSectionSize)
		}
		return tensorOffset, 0, nil
	}

	rawBytes, err = ti.checkedNumBytes()
	if err != nil {
		return 0, 0, err
	}
	if rawBytes > maxAllocBytes() {
		return 0, 0, errors.Errorf("tensor raw size %d bytes exceeds limit %d bytes", rawBytes, maxAllocBytes())
	}
	if uint64(rawBytes) > remaining {
		return 0, 0, errors.Errorf("tensor requires %d bytes at offset %d, exceeding %d-byte data section", rawBytes, ti.Offset, dataSectionSize)
	}
	return tensorOffset, rawBytes, nil
}

// NumBytes returns the total number of bytes the tensor data occupies in the file.
// Returns 0 if the type is unsupported or the result would overflow int64.
func (ti *TensorInfo) NumBytes() int64 {
	n, err := ti.checkedNumBytes()
	if err != nil {
		return 0
	}
	return n
}

// GoMLXShape returns the GoMLX dtype and dimensions for this tensor.
// GGUF stores dimensions innermost-first; this reverses them to the
// outermost-first convention used by GoMLX and HuggingFace.
//
// Quantized dtypes get converted to dtypes.Float32.
func (ti *TensorInfo) GoMLXShape() (dtypes.DType, []int) {
	dtype := ti.Type.GoMLXDType()
	dims := make([]int, len(ti.Shape))
	for i, d := range ti.Shape {
		dims[i] = int(d)
	}
	slices.Reverse(dims)
	return dtype, dims
}

// GoMLXRawShape returns dtypes.Uint8 and the byte-level dimensions for this tensor
// in its raw (un-dequantized) GGUF block layout.
// GGUF stores dimensions innermost-first; this reverses them to outermost-first
// and converts the innermost dimension from logical elements (K) to raw bytes:
// bytesPerRow = (K / BlockSize) * TypeSize.
// For example, a 2D quantized weight matrix with logical shape [N, K] has
// raw shape [N, bytesPerRow] with dtype Uint8, matching GoMLX's QuantGGML layout.
func (ti *TensorInfo) GoMLXRawShape() (dtypes.DType, []int) {
	if len(ti.Shape) == 0 {
		return dtypes.Uint8, nil
	}
	bs := ti.Type.BlockSize()
	ts := ti.Type.TypeSize()
	if bs == 0 || ts == 0 {
		return dtypes.Uint8, nil
	}
	dims := make([]int, len(ti.Shape))
	for i, d := range ti.Shape {
		dims[i] = int(d)
	}
	slices.Reverse(dims)
	lastIdx := len(dims) - 1
	dims[lastIdx] = (dims[lastIdx] / bs) * ts
	return dtypes.Uint8, dims
}
