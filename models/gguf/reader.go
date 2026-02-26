package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"golang.org/x/exp/mmap"
)

// MMapReader provides memory-mapped access to tensor data in a GGUF file.
type MMapReader struct {
	reader     *mmap.ReaderAt
	file       *File
	dataOffset int64
}

// NewMMapReader opens a memory-mapped reader for the GGUF file.
func NewMMapReader(path string, file *File) (*MMapReader, error) {
	reader, err := mmap.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: mmap %s: %w", path, err)
	}
	return &MMapReader{
		reader:     reader,
		file:       file,
		dataOffset: file.DataOffset(),
	}, nil
}

// Close closes the underlying memory-mapped file.
func (mr *MMapReader) Close() error {
	return mr.reader.Close()
}

// ReadTensor reads a tensor by name, dequantizing quantized data to Float32.
// Native types (F32, F16, BF16, I8, etc.) are loaded directly.
func (mr *MMapReader) ReadTensor(tensorName string) (*tensors.Tensor, error) {
	info, ok := mr.file.GetTensorInfo(tensorName)
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", tensorName)
	}

	dtype, dims := info.GoMLXShape()
	t := tensors.FromShape(shapes.Make(dtype, dims...))

	tensorOffset := mr.dataOffset + int64(info.Offset)

	if !info.Type.IsQuantized() {
		// Native type: direct copy into tensor memory.
		var readErr error
		t.MutableBytes(func(data []byte) {
			_, readErr = mr.reader.ReadAt(data, tensorOffset)
			if readErr == io.EOF {
				readErr = nil
			}
		})
		if readErr != nil {
			return nil, fmt.Errorf("gguf: read tensor %q: %w", tensorName, readErr)
		}
		return t, nil
	}

	// Quantized type: read raw bytes, then dequantize into float32 tensor.
	dequant, err := getDequantFunc(info.Type)
	if err != nil {
		return nil, fmt.Errorf("gguf: tensor %q: %w", tensorName, err)
	}

	rawSize := info.NumBytes()
	rawBuf := make([]byte, rawSize)
	if _, err := mr.reader.ReadAt(rawBuf, tensorOffset); err != nil && err != io.EOF {
		return nil, fmt.Errorf("gguf: read raw tensor %q: %w", tensorName, err)
	}

	blockSize := info.Type.BlockSize()
	typeSize := info.Type.TypeSize()
	nElements := int(info.NumElements())

	var dequantErr error
	t.MutableBytes(func(data []byte) {
		dst := bytesToFloat32(data)
		if len(dst) != nElements {
			dequantErr = fmt.Errorf("tensor %q: expected %d float32 elements, got buffer for %d",
				tensorName, nElements, len(dst))
			return
		}

		nBlocks := nElements / blockSize
		for b := range nBlocks {
			srcStart := b * typeSize
			srcEnd := srcStart + typeSize
			dstStart := b * blockSize
			dstEnd := dstStart + blockSize
			dequant(rawBuf[srcStart:srcEnd], dst[dstStart:dstEnd])
		}
	})
	if dequantErr != nil {
		return nil, fmt.Errorf("gguf: dequant tensor %q: %w", tensorName, dequantErr)
	}

	return t, nil
}

// ReadTensorRaw reads the raw bytes for a tensor without dequantization.
func (mr *MMapReader) ReadTensorRaw(tensorName string) ([]byte, *TensorInfo, error) {
	info, ok := mr.file.GetTensorInfo(tensorName)
	if !ok {
		return nil, nil, fmt.Errorf("gguf: tensor %q not found", tensorName)
	}

	rawSize := info.NumBytes()
	buf := make([]byte, rawSize)
	tensorOffset := mr.dataOffset + int64(info.Offset)
	if _, err := mr.reader.ReadAt(buf, tensorOffset); err != nil && err != io.EOF {
		return nil, nil, fmt.Errorf("gguf: read raw tensor %q: %w", tensorName, err)
	}

	return buf, &info, nil
}

// bytesToFloat32 reinterprets a byte slice as a float32 slice.
// The byte slice length must be a multiple of 4.
func bytesToFloat32(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}

// nativeTensorTypeToDType maps non-quantized GGUF tensor types to GoMLX dtypes.
// Returns dtypes.InvalidDType for quantized types.
func nativeTensorTypeToDType(t TensorType) dtypes.DType {
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
		return dtypes.InvalidDType
	}
}

// bfloat16ToFloat32Slice converts a []byte of BF16 values to []float32 in place.
// This is unused currently (BF16 is handled natively by GoMLX) but available
// for users who need explicit conversion.
func bfloat16ToFloat32Slice(src []byte, dst []float32) {
	for i := range dst {
		bits := binary.LittleEndian.Uint16(src[i*2 : i*2+2])
		dst[i] = math.Float32frombits(uint32(bits) << 16)
	}
}
