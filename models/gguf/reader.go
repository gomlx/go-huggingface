package gguf

import (
	"fmt"
	"io"
	"os"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// Reader provides random-access to tensor data in a GGUF file.
type Reader struct {
	file *os.File
	gguf *File
}

// NewReader opens a reader for the given parsed GGUF file.
func NewReader(gguf *File) (*Reader, error) {
	f, err := os.Open(gguf.Path())
	if err != nil {
		return nil, fmt.Errorf("gguf: open %s: %w", gguf.Path(), err)
	}
	return &Reader{file: f, gguf: gguf}, nil
}

// Close closes the underlying file.
func (r *Reader) Close() error {
	return r.file.Close()
}

// ReadTensor reads a tensor by name, dequantizing quantized data to Float32.
// Native types (F32, F16, BF16, I8, etc.) are loaded directly.
func (r *Reader) ReadTensor(tensorName string) (*tensors.Tensor, error) {
	info, ok := r.gguf.GetTensorInfo(tensorName)
	if !ok {
		return nil, fmt.Errorf("gguf: tensor %q not found", tensorName)
	}

	dtype, dims := info.GoMLXShape()
	t := tensors.FromShape(shapes.Make(dtype, dims...))

	tensorOffset := r.gguf.DataOffset() + int64(info.Offset)

	if !info.Type.IsQuantized() {
		// Native type: direct copy into tensor memory.
		var readErr error
		t.MutableBytes(func(data []byte) {
			n, err := r.file.ReadAt(data, tensorOffset)
			if err != nil && err != io.EOF {
				readErr = err
			} else if n != len(data) {
				readErr = fmt.Errorf("short read: got %d bytes, expected %d", n, len(data))
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
	n, err := r.file.ReadAt(rawBuf, tensorOffset)
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("gguf: read raw tensor %q: %w", tensorName, err)
	}
	if n != len(rawBuf) {
		return nil, fmt.Errorf("gguf: read raw tensor %q: short read: got %d bytes, expected %d", tensorName, n, len(rawBuf))
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
func (r *Reader) ReadTensorRaw(tensorName string) ([]byte, *TensorInfo, error) {
	info, ok := r.gguf.GetTensorInfo(tensorName)
	if !ok {
		return nil, nil, fmt.Errorf("gguf: tensor %q not found", tensorName)
	}

	rawSize := info.NumBytes()
	buf := make([]byte, rawSize)
	tensorOffset := r.gguf.DataOffset() + int64(info.Offset)
	n, err := r.file.ReadAt(buf, tensorOffset)
	if err != nil && err != io.EOF {
		return nil, nil, fmt.Errorf("gguf: read raw tensor %q: %w", tensorName, err)
	}
	if n != len(buf) {
		return nil, nil, fmt.Errorf("gguf: read raw tensor %q: short read: got %d bytes, expected %d", tensorName, n, len(buf))
	}

	return buf, &info, nil
}

// bytesToFloat32 reinterprets a byte slice as a float32 slice.
// The byte slice length must be a multiple of 4.
//
// Safety: This relies on Go's heap allocation guarantee of at least 8-byte alignment
// for the backing array. The caller (tensors.MutableBytes) provides heap-allocated memory.
// GGUF is a little-endian format; this reinterpretation is only correct on little-endian
// architectures (x86-64, arm64), which covers all platforms Go currently targets.
func bytesToFloat32(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&b[0])), len(b)/4)
}
