package gguf

import (
	"io"
	"os"
	"unsafe"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
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
		return nil, errors.Wrapf(err, "gguf: open %s", gguf.Path())
	}
	return &Reader{file: f, gguf: gguf}, nil
}

// Close closes the underlying file.
func (r *Reader) Close() error {
	return r.file.Close()
}

// ReadTensor reads a tensor by name:
// native types (F32, F16, BF16, I8, etc.) are loaded directly;
// GGUF quantized types are dequantized to Float32.
func (r *Reader) ReadTensor(backend backends.Backend, tensorName string) (*tensors.Tensor, error) {
	info, ok := r.gguf.GetTensorInfo(tensorName)
	if !ok {
		return nil, errors.Errorf("gguf: tensor %q not found", tensorName)
	}
	dtype, dims := info.GoMLXShape()
	shape := shapes.Make(dtype, dims...)
	t, err := tensors.FromShapeForBackend(backend, 0, shape)
	if err != nil {
		return nil, errors.Wrapf(err, "gguf: failed to create tensor %q with shape %s", tensorName, shape)
	}
	tensorOffset := r.gguf.DataOffset() + int64(info.Offset)

	if info.Type.IsQuantized() {
		err := r.readQuantizedTensor(info, tensorOffset, t)
		if err != nil {
			return nil, err
		}
		return t, nil
	}

	// Native type: direct read into tensor memory -- it assumes current architecture uses
	// the same number formats (same byte-endianness and float representation)
	var readErr error
	t.MutableBytes(func(data []byte) {
		n, err := r.file.ReadAt(data, tensorOffset)
		if err != nil && err != io.EOF {
			readErr = errors.WithStack(err)
		} else if n != len(data) {
			readErr = errors.Errorf("short read: got %d bytes, expected %d", n, len(data))
		}
	})
	if readErr != nil {
		return nil, errors.WithMessagef(readErr, "gguf: read tensor %q", tensorName)
	}

	// If backend is configured, make sure to materialize it on-device and free the local copy.
	if backend != nil {
		err := t.ToDevice(backend, 0)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to move tensor %q (%s) to backend's device #0", tensorName, t.Shape())
		}
	}

	return t, nil
}

// readQuantizedTensor on-the-fly converts the quantized stored values to float32.
func (r *Reader) readQuantizedTensor(info TensorInfo, tensorOffset int64, output *tensors.Tensor) error {
	// Quantized type: read raw bytes, then dequantize into float32 tensor.
	dequant, err := getDequantFunc(info.Type)
	if err != nil {
		return errors.Wrapf(err, "gguf: tensor %q", info.Name)
	}

	rawSize := info.NumBytes()
	rawBuf := make([]byte, rawSize)
	n, err := r.file.ReadAt(rawBuf, tensorOffset)
	if err != nil && err != io.EOF {
		return errors.Wrapf(err, "gguf: read raw tensor %q", info.Name)
	}
	if n != len(rawBuf) {
		return errors.Errorf("gguf: read raw tensor %q: short read: got %d bytes, expected %d", info.Name, n, len(rawBuf))
	}

	blockSize := info.Type.BlockSize()
	typeSize := info.Type.TypeSize()
	nElements := int(info.NumElements())

	var dequantErr error
	output.MutableFlatData(func(flatAny any) {
		dst, ok := flatAny.([]float32)
		if !ok {
			dequantErr = errors.Errorf("tensor %q: expected []float32, got %T", info.Name, flatAny)
			return
		}
		if len(dst) != nElements {
			dequantErr = errors.Errorf("tensor %q: expected %d float32 elements, got buffer for %d",
				info.Name, nElements, len(dst))
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
		return errors.WithMessagef(dequantErr, "gguf: dequantizing tensor %q", info.Name)
	}
	return nil
}

// ReadTensorRaw reads the raw bytes for a tensor without dequantization.
func (r *Reader) ReadTensorRaw(tensorName string) ([]byte, *TensorInfo, error) {
	info, ok := r.gguf.GetTensorInfo(tensorName)
	if !ok {
		return nil, nil, errors.Errorf("gguf: tensor %q not found", tensorName)
	}

	rawSize := info.NumBytes()
	buf := make([]byte, rawSize)
	tensorOffset := r.gguf.DataOffset() + int64(info.Offset)
	n, err := r.file.ReadAt(buf, tensorOffset)
	if err != nil && err != io.EOF {
		return nil, nil, errors.Wrapf(err, "gguf: read raw tensor %q", tensorName)
	}
	if n != len(buf) {
		return nil, nil, errors.Errorf("gguf: read raw tensor %q: short read: got %d bytes, expected %d", tensorName, n, len(buf))
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
