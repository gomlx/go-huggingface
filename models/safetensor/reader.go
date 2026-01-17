package safetensor

import (
	"io"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
	"golang.org/x/exp/mmap"
)

// MMapReader provides streaming access to tensor data via io.ReaderAt.
type MMapReader struct {
	reader     *mmap.ReaderAt
	dataOffset int64
	header     *Header
	meta       *TensorMetadata
}

// ReadAt implements io.ReaderAt for the tensor data.
func (sr *MMapReader) ReadAt(p []byte, off int64) (n int, err error) {
	tensorOffset := sr.dataOffset + sr.meta.DataOffsets[0] + off
	return sr.reader.ReadAt(p, tensorOffset)
}

// Len returns the size of the tensor data in bytes.
func (sr *MMapReader) Len() int {
	return int(sr.meta.DataOffsets[1] - sr.meta.DataOffsets[0])
}

// Close closes the underlying memory-mapped file.
func (sr *MMapReader) Close() error {
	return sr.reader.Close()
}

// Metadata returns the tensor metadata.
func (sr *MMapReader) Metadata() *TensorMetadata {
	return sr.meta
}

// ReadTensor reads a tensor by name from the memory-mapped file.
func (mr *MMapReader) ReadTensor(tensorName string) (*tensors.Tensor, error) {
	meta, ok := mr.header.Tensors[tensorName]
	if !ok {
		return nil, errors.Errorf("tensor %s not found", tensorName)
	}

	// Convert dtype
	dtype, err := dtypeToGoMLX(meta.Dtype)
	if err != nil {
		return nil, err
	}

	// Convert shape to ints
	t := tensors.FromShape(shapes.Make(dtype, meta.Shape...))

	// Read from mmap directly into tensor memory
	tensorOffset := mr.dataOffset + meta.DataOffsets[0]
	var readErr error
	t.MutableBytes(func(data []byte) {
		expectedBytes := int64(t.Shape().Size()) * int64(dtype.Size())
		if int64(len(data)) != expectedBytes {
			readErr = errors.Errorf("tensor shape %s expected %d bytes, but got %d bytes", t.Shape(), expectedBytes, len(data))
			return
		}
		_, readErr = mr.reader.ReadAt(data, tensorOffset)
		if readErr != nil && readErr != io.EOF {
			readErr = errors.Wrapf(readErr, "failed to read tensor %s", tensorName)
		}
	})
	if readErr != nil {
		return nil, readErr
	}

	return t, nil
}
