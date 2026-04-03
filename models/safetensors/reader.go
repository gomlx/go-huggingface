package safetensors

import (
	"io"
	"os"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

// TensorReader provides streaming access to tensor data via io.ReadSeeker.
type TensorReader struct {
	reader        io.ReadSeeker
	dataOffset    int64
	currentOffset int64
	Header        *Header
}

// NewTensorReader creates a new TensorReader for a specific .safetensors file.
func (m *Model) NewTensorReader(fileName string) (*TensorReader, error) {
	localPath, err := m.Repo.DownloadFile(fileName)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to download %s", fileName)
	}

	header, dataOffset, err := m.parseHeader(localPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to parse header for %s", localPath)
	}

	// Open file for reading
	f, err := os.Open(localPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open %s", localPath)
	}

	// Create TensorReader
	return &TensorReader{
		reader:     f,
		dataOffset: dataOffset,
		Header:     header,
	}, nil
}

// Close closes the underlying file if it implements io.Closer.
func (sr *TensorReader) Close() error {
	if closer, ok := sr.reader.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}

// ReadTensor reads a tensor by name from the file.
func (mr *TensorReader) ReadTensor(backend backends.Backend, tensorName string) (*tensors.Tensor, error) {
	meta, ok := mr.Header.Tensors[tensorName]
	if !ok {
		return nil, errors.Errorf("tensor %s not found", tensorName)
	}

	// Convert dtype
	dtype, err := dtypeToGoMLX(meta.Dtype)
	if err != nil {
		return nil, err
	}

	// Convert shape to ints
	shape := shapes.Make(dtype, meta.Shape...)
	t, err := tensors.FromShapeForBackend(backend, shape)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to create tensor %q with shape %s", tensorName, shape)
	}

	// Read directly into tensor memory
	tensorOffset := mr.dataOffset + meta.DataOffsets[0]
	var readErr error
	t.MutableBytes(func(data []byte) {
		expectedBytes := int64(t.Shape().Size()) * int64(dtype.Size())
		if int64(len(data)) != expectedBytes {
			readErr = errors.Errorf("tensor shape %s expected %d bytes, but got %d bytes", t.Shape(), expectedBytes, len(data))
			return
		}
		if tensorOffset != mr.currentOffset {
			_, err := mr.reader.Seek(tensorOffset, io.SeekStart)
			if err != nil {
				readErr = errors.Wrapf(err, "failed to seek to offset %d for tensor %s", tensorOffset, tensorName)
				return
			}
			mr.currentOffset = tensorOffset
		}
		var n int
		n, readErr = io.ReadFull(mr.reader, data)
		mr.currentOffset += int64(n)
		if readErr != nil && readErr != io.EOF {
			readErr = errors.Wrapf(readErr, "failed to read tensor %s", tensorName)
		}
	})
	if readErr != nil {
		return nil, readErr
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
