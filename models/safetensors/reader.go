package safetensors

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
	Header     *Header
}

// NewMMapReader creates a new MMapReader for a specific .safetensors file.
func (m *Model) NewMMapReader(fileName string) (*MMapReader, error) {
	localPath, err := m.Repo.DownloadFile(fileName)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to download %s", fileName)
	}

	header, dataOffset, err := m.parseHeader(localPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to parse header for %s", localPath)
	}

	// Open mmap for reading
	reader, err := mmap.Open(localPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to mmap %s", localPath)
	}

	// Create MMapReader
	return &MMapReader{
		reader:     reader,
		dataOffset: dataOffset,
		Header:     header,
	}, nil
}

// Close closes the underlying memory-mapped file.
func (sr *MMapReader) Close() error {
	return sr.reader.Close()
}

// ReadTensor reads a tensor by name from the memory-mapped file.
func (mr *MMapReader) ReadTensor(tensorName string) (*tensors.Tensor, error) {
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
