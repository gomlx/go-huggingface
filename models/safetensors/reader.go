package safetensors

import (
	"io"
	"iter"
	"os"
	"sync"

	"github.com/gomlx/gomlx/backends"
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

	// Create shape & tensor.
	shape, err := meta.GoMLXShape()
	if err != nil {
		return nil, err
	}
	t, err := tensors.FromShapeForBackend(backend, shape)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to create tensor %q with shape %s", tensorName, shape)
	}

	// Read directly into tensor memory
	tensorOffset := mr.dataOffset + meta.DataOffsets[0]
	var readErr error
	t.MutableBytes(func(data []byte) {
		expectedBytes := int64(t.Shape().Memory())
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

// IterTensors reads multiple tensors from the file, yielding them one by one.
// It uses a 3-stage pipeline (create, read IO, move to device) so that while a tensor
// is being read from IO, the previous one is being moved to device and the next one
// is being created in parallel.
func (mr *TensorReader) IterTensors(backend backends.Backend, tensorNames []string) iter.Seq2[TensorAndName, error] {
	return func(yield func(TensorAndName, error) bool) {
		done := make(chan struct{})
		var wg sync.WaitGroup

		defer wg.Wait()
		defer close(done)

		type tensorData struct {
			name          string
			tensor        *tensors.Tensor
			err           error
			tensorOffset  int64
			expectedBytes int64
		}

		chRead := make(chan tensorData, 10)
		wg.Go(func() {
			defer close(chRead)
			for _, name := range tensorNames {
				select {
				case <-done:
					return
				default:
				}

				meta, ok := mr.Header.Tensors[name]
				if !ok {
					select {
					case chRead <- tensorData{err: errors.Errorf("tensor %s not found", name)}:
					case <-done:
					}
					return
				}

				shape, err := meta.GoMLXShape()
				if err != nil {
					select {
					case chRead <- tensorData{err: err}:
					case <-done:
					}
					return
				}

				t, err := tensors.FromShapeForBackend(backend, shape)
				if err != nil {
					select {
					case chRead <- tensorData{err: errors.Wrapf(err, "failed to create tensor %q with shape %s", name, shape)}:
					case <-done:
					}
					return
				}

				expectedBytes := int64(shape.Memory())
				tensorOffset := mr.dataOffset + meta.DataOffsets[0]

				select {
				case <-done:
					return
				case chRead <- tensorData{
					name:          name,
					tensor:        t,
					tensorOffset:  tensorOffset,
					expectedBytes: expectedBytes,
				}:
				}
			}
		})

		chDevice := make(chan tensorData, 10)
		wg.Go(func() {
			defer close(chDevice)
			for {
				select {
				case <-done:
					return
				case data, ok := <-chRead:
					if !ok {
						return
					}
					if data.err != nil {
						select {
						case <-done:
						case chDevice <- data:
						}
						return
					}

					var readErr error
					data.tensor.MutableBytes(func(b []byte) {
						if int64(len(b)) != data.expectedBytes {
							readErr = errors.Errorf("tensor shape %s expected %d bytes, but got %d bytes", data.tensor.Shape(), data.expectedBytes, len(b))
							return
						}
						if data.tensorOffset != mr.currentOffset {
							_, err := mr.reader.Seek(data.tensorOffset, io.SeekStart)
							if err != nil {
								readErr = errors.Wrapf(err, "failed to seek to offset %d for tensor %s", data.tensorOffset, data.name)
								return
							}
							mr.currentOffset = data.tensorOffset
						}
						var n int
						n, readErr = io.ReadFull(mr.reader, b)
						mr.currentOffset += int64(n)
						if readErr != nil && readErr != io.EOF {
							readErr = errors.Wrapf(readErr, "failed to read tensor %s", data.name)
						} else {
							readErr = nil
						}
					})

					if readErr != nil {
						data.err = readErr
					}

					select {
					case <-done:
						return
					case chDevice <- data:
						if data.err != nil {
							return
						}
					}
				}
			}
		})

		chOut := make(chan tensorData, 10)
		wg.Go(func() {
			defer close(chOut)
			for {
				select {
				case <-done:
					return
				case data, ok := <-chDevice:
					if !ok {
						return
					}
					if data.err != nil {
						select {
						case <-done:
						case chOut <- data:
						}
						return
					}

					if backend != nil {
						err := data.tensor.ToDevice(backend, 0)
						if err != nil {
							data.err = errors.WithMessagef(err, "failed to move tensor %q (%s) to backend's device #0", data.name, data.tensor.Shape())
						}
					}

					select {
					case <-done:
						return
					case chOut <- data:
						if data.err != nil {
							return
						}
					}
				}
			}
		})

		defer wg.Wait()
		for data := range chOut {
			if data.err != nil {
				yield(TensorAndName{}, data.err)
				return
			}
			if !yield(TensorAndName{Name: data.name, Tensor: data.tensor}, nil) {
				return
			}
		}
	}
}
