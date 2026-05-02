package safetensors

import (
	"iter"
	"os"
	"sync"

	"github.com/edsrzf/mmap-go"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

// TensorReader provides memory-mapped access to tensor data via mmap.
type TensorReader struct {
	mmapBuf    mmap.MMap
	file       *os.File
	dataOffset int64
	Header     *Header
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

	var mmapBuf mmap.MMap
	fi, err := f.Stat()
	if err == nil && fi.Size() > 0 {
		mmapBuf, err = mmap.Map(f, mmap.RDONLY, 0)
		if err != nil {
			f.Close()
			return nil, errors.Wrapf(err, "failed to mmap %s", localPath)
		}
	}

	// Create TensorReader
	return &TensorReader{
		mmapBuf:    mmapBuf,
		file:       f,
		dataOffset: dataOffset,
		Header:     header,
	}, nil
}

// Close closes the underlying file and unmaps the memory-mapped buffer.
func (sr *TensorReader) Close() error {
	var err1, err2 error
	if sr.mmapBuf != nil {
		err1 = sr.mmapBuf.Unmap()
		sr.mmapBuf = nil
	}
	if sr.file != nil {
		err2 = sr.file.Close()
		sr.file = nil
	}
	if err1 != nil {
		return err1
	}
	return err2
}

// ReadTensor reads a tensor by name from the file.
func (mr *TensorReader) ReadTensor(backend compute.Backend, tensorName string) (*tensors.Tensor, error) {
	meta, ok := mr.Header.Tensors[tensorName]
	if !ok {
		return nil, errors.Errorf("tensor %s not found", tensorName)
	}

	// Create shape.
	shape, err := meta.GoMLXShape()
	if err != nil {
		return nil, err
	}

	if mr.mmapBuf == nil {
		return nil, errors.New("file is not mmaped")
	}

	// Get bytes directly from memory-mapped file
	tensorOffset := mr.dataOffset + meta.DataOffsets[0]
	tensorEnd := mr.dataOffset + meta.DataOffsets[1]

	expectedBytes := int64(shape.ByteSize())
	if tensorEnd-tensorOffset != expectedBytes {
		return nil, errors.Errorf("tensor shape %s expected %d bytes, but got %d bytes in file", shape, expectedBytes, tensorEnd-tensorOffset)
	}

	readBuffer := mr.mmapBuf[tensorOffset:tensorEnd]

	t, err := tensors.FromRaw(backend, 0, shape, readBuffer)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to create tensor %q (%s) from bytes", tensorName, shape)
	}

	return t, nil
}

// IterTensors reads multiple tensors from the file, yielding them one by one.
// It uses a 2-stage pipeline (parse, upload to device) so that while a tensor
// is being parsed, the previous one is being moved to device in parallel.
func (mr *TensorReader) IterTensors(backend compute.Backend, tensorNames []string) iter.Seq2[TensorAndName, error] {
	return func(yield func(TensorAndName, error) bool) {
		done := make(chan struct{})
		var wg sync.WaitGroup

		defer wg.Wait()
		defer close(done)

		type tensorData struct {
			name       string
			tensor     *tensors.Tensor
			err        error
			shape      shapes.Shape
			readBuffer []byte
		}

		chParse := make(chan tensorData, 10)
		wg.Go(func() {
			defer close(chParse)
			for _, name := range tensorNames {
				select {
				case <-done:
					return
				default:
				}

				meta, ok := mr.Header.Tensors[name]
				if !ok {
					select {
					case chParse <- tensorData{err: errors.Errorf("tensor %s not found", name)}:
					case <-done:
					}
					return
				}

				shape, err := meta.GoMLXShape()
				if err != nil {
					select {
					case chParse <- tensorData{err: err}:
					case <-done:
					}
					return
				}

				tensorOffset := mr.dataOffset + meta.DataOffsets[0]
				tensorEnd := mr.dataOffset + meta.DataOffsets[1]
				expectedBytes := int64(shape.ByteSize())
				if tensorEnd-tensorOffset != expectedBytes {
					select {
					case chParse <- tensorData{err: errors.Errorf("tensor shape %s expected %d bytes, but got %d bytes in file", shape, expectedBytes, tensorEnd-tensorOffset)}:
					case <-done:
					}
					return
				}

				var readBuffer []byte
				if mr.mmapBuf != nil {
					readBuffer = mr.mmapBuf[tensorOffset:tensorEnd]
				}

				select {
				case <-done:
					return
				case chParse <- tensorData{
					name:       name,
					shape:      shape,
					readBuffer: readBuffer,
				}:
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
				case data, ok := <-chParse:
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

					if mr.mmapBuf == nil {
						data.err = errors.New("file is not mmaped")
					} else {
						data.tensor, data.err = tensors.FromRaw(backend, 0, data.shape, data.readBuffer)
						if data.err != nil {
							data.err = errors.WithMessagef(data.err, "failed to create tensor %q (%s) from bytes", data.name, data.shape)
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
