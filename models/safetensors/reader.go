package safetensors

import (
	"io"
	"iter"
	"os"
	"sync"

	"github.com/edsrzf/mmap-go"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/pkg/errors"
)

// TensorReader provides access to tensor data via mmap or standard file I/O.
type TensorReader struct {
	mmapBuf    mmap.MMap
	file       *os.File
	fileSize   int64
	dataOffset int64
	Header     *Header
}

// NewTensorReader creates a new TensorReader for a specific .safetensors file.
// It uses mmap if m.UseMMap() is true, or standard file I/O (ReadAt) otherwise.
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

	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, errors.Wrapf(err, "failed to stat %s", localPath)
	}
	fileSize := fi.Size()

	var mmapBuf mmap.MMap
	if m.UseMMap() && fileSize > 0 {
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
		fileSize:   fileSize,
		dataOffset: dataOffset,
		Header:     header,
	}, nil
}

// Close closes the underlying file and unmaps the memory-mapped buffer (if any).
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

// readTensorFromFileAt reads a tensor of the given shape from r at tensorOffset.
// For host (nil) or shared-buffer backends, it reads directly into the tensor's memory without an intermediate buffer.
// For non-shared-buffer backends (e.g., discrete accelerators), it reads into scratchBuf (growing it if needed)
// and transfers it to the backend via tensors.FromRaw.
func readTensorFromFileAt(backend compute.Backend, r io.ReaderAt, tensorName string, shape shapes.Shape, tensorOffset int64, scratchBuf *[]byte) (*tensors.Tensor, error) {
	expectedBytes := int(shape.ByteSize())
	if backend == nil || backend.HasSharedBuffers() {
		t, err := tensors.FromShapeForBackend(backend, 0, shape)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to allocate tensor %q (%s)", tensorName, shape)
		}
		if expectedBytes == 0 {
			return t, nil
		}
		var readErr error
		t.MutableBytes(func(buf []byte) {
			n, err := r.ReadAt(buf, tensorOffset)
			if err != nil && err != io.EOF {
				readErr = errors.Wrapf(err, "failed to read tensor %q (%s) from file", tensorName, shape)
			} else if n != len(buf) {
				readErr = errors.Errorf("short read for tensor %q (%s): got %d bytes, expected %d", tensorName, shape, n, len(buf))
			}
		})
		if readErr != nil {
			t.FinalizeAll()
			return nil, readErr
		}
		return t, nil
	}

	// Non-shared-buffer backend (e.g. discrete GPU): read into a temporary buffer and upload via FromRaw.
	var buf []byte
	if scratchBuf != nil {
		if cap(*scratchBuf) < expectedBytes {
			*scratchBuf = make([]byte, expectedBytes)
		} else {
			*scratchBuf = (*scratchBuf)[:expectedBytes]
		}
		buf = *scratchBuf
	} else {
		buf = make([]byte, expectedBytes)
	}
	if expectedBytes > 0 {
		n, err := r.ReadAt(buf, tensorOffset)
		if err != nil && err != io.EOF {
			return nil, errors.Wrapf(err, "failed to read tensor %q (%s) from file", tensorName, shape)
		} else if n != expectedBytes {
			return nil, errors.Errorf("short read for tensor %q (%s): got %d bytes, expected %d", tensorName, shape, n, expectedBytes)
		}
	}
	t, err := tensors.FromRaw(backend, 0, shape, buf)
	if err != nil {
		return nil, errors.WithMessagef(err, "failed to create tensor %q (%s) from bytes", tensorName, shape)
	}
	return t, nil
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

	if mr.mmapBuf == nil && mr.file == nil {
		return nil, errors.New("file is not open or mmaped")
	}

	tensorOffset := mr.dataOffset + meta.DataOffsets[0]
	tensorEnd := mr.dataOffset + meta.DataOffsets[1]

	expectedBytes := int64(shape.ByteSize())
	if tensorEnd-tensorOffset != expectedBytes {
		return nil, errors.Errorf("tensor shape %s expected %d bytes, but got %d bytes in file", shape, expectedBytes, tensorEnd-tensorOffset)
	}

	dataSize := mr.fileSize
	if mr.mmapBuf != nil {
		dataSize = int64(len(mr.mmapBuf))
	}

	// data_offsets come from the (untrusted) file header, so validate they fall
	// within the file/mapped data before slicing or reading: otherwise a crafted header makes
	// mr.mmapBuf[tensorOffset:tensorEnd] panic with a slice-out-of-range or reads out of bounds.
	if tensorOffset < 0 || tensorEnd < tensorOffset || tensorEnd > dataSize {
		return nil, errors.Errorf("tensor %q data offsets [%d:%d] out of range for %d-byte data section", tensorName, tensorOffset, tensorEnd, dataSize)
	}

	if mr.mmapBuf != nil {
		readBuffer := mr.mmapBuf[tensorOffset:tensorEnd]
		t, err := tensors.FromRaw(backend, 0, shape, readBuffer)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to create tensor %q (%s) from bytes", tensorName, shape)
		}
		return t, nil
	}

	return readTensorFromFileAt(backend, mr.file, tensorName, shape, tensorOffset, nil)
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
			offset     int64
		}

		chParse := make(chan tensorData, 10)
		wg.Go(func() {
			defer close(chParse)
			dataSize := mr.fileSize
			if mr.mmapBuf != nil {
				dataSize = int64(len(mr.mmapBuf))
			}

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

				if tensorOffset < 0 || tensorEnd < tensorOffset || tensorEnd > dataSize {
					select {
					case chParse <- tensorData{err: errors.Errorf("tensor %q data offsets [%d:%d] out of range for %d-byte data section", name, tensorOffset, tensorEnd, dataSize)}:
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
					offset:     tensorOffset,
				}:
				}
			}
		})

		chOut := make(chan tensorData, 10)
		wg.Go(func() {
			defer close(chOut)
			var scratchBuf []byte
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

					switch {
					case mr.mmapBuf != nil:
						data.tensor, data.err = tensors.FromRaw(backend, 0, data.shape, data.readBuffer)
						if data.err != nil {
							data.err = errors.WithMessagef(data.err, "failed to create tensor %q (%s) from bytes", data.name, data.shape)
						}
					case mr.file != nil:
						data.tensor, data.err = readTensorFromFileAt(backend, mr.file, data.name, data.shape, data.offset, &scratchBuf)
					default:
						data.err = errors.New("file is not open or mmaped")
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
