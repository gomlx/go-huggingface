package safetensors

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/edsrzf/mmap-go"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

type iterTensorData struct {
	name           string
	tensor         *tensors.Tensor
	shape          shapes.Shape
	readBuffer     []byte
	mmapBuf        mmap.MMap
	err            error
	f              *os.File
	closeFileOnEnd bool
	tensorOffset   int64
	currentOffset  int64
	expectedBytes  int64
}

// IterTensorsFromRepo iterates over all tensors in the repository.
//
// Tensors are loaded into the backend directly (e.g.: GPU, or a shared memory tensor on CPU, etc).
// If the backend is nil, it instead loads them in host memory.
func IterTensorsFromRepo(backend backends.Backend, repo *hub.Repo) func(yield func(TensorAndName, error) bool) {
	return func(yield func(TensorAndName, error) bool) {
		done := make(chan struct{})
		var wg sync.WaitGroup

		var mmaps []mmap.MMap
		var mmapsMu sync.Mutex

		defer func() {
			wg.Wait()
			mmapsMu.Lock()
			defer mmapsMu.Unlock()
			for _, m := range mmaps {
				if m != nil {
					m.Unmap()
				}
			}
		}()
		defer close(done)

		chRead := make(chan iterTensorData, 10)
		wg.Go(func() { iterFromRepoDownload(backend, repo, done, chRead, &mmaps, &mmapsMu) })

		chDevice := make(chan iterTensorData, 10)
		wg.Go(func() { iterFromRepoRead(done, chRead, chDevice) })

		chOut := make(chan iterTensorData, 100)
		wg.Go(func() { iterFromRepoToDevice(backend, done, chDevice, chOut) })

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

func iterFromRepoDownload(backend backends.Backend, repo *hub.Repo, done <-chan struct{}, chRead chan<- iterTensorData, mmaps *[]mmap.MMap, mmapsMu *sync.Mutex) {
	start := time.Now()
	var waitTime time.Duration
	if klog.V(1).Enabled() {
		defer func() {
			klog.Infof(" - Repo download / Mmap open time: %s (of which %s is waiting)", time.Since(start), waitTime)
		}()
	}
	defer close(chRead)

	var openFile *os.File
	var sentFileOwnership bool

	reportErrFn := func(err error) {
		waitStart := time.Now()
		select {
		case <-done:
		case chRead <- iterTensorData{err: err}:
			waitTime += time.Since(waitStart)
		}
	}

	defer func() {
		if openFile != nil && !sentFileOwnership {
			openFile.Close()
		}
	}()

	var waitStart time.Time
	for filename, err := range repo.IterFileNames() {
		select {
		case <-done:
			return
		default:
		}

		if err != nil {
			reportErrFn(errors.Wrap(err, "failed to iterate repo files"))
			return
		}

		if filepath.Ext(filename) != ".safetensors" && !strings.HasSuffix(filename, ".safetensors") {
			continue
		}

		localPath, err := repo.DownloadFile(filename)
		if err != nil {
			reportErrFn(errors.Wrapf(err, "failed to download %s", filename))
			return
		}

		header, dataOffset, err := (*Model)(nil).parseHeader(localPath)
		if err != nil {
			reportErrFn(errors.Wrapf(err, "failed to parse header for %s", localPath))
			return
		}

		var tensorNames []string
		for name := range header.Tensors {
			tensorNames = append(tensorNames, name)
		}
		tensorNames = sortTensorsByOffset(tensorNames, header)

		if openFile != nil && !sentFileOwnership {
			openFile.Close()
		}

		openFile, err = os.Open(localPath)
		sentFileOwnership = false
		if err != nil {
			reportErrFn(errors.Wrapf(err, "failed to open %s", localPath))
			return
		}

		var mmapBuf mmap.MMap
		fi, err := openFile.Stat()
		if err == nil && fi.Size() > 0 {
			mmapBuf, err = mmap.Map(openFile, mmap.RDONLY, 0)
			if err != nil {
				mmapBuf = nil
			} else {
				mmapsMu.Lock()
				*mmaps = append(*mmaps, mmapBuf)
				mmapsMu.Unlock()
			}
		}

		var currentOffset int64
		if len(tensorNames) > 0 {
			currentOffset = dataOffset + header.Tensors[tensorNames[0]].DataOffsets[0]
		}

		for i, name := range tensorNames {
			select {
			case <-done:
				return
			default:
			}

			meta := header.Tensors[name]
			shape, err := meta.GoMLXShape()
			if err != nil {
				reportErrFn(err)
				return
			}

			var t *tensors.Tensor
			var readBuffer []byte
			expectedBytes := int64(shape.Memory())

			if backend != nil && !backend.HasSharedBuffers() {
				if mmapBuf != nil {
					readBuffer = mmapBuf[dataOffset+meta.DataOffsets[0] : dataOffset+meta.DataOffsets[1]]
				} else {
					readBuffer = make([]byte, expectedBytes)
				}
			} else {
				t, err = tensors.FromShapeForBackend(backend, shape)
				if err != nil {
					reportErrFn(errors.Wrapf(err, "failed to create tensor %q with shape %s", name, shape))
					return
				}
			}

			tensorOffset := dataOffset + meta.DataOffsets[0]
			closeFile := (i == len(tensorNames)-1)

			waitStart = time.Now()
			select {
			case <-done:
				waitTime += time.Since(waitStart)
				return
			case chRead <- iterTensorData{
				name:           name,
				tensor:         t,
				shape:          shape,
				readBuffer:     readBuffer,
				mmapBuf:        mmapBuf,
				f:              openFile,
				closeFileOnEnd: closeFile,
				tensorOffset:   tensorOffset,
				currentOffset:  currentOffset,
				expectedBytes:  expectedBytes,
			}:
				waitTime += time.Since(waitStart)
				if closeFile {
					sentFileOwnership = true
				}
			}
			currentOffset = tensorOffset + expectedBytes
		}
	}
}

func iterFromRepoRead(done <-chan struct{}, chRead <-chan iterTensorData, chDevice chan<- iterTensorData) {
	start := time.Now()
	var waitTime time.Duration
	if klog.V(1).Enabled() {
		defer func() {
			klog.Infof("- Read files time: %s (of which %s is waiting)", time.Since(start), waitTime)
		}()
	}

	defer close(chDevice)

	var currentFile *os.File
	var currentData *iterTensorData
	var waitStart time.Time

	defer func() {
		if currentData != nil && currentData.closeFileOnEnd {
			if currentData.f != nil {
				currentData.f.Close()
			}
		}

		for data := range chRead {
			if data.closeFileOnEnd {
				if data.f != nil {
					data.f.Close()
				}
			}
		}
	}()

	for {
		waitStart = time.Now()
		select {
		case <-done:
			waitTime += time.Since(waitStart)
			return
		case data, ok := <-chRead:
			waitTime += time.Since(waitStart)
			if !ok {
				return
			}
			currentData = &data

			if data.err != nil {
				waitStart = time.Now()
				select {
				case <-done:
				case chDevice <- data:
				}
				waitTime += time.Since(waitStart)
				return
			}

			if currentFile != data.f {
				currentFile = data.f
			}

			var readErr error
			if data.f != nil {
				readFn := func(b []byte) {
					if int64(len(b)) != data.expectedBytes {
						readErr = errors.Errorf("tensor shape %s expected %d bytes, but got %d bytes", data.shape, data.expectedBytes, len(b))
						return
					}
					if data.mmapBuf != nil {
						copy(b, data.mmapBuf[data.tensorOffset:data.tensorOffset+data.expectedBytes])
						readErr = nil
					} else {
						if data.tensorOffset != data.currentOffset {
							_, err := data.f.Seek(data.tensorOffset, io.SeekStart)
							if err != nil {
								readErr = errors.Wrapf(err, "failed to seek to offset %d for tensor %s", data.tensorOffset, data.name)
								return
							}
						}
						_, readErr = io.ReadFull(data.f, b)
						if readErr != nil && readErr != io.EOF {
							readErr = errors.Wrapf(readErr, "failed to read tensor %s", data.name)
						} else {
							readErr = nil
						}
					}
				}
				if data.tensor != nil {
					data.tensor.MutableBytes(readFn)
				} else if data.mmapBuf == nil {
					readFn(data.readBuffer)
				}
			}

			if readErr != nil {
				data.err = readErr
			}

			if data.closeFileOnEnd {
				if data.f != nil {
					data.f.Close()
				}
				currentFile = nil
			}

			currentData = nil

			waitStart = time.Now()
			select {
			case <-done:
				waitTime += time.Since(waitStart)
				return
			case chDevice <- data:
				waitTime += time.Since(waitStart)
				if data.err != nil {
					return
				}
			}
		}
	}
}

var MaxParallelBufferTransfers = 4

func iterFromRepoToDevice(backend backends.Backend, done <-chan struct{}, chDevice <-chan iterTensorData, chOut chan<- iterTensorData) {
	defer close(chOut)
	start := time.Now()
	var totalWaitTime atomic.Int64
	if klog.V(1).Enabled() {
		defer func() {
			klog.Infof("- Send to device time: %s (of which %s is waiting, across %d workers)", time.Since(start), time.Duration(totalWaitTime.Load()), MaxParallelBufferTransfers)
		}()
	}

	var wg sync.WaitGroup
	for i := 0; i < MaxParallelBufferTransfers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var waitTime time.Duration
			defer func() {
				totalWaitTime.Add(int64(waitTime))
			}()

			var waitStart time.Time

			for {
				waitStart = time.Now()
				select {
				case <-done:
					waitTime += time.Since(waitStart)
					return
				case data, ok := <-chDevice:
					waitTime += time.Since(waitStart)
					if !ok {
						return
					}
					if data.err != nil {
						waitStart = time.Now()
						select {
						case <-done:
						case chOut <- data:
						}
						waitTime += time.Since(waitStart)
						return
					}

					if backend != nil {
						if data.tensor == nil {
							bytesPtr := unsafe.Pointer(&data.readBuffer[0])
							length := data.shape.Size() / data.shape.DType.ValuesPerStorageUnit()
							flatAny := dtypes.UnsafeAnySliceFromBytes(bytesPtr, data.shape.DType, length)

							backendBuf, err := backend.BufferFromFlatData(0, flatAny, data.shape)
							if err != nil {
								data.err = errors.WithMessagef(err, "failed to create backend buffer for tensor %q (%s)", data.name, data.shape)
							} else {
								data.tensor, err = tensors.FromBuffer(backend, backendBuf)
								if err != nil {
									data.err = errors.WithMessagef(err, "failed to create tensor from buffer for %q (%s)", data.name, data.shape)
								}
							}
						} else {
							err := data.tensor.ToDevice(backend, 0)
							if err != nil {
								data.err = errors.WithMessagef(err, "failed to move tensor %q (%s) to backend's device #0", data.name, data.shape)
							}
						}
					}

					waitStart = time.Now()
					select {
					case <-done:
						waitTime += time.Since(waitStart)
						return
					case chOut <- data:
					}
					waitTime += time.Since(waitStart)
				}
			}
		}()
	}
	wg.Wait()
}
