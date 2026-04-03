package safetensors

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

type iterTensorData struct {
	name           string
	tensor         *tensors.Tensor
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

		defer wg.Wait()
		defer close(done)

		chRead := make(chan iterTensorData, 10)
		wg.Go(func() { iterFromRepoDownload(backend, repo, done, chRead) })

		chDevice := make(chan iterTensorData, 10)
		wg.Go(func() { iterFromRepoRead(done, chRead, chDevice) })

		chOut := make(chan iterTensorData, 3)
		wg.Go(func() { iterFromRepoToDevice(backend, done, chDevice, chOut) })

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

func iterFromRepoDownload(backend backends.Backend, repo *hub.Repo, done <-chan struct{}, chRead chan<- iterTensorData) {
	defer close(chRead)

	var waitTime time.Duration
	if klog.V(1).Enabled() {
		defer func() {
			klog.Infof("- Total wait time not downloading/allocating anything: %s", waitTime)
		}()
	}

	var openFile *os.File
	var sentFileOwnership bool

	defer func() {
		// Close any file that hasn't been sent off by a closeFileOnEnd=true payload
		if openFile != nil && !sentFileOwnership {
			openFile.Close()
		}
	}()

	for filename, err := range repo.IterFileNames() {
		select {
		case <-done:
			return
		default:
		}

		if err != nil {
			start := time.Now()
			select {
			case <-done:
			case chRead <- iterTensorData{err: errors.Wrap(err, "failed to iterate repo files")}:
				waitTime += time.Since(start)
			}
			return
		}

		if filepath.Ext(filename) != ".safetensors" && !strings.HasSuffix(filename, ".safetensors") {
			continue
		}

		localPath, err := repo.DownloadFile(filename)
		if err != nil {
			start := time.Now()
			select {
			case <-done:
			case chRead <- iterTensorData{err: errors.Wrapf(err, "failed to download %s", filename)}:
				waitTime += time.Since(start)
			}
			return
		}

		header, dataOffset, err := (*Model)(nil).parseHeader(localPath)
		if err != nil {
			start := time.Now()
			select {
			case <-done:
			case chRead <- iterTensorData{err: errors.Wrapf(err, "failed to parse header for %s", localPath)}:
				waitTime += time.Since(start)
			}
			return
		}

		var tensorNames []string
		for name := range header.Tensors {
			tensorNames = append(tensorNames, name)
		}
		tensorNames = sortTensorsByOffset(tensorNames, header)

		// Ensure any left over file from a previous iteration without elements is closed
		if openFile != nil && !sentFileOwnership {
			openFile.Close()
		}

		openFile, err = os.Open(localPath)
		sentFileOwnership = false
		if err != nil {
			start := time.Now()
			select {
			case <-done:
			case chRead <- iterTensorData{err: errors.Wrapf(err, "failed to open %s", localPath)}:
				waitTime += time.Since(start)
			}
			return
		}

		var currentOffset int64
		if len(tensorNames) > 0 {
			firstTensorOffset := dataOffset + header.Tensors[tensorNames[0]].DataOffsets[0]
			_, err = openFile.Seek(firstTensorOffset, io.SeekStart)
			if err != nil {
				start := time.Now()
				select {
				case <-done:
				case chRead <- iterTensorData{err: errors.Wrapf(err, "failed to seek in %s", localPath)}:
					waitTime += time.Since(start)
				}
				return
			}
			currentOffset = firstTensorOffset
		}

		for i, name := range tensorNames {
			select {
			case <-done:
				return
			default:
			}

			meta := header.Tensors[name]
			dtype, err := dtypeToGoMLX(meta.Dtype)
			if err != nil {
				start := time.Now()
				select {
				case <-done:
				case chRead <- iterTensorData{err: err}:
					waitTime += time.Since(start)
				}
				return
			}

			shape := shapes.Make(dtype, meta.Shape...)
			t, err := tensors.FromShapeForBackend(backend, shape)
			if err != nil {
				start := time.Now()
				select {
				case <-done:
				case chRead <- iterTensorData{err: errors.Wrapf(err, "failed to create tensor %q with shape %s", name, shape)}:
					waitTime += time.Since(start)
				}
				return
			}

			expectedBytes := int64(shape.Size()) * int64(dtype.Size())
			tensorOffset := dataOffset + meta.DataOffsets[0]
			closeFile := (i == len(tensorNames)-1)

			start := time.Now()
			select {
			case <-done:
				return
			case chRead <- iterTensorData{
				name:           name,
				tensor:         t,
				f:              openFile,
				closeFileOnEnd: closeFile,
				tensorOffset:   tensorOffset,
				currentOffset:  currentOffset,
				expectedBytes:  expectedBytes,
			}:
				waitTime += time.Since(start)
				if closeFile {
					sentFileOwnership = true
				}
			}
			currentOffset = tensorOffset + expectedBytes
		}

		if len(tensorNames) == 0 {
			openFile.Close()
			sentFileOwnership = true
		}
	}
}

func iterFromRepoRead(done <-chan struct{}, chRead <-chan iterTensorData, chDevice chan<- iterTensorData) {
	defer close(chDevice)

	var currentFile *os.File
	var currentData *iterTensorData

	defer func() {
		// Close currently held file if active on abort
		if currentFile != nil {
			currentFile.Close()
		} else if currentData != nil && currentData.closeFileOnEnd && currentData.f != nil {
			currentData.f.Close()
		}

		// Clean-up function that drains the channel
		for data := range chRead {
			if data.closeFileOnEnd && data.f != nil {
				data.f.Close()
			}
		}
	}()

	var waitTime time.Duration
	if klog.V(1).Enabled() {
		defer func() {
			klog.Infof("- Total wait time not reading anything: %s", waitTime)
		}()
	}

	var start time.Time
	for {
		start = time.Now()
		select {
		case <-done:
			return
		case data, ok := <-chRead:
			waitTime += time.Since(start)
			if !ok {
				return
			}
			currentData = &data

			if data.err != nil {
				start = time.Now()
				select {
				case <-done:
				case chDevice <- data:
					waitTime += time.Since(start)
				}
				return
			}

			if currentFile != data.f {
				currentFile = data.f
			}

			var readErr error
			if data.f != nil {
				data.tensor.MutableBytes(func(b []byte) {
					if int64(len(b)) != data.expectedBytes {
						readErr = errors.Errorf("tensor shape %s expected %d bytes, but got %d bytes", data.tensor.Shape(), data.expectedBytes, len(b))
						return
					}
					if data.tensorOffset != data.currentOffset {
						start = time.Now()
						_, err := data.f.Seek(data.tensorOffset, io.SeekStart)
						if err != nil {
							readErr = errors.Wrapf(err, "failed to seek to offset %d for tensor %s", data.tensorOffset, data.name)
							return
						}
						waitTime += time.Since(start)
					}
					_, readErr = io.ReadFull(data.f, b)
					if readErr != nil && readErr != io.EOF {
						readErr = errors.Wrapf(readErr, "failed to read tensor %s", data.name)
					} else {
						readErr = nil
					}
				})
			}

			if readErr != nil {
				data.err = readErr
			}

			if data.closeFileOnEnd && data.f != nil {
				data.f.Close()
				currentFile = nil
			}

			currentData = nil

			start = time.Now()
			select {
			case <-done:
				return
			case chDevice <- data:
				waitTime += time.Since(start)
				if data.err != nil {
					return
				}
			}
		}
	}
}

func iterFromRepoToDevice(backend backends.Backend, done <-chan struct{}, chDevice <-chan iterTensorData, chOut chan<- iterTensorData) {
	defer close(chOut)

	var waitTime time.Duration
	if klog.V(1).Enabled() {
		defer func() {
			klog.Infof("- Total wait time not moving to device: %s", waitTime)
		}()
	}

	var start time.Time
	for {
		start = time.Now()
		select {
		case <-done:
			return
		case data, ok := <-chDevice:
			waitTime += time.Since(start)
			if !ok {
				return
			}
			if data.err != nil {
				start = time.Now()
				select {
				case <-done:
				case chOut <- data:
					waitTime += time.Since(start)
				}
				return
			}

			if backend != nil {
				err := data.tensor.ToDevice(backend, 0)
				if err != nil {
					data.err = errors.WithMessagef(err, "failed to move tensor %q (%s) to backend's device #0", data.name, data.tensor.Shape())
				}
			}

			start = time.Now()
			select {
			case <-done:
				return
			case chOut <- data:
				waitTime += time.Since(start)
				if data.err != nil {
					return
				}
			}
		}
	}
}
