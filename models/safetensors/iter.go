package safetensors

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/edsrzf/mmap-go"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

type fileRef struct {
	f        *os.File
	mmap     mmap.MMap
	refCount atomic.Int32
	mu       sync.Mutex
}

func (fr *fileRef) Dec() {
	if fr.refCount.Add(-1) == 0 {
		fr.Close()
	}
}

func (fr *fileRef) Close() {
	fr.mu.Lock()
	defer fr.mu.Unlock()
	if fr.mmap != nil {
		fr.mmap.Unmap()
		fr.mmap = nil
	}
	if fr.f != nil {
		fr.f.Close()
		fr.f = nil
	}
}

type iterTensorData struct {
	name       string
	tensor     *tensors.Tensor
	shape      shapes.Shape
	readBuffer []byte
	err        error
	fileRef    *fileRef
}

// IterTensorsFromRepo iterates over all tensors in the repository.
//
// Tensors are loaded into the backend directly (e.g.: GPU, or a shared memory tensor on CPU, etc).
// If the backend is nil, it instead loads them in host memory.
//
// This is a more performant version of `Model.IterTensors` that avoids the overhead of creating a `Model` struct and
// parallelizes allocations and copying around.
func IterTensorsFromRepo(backend compute.Backend, repo *hub.Repo) func(yield func(TensorAndName, error) bool) {
	return func(yield func(TensorAndName, error) bool) {
		done := make(chan struct{})
		var wg sync.WaitGroup

		var openFiles []*fileRef
		var filesMu sync.Mutex

		defer func() {
			wg.Wait()
			filesMu.Lock()
			defer filesMu.Unlock()
			for _, fr := range openFiles {
				fr.Close()
			}
		}()
		defer close(done)

		chDevice := make(chan iterTensorData, 10)
		wg.Go(func() { iterFromRepoDownload(backend, repo, done, chDevice, &openFiles, &filesMu) })

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

func iterFromRepoDownload(backend compute.Backend, repo *hub.Repo, done <-chan struct{}, chDevice chan<- iterTensorData, openFiles *[]*fileRef, filesMu *sync.Mutex) {
	start := time.Now()
	var waitTime time.Duration
	if klog.V(1).Enabled() {
		defer func() {
			klog.Infof(" - Repo download / Mmap open time: %s (of which %s is waiting)", time.Since(start), waitTime)
		}()
	}
	defer close(chDevice)

	reportErrFn := func(err error) {
		waitStart := time.Now()
		select {
		case <-done:
		case chDevice <- iterTensorData{err: err}:
			waitTime += time.Since(waitStart)
		}
	}

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

		openFile, err := os.Open(localPath)
		if err != nil {
			reportErrFn(errors.Wrapf(err, "failed to open %s", localPath))
			return
		}

		mmapBuf, err := mmap.Map(openFile, mmap.RDONLY, 0)
		if err != nil {
			openFile.Close()
			reportErrFn(errors.Wrapf(err, "failed to mmap %s", localPath))
			return
		}

		fr := &fileRef{f: openFile, mmap: mmapBuf}
		fr.refCount.Store(int32(len(tensorNames)))

		filesMu.Lock()
		*openFiles = append(*openFiles, fr)
		filesMu.Unlock()

		if len(tensorNames) == 0 {
			fr.Close()
			continue
		}

		for _, name := range tensorNames {
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

			readBuffer := fr.mmap[dataOffset+meta.DataOffsets[0] : dataOffset+meta.DataOffsets[1]]

			waitStart = time.Now()
			select {
			case <-done:
				waitTime += time.Since(waitStart)
				return
			case chDevice <- iterTensorData{
				name:       name,
				tensor:     nil,
				shape:      shape,
				readBuffer: readBuffer,
				fileRef:    fr,
			}:
				waitTime += time.Since(waitStart)
			}
		}
	}
}

var MaxParallelBufferTransfers = 4

func iterFromRepoToDevice(backend compute.Backend, done <-chan struct{}, chDevice <-chan iterTensorData, chOut chan<- iterTensorData) {
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
						if data.fileRef != nil {
							data.fileRef.Dec()
						}
						return
					}

					data.tensor, data.err = tensors.FromRaw(backend, 0, data.shape, data.readBuffer)
					if data.err != nil {
						data.err = errors.WithMessagef(data.err,
							"failed to create tensor %q (%s) from bytes",
							data.name, data.shape)
					}

					if data.fileRef != nil {
						data.fileRef.Dec()
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
