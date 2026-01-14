package safetensor

import (
	"encoding/json"
	"io"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
	"golang.org/x/exp/mmap"
)

// LoadModel loads a model as a Model, whether it's sharded or a single file.
// This provides a unified interface for loading any safetensors model.
// It automatically detects sharded models via index files, otherwise treats the first
// .safetensors file as a single-file model.
func (r *ModelSafetensor) LoadModel() (*ModelSafetensor, error) {
	indexFile, isSharded, err := r.DetectShardedModel()
	if err != nil {
		return nil, err
	}

	if isSharded {
		return r.LoadShardedModel(indexFile)
	}
	return r.LoadSingleFileModel()
}

// DetectShardedModel checks if the repository contains a sharded model and returns the index filename.
func (r *ModelSafetensor) DetectShardedModel() (string, bool, error) {
	if r.Repo == nil {
		return "", false, errors.New("Repo is nil, create a ModelSafetensor with NewModelSafetensor first")
	}

	// Look for model.safetensors.index.json or pytorch_model.bin.index.json
	commonIndexFiles := []string{
		"model.safetensors.index.json",
		"pytorch_model.safetensors.index.json",
	}

	for filename, err := range r.Repo.IterFileNames() {
		if err != nil {
			return "", false, err
		}

		for _, indexName := range commonIndexFiles {
			if filename == indexName || filepath.Base(filename) == indexName {
				return filename, true, nil
			}
		}
	}

	return "", false, nil
}

// LoadSingleFileModel loads a single-file safetensors model.
func (r *ModelSafetensor) LoadSingleFileModel() (*ModelSafetensor, error) {
	if r.Repo == nil {
		return nil, errors.New("Repo is nil, create a ModelSafetensor with NewModelSafetensor first")
	}

	localPaths := []string{}
	for filename, err := range r.Repo.IterFileNames() {
		if err != nil {
			return nil, err
		}

		if filepath.Ext(filename) == ".safetensors" {
			// Download and parse the file to get tensor names
			localPath, err := r.Repo.DownloadFile(filename)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to download %s", filename)
			}
			localPaths = append(localPaths, localPath)
		}
	}

	if len(localPaths) == 0 {
		return nil, errors.New("no .safetensors files found in repository")
	}

	header, _, err := r.ParseSafetensorHeader(localPaths[0])
	if err != nil {
		return nil, errors.Wrapf(err, "failed to parse header for %s", localPaths[0])
	}

	// Create a synthetic index with all tensors pointing to this one file
	weightMap := make(map[string]string)
	for tensorName := range header.Tensors {
		weightMap[tensorName] = path.Base(localPaths[0])
	}

	r.headers = make(map[string]*SafetensorHeader)
	r.Index = &ShardedModelIndex{
		WeightMap: weightMap,
	}
	r.IndexFile = localPaths[0]
	r.headers = map[string]*SafetensorHeader{
		path.Base(localPaths[0]): header,
	}

	return r, nil
}

// LoadShardedModel loads a sharded model index file (typically model.safetensors.index.json).
func (r *ModelSafetensor) LoadShardedModel(indexFilename string) (*ModelSafetensor, error) {
	if r.Repo == nil {
		return nil, errors.New("Repo is nil, create a ModelSafetensor with NewModelSafetensor first")
	}

	localPath, err := r.Repo.DownloadFile(indexFilename)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to download %s", indexFilename)
	}

	data, err := os.ReadFile(localPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read %s", localPath)
	}

	var index ShardedModelIndex
	if err := json.Unmarshal(data, &index); err != nil {
		return nil, errors.Wrap(err, "failed to parse sharded model index")
	}

	return &ModelSafetensor{
		Repo:      r.Repo,
		IndexFile: indexFilename,
		Index:     &index,
		headers:   make(map[string]*SafetensorHeader),
	}, nil
}

// GetSafetensor returns the parsed safetensor header for a specific tensor.
func (r *ModelSafetensor) GetSafetensor(filename string) (*SafetensorFileInfo, error) {
	if r.Repo == nil {
		return nil, errors.New("Repo is nil, create a ModelSafetensor with NewModelSafetensor first")
	}

	if !strings.HasSuffix(filename, ".safetensors") {
		return nil, errors.Errorf("filename %s is not a .safetensors file", filename)
	}

	localPath, err := r.Repo.DownloadFile(filename)
	if err != nil {
		return nil, err
	}

	header, _, err := r.ParseSafetensorHeader(localPath)
	if err != nil {
		return nil, err
	}

	return &SafetensorFileInfo{Filename: filename, Header: header}, nil
}

// IterSafetensors returns an iterator over all .safetensors files in the repository.
func (r *ModelSafetensor) IterSafetensors() func(yield func(SafetensorFileInfo, error) bool) {
	return func(yield func(SafetensorFileInfo, error) bool) {
		for filename, err := range r.Repo.IterFileNames() {
			if err != nil {
				yield(SafetensorFileInfo{}, err)
				return
			}

			// Only process .safetensors files
			if !strings.HasSuffix(filename, ".safetensors") {
				continue
			}

			// Download and parse header
			localPath, err := r.Repo.DownloadFile(filename)
			if err != nil {
				yield(SafetensorFileInfo{}, errors.Wrapf(err, "failed to download %s", filename))
				return
			}

			header, _, err := r.ParseSafetensorHeader(localPath)
			if err != nil {
				yield(SafetensorFileInfo{}, errors.Wrapf(err, "failed to parse header for %s", filename))
				return
			}

			if !yield(SafetensorFileInfo{Filename: filename, Header: header}, nil) {
				return
			}
		}
	}
}

// GetTensor loads a tensor from a safetensors file and converts it to a GoMLX tensor.
// The returned tensor can be used with graph.ConstTensor().
func (r *ModelSafetensor) GetTensor(filename, tensorName string) (*TensorWithName, error) {
	if r.Repo == nil {
		return nil, errors.New("Repo is nil, create a ModelSafetensor with NewModelSafetensor first")
	}
	if r.Index == nil || len(r.Index.WeightMap) == 0 {
		return nil, errors.New("model not loaded, call LoadModel first")
	}

	localPath, err := r.Repo.DownloadFile(filename)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to download %s", filename)
	}

	header, dataOffset, err := r.ParseSafetensorHeader(localPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to parse header for %s", localPath)
	}

	meta, ok := header.Tensors[tensorName]
	if !ok {
		return nil, errors.Errorf("tensor %s not found in %s", tensorName, localPath)
	}

	// Convert dtype
	dtype, err := safetensorDtypeToGoMLX(meta.Dtype)
	if err != nil {
		return nil, err
	}

	// Convert shape to ints
	dims := make([]int, len(meta.Shape))
	copy(dims, meta.Shape)

	// Create shape and tensor
	shape := shapes.Make(dtype, dims...)
	t := tensors.FromShape(shape)

	// Open file and read directly into tensor memory
	f, err := os.Open(localPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open %s", localPath)
	}
	defer f.Close()

	// Seek to tensor data position
	offset := dataOffset + meta.DataOffsets[0]
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return nil, errors.Wrap(err, "failed to seek to tensor data")
	}

	var readErr error
	t.MutableBytes(func(data []byte) {
		if int64(len(data)) != meta.SizeBytes() {
			readErr = errors.Errorf("tensor shape %s expected %d bytes, but safetensor has %d bytes", shape, len(data), meta.SizeBytes())
			return
		}
		_, readErr = io.ReadFull(f, data)
		if readErr != nil {
			readErr = errors.Wrapf(readErr, "failed to read from %s", localPath)
		}
	})
	if readErr != nil {
		return nil, readErr
	}

	return &TensorWithName{Name: tensorName, Tensor: t}, nil
}

// IterTensors returns an iterator over all tensors as GoMLX tensors.
// It uses mmap efficiently: opens each shard file once and reads all tensors from it sequentially.
// This is optimal for startup when loading many tensors.
func (r *ModelSafetensor) IterTensors() func(yield func(TensorWithName, error) bool) {
	return func(yield func(TensorWithName, error) bool) {
		if r.Repo == nil {
			yield(TensorWithName{}, errors.New("Repo is nil, create a ModelSafetensor with NewModelSafetensor first"))
			return
		}
		if r.Index == nil || len(r.Index.WeightMap) == 0 {
			yield(TensorWithName{}, errors.New("model not loaded, call LoadModel first"))
			return
		}

		// Group tensors by shard file for efficient reading
		shardToTensors := make(map[string][]string)
		for tensorName, filename := range r.Index.WeightMap {
			shardToTensors[filename] = append(shardToTensors[filename], tensorName)
		}

		// Process each shard file with one mmap
		for filename, tensorNames := range shardToTensors {
			// Download shard file
			localPath, err := r.Repo.DownloadFile(filename)
			if err != nil {
				yield(TensorWithName{}, errors.Wrapf(err, "failed to download %s", filename))
				return
			}

			// Parse header once
			header, dataOffset, err := r.ParseSafetensorHeader(localPath)
			if err != nil {
				yield(TensorWithName{}, errors.Wrapf(err, "failed to parse header for %s", filename))
				return
			}

			// Open mmap once for this shard
			reader, err := mmap.Open(localPath)
			if err != nil {
				yield(TensorWithName{}, errors.Wrapf(err, "failed to mmap %s", localPath))
				return
			}

			// Sort tensors by file offset for sequential reading
			sortedTensors := sortTensorsByOffset(tensorNames, header)

			// Read all tensors from this shard
			for _, tensorName := range sortedTensors {
				meta, ok := header.Tensors[tensorName]
				if !ok {
					reader.Close()
					yield(TensorWithName{}, errors.Errorf("tensor %s not found in %s", tensorName, filename))
					return
				}

				// Convert dtype
				dtype, err := safetensorDtypeToGoMLX(meta.Dtype)
				if err != nil {
					reader.Close()
					yield(TensorWithName{}, err)
					return
				}

				// Convert shape to ints
				dims := make([]int, len(meta.Shape))
				copy(dims, meta.Shape)

				// Create shape and tensor
				shape := shapes.Make(dtype, dims...)
				t := tensors.FromShape(shape)

				// Read from mmap directly into tensor memory
				tensorOffset := dataOffset + meta.DataOffsets[0]
				var readErr error
				t.MutableBytes(func(data []byte) {
					if int64(len(data)) != meta.SizeBytes() {
						readErr = errors.Errorf("tensor shape %s expected %d bytes, but safetensor has %d bytes", shape, len(data), meta.SizeBytes())
						return
					}
					_, readErr = reader.ReadAt(data, tensorOffset)
					if readErr != nil && readErr != io.EOF {
						readErr = errors.Wrapf(readErr, "failed to read from %s", filename)
					}
				})
				if readErr != nil {
					reader.Close()
					yield(TensorWithName{}, readErr)
					return
				}

				if !yield(TensorWithName{Name: tensorName, Tensor: t}, nil) {
					reader.Close()
					return
				}
			}

			// Close mmap for this shard
			reader.Close()
		}
	}
}

// sortTensorsByOffset sorts tensor names by their file offset for sequential reading.
func sortTensorsByOffset(tensorNames []string, header *SafetensorHeader) []string {
	type tensorOffset struct {
		name   string
		offset int64
	}

	// Collect offsets
	offsets := make([]tensorOffset, 0, len(tensorNames))
	for _, name := range tensorNames {
		if meta, ok := header.Tensors[name]; ok {
			offsets = append(offsets, tensorOffset{name: name, offset: meta.DataOffsets[0]})
		}
	}

	// Sort by offset using slices.SortFunc
	slices.SortFunc(offsets, func(a, b tensorOffset) int {
		if a.offset < b.offset {
			return -1
		}
		if a.offset > b.offset {
			return 1
		}
		return 0
	})

	// Extract names in sorted order
	result := make([]string, len(offsets))
	for i, to := range offsets {
		result[i] = to.name
	}
	return result
}
