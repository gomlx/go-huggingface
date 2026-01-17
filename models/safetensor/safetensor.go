// Package safetensor provides a Model object for safetensors-based models,
// from which one can load individual weights (tensors) or interate over them, with access to headers.
//
// Example:
//
//	repo := hub.New(modelID).WithAuth(hfAuthToken)
//	model, err := safetensor.New(repo)
//	if err != nil {
//		panic(err)
//	}
//
//	tensor, err := model.GetTensor("model.safetensors", "embeddings.position_embeddings.weight")
//	if err != nil {
//		panic(err)
//	}
//
// Or use a simpler interface to directly iterate over the tensors of the model.
//
//	repo := hub.New(modelID).WithAuth(hfAuthToken)
//	for tensorAndName, err := safetensor.IterTensorsFromRepo(repo) {
//		if err != nil {
//			panic(err)
//		}
//		fmt.Printf("- Tensor %s: shape=%s\n", tensorAndName.Name, tensorAndName.Tensor.Shape())
//	}
package safetensor

import (
	"encoding/json"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/pkg/errors"
	"golang.org/x/exp/mmap"
)

// Load loads the model from the repo, whether it's sharded or a single file.
// It automatically detects sharded models via index files, otherwise treats the first
// .safetensors file as a single-file model.
func (m *Model) Load() error {
	indexFile, isSharded, err := m.DetectShardedModel()
	if err != nil {
		return err
	}

	if isSharded {
		return m.LoadShardedModel(indexFile)
	}
	return m.LoadSingleFileModel()
}

// DetectShardedModel checks if the repository contains a sharded model and returns the index filename.
func (m *Model) DetectShardedModel() (string, bool, error) {
	if m.Repo == nil {
		return "", false, errors.New("Repo is nil, create a ModelSafetensor with NewModelSafetensor first")
	}

	// Look for model.safetensors.index.json or pytorch_model.bin.index.json
	commonIndexFiles := []string{
		"model.safetensors.index.json",
		"pytorch_model.safetensors.index.json",
	}

	for filename, err := range m.Repo.IterFileNames() {
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
func (m *Model) LoadSingleFileModel() error {
	if m.Repo == nil {
		return errors.New("Repocreate a ModelSafetensor with NewModelSafetensor first")
	}

	localPaths := []string{}
	for filename, err := range m.Repo.IterFileNames() {
		if err != nil {
			return err
		}

		if filepath.Ext(filename) == ".safetensors" {
			// Download and parse the file to get tensor names
			localPath, err := m.Repo.DownloadFile(filename)
			if err != nil {
				return errors.Wrapf(err, "failed to download %s", filename)
			}
			localPaths = append(localPaths, localPath)
		}
	}

	if len(localPaths) == 0 {
		return errors.New("no .safetensors files found in repository")
	}

	header, _, err := m.parseHeader(localPaths[0])
	if err != nil {
		return errors.Wrapf(err, "failed to parse header for %s", localPaths[0])
	}

	// Create a synthetic index with all tensors pointing to this one file
	weightMap := make(map[string]string)
	for tensorName := range header.Tensors {
		weightMap[tensorName] = path.Base(localPaths[0])
	}

	m.Index = &ShardedModelIndex{
		WeightMap: weightMap,
	}
	m.IndexFile = localPaths[0]
	m.Headers = map[string]*Header{
		path.Base(localPaths[0]): header,
	}

	return nil
}

// LoadShardedModel loads a sharded model index file (typically model.safetensors.index.json).
func (m *Model) LoadShardedModel(indexFilename string) error {
	if m.Repo == nil {
		return errors.New("Repo is nil, create a ModelSafetensor with NewModelSafetensor first")
	}

	localPath, err := m.Repo.DownloadFile(indexFilename)
	if err != nil {
		return errors.Wrapf(err, "failed to download %s", indexFilename)
	}

	data, err := os.ReadFile(localPath)
	if err != nil {
		return errors.Wrapf(err, "failed to read %s", localPath)
	}

	var index ShardedModelIndex
	if err := json.Unmarshal(data, &index); err != nil {
		return errors.Wrap(err, "failed to parse sharded model index")
	}

	m.IndexFile = indexFilename
	m.Index = &index
	m.Headers = make(map[string]*Header)

	return nil
}

// GetSafetensor returns the parsed .safetensors file header for a specific tensor.
//
// It returns a FileInfo object for the .safetensor file, with its file name and header.
// The header holds metadata to all tensors contained in the file.
func (m *Model) GetSafetensor(filename string) (*FileInfo, error) {
	if m.Repo == nil {
		return nil, errors.New("Repo is nil, create a ModelSafetensor with NewModelSafetensor first")
	}

	if !strings.HasSuffix(filename, ".safetensors") {
		return nil, errors.Errorf("filename %s is not a .safetensors file", filename)
	}

	localPath, err := m.Repo.DownloadFile(filename)
	if err != nil {
		return nil, err
	}

	header, _, err := m.parseHeader(localPath)
	if err != nil {
		return nil, err
	}

	return &FileInfo{Filename: filename, Header: header}, nil
}

// IterSafetensors returns an iterator over all .safetensors files in the repository.
//
// It yields FileInfo objects for each .safetensors file, with its file name and header.
// The header holds metadata to all tensors contained in the file.
func (m *Model) IterSafetensors() func(yield func(FileInfo, error) bool) {
	return func(yield func(FileInfo, error) bool) {
		for filename, err := range m.Repo.IterFileNames() {
			if err != nil {
				yield(FileInfo{}, err)
				return
			}

			// Only process .safetensors files
			if !strings.HasSuffix(filename, ".safetensors") {
				continue
			}

			// Download and parse header
			localPath, err := m.Repo.DownloadFile(filename)
			if err != nil {
				yield(FileInfo{}, errors.Wrapf(err, "failed to download %s", filename))
				return
			}

			header, _, err := m.parseHeader(localPath)
			if err != nil {
				yield(FileInfo{}, errors.Wrapf(err, "failed to parse header for %s", filename))
				return
			}

			if !yield(FileInfo{Filename: filename, Header: header}, nil) {
				return
			}
		}
	}
}

// GetTensor by its name.
func (m *Model) GetTensor(tensorName string) (*TensorWithName, error) {
	filename, err := m.GetTensorFilename(tensorName)
	if err != nil {
		return nil, err
	}
	return m.GetTensorFromFile(filename, tensorName)
}

// GetTensorFromFile loads a tensor from within a .safetensors file and converts it to a GoMLX tensor.
//
// This requires a loaded model -- see Model.Load().
func (m *Model) GetTensorFromFile(filename, tensorName string) (*TensorWithName, error) {
	if m.Repo == nil {
		return nil, errors.New("Repo is nil, create a ModelSafetensor with NewModelSafetensor first")
	}
	if m.Index == nil || len(m.Index.WeightMap) == 0 {
		return nil, errors.New("model not loaded, call LoadModel first")
	}

	localPath, err := m.Repo.DownloadFile(filename)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to download %s", filename)
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
	defer reader.Close()

	// Create MMapReader
	mmapReader := &MMapReader{
		reader:     reader,
		dataOffset: dataOffset,
		header:     header,
	}

	// Read tensor
	tensor, err := mmapReader.ReadTensor(tensorName)
	if err != nil {
		return nil, err
	}

	return &TensorWithName{Name: tensorName, Tensor: tensor}, nil
}

// IterTensors returns an iterator over all tensors as GoMLX tensors.
// It uses mmap efficiently: opens each shard file once and reads all tensors from it sequentially.
// This is optimal for startup when loading many tensors.
func (m *Model) IterTensors() func(yield func(TensorWithName, error) bool) {
	return func(yield func(TensorWithName, error) bool) {
		if m.Repo == nil {
			yield(TensorWithName{}, errors.New("Repo is nil, create a ModelSafetensor with NewModelSafetensor first"))
			return
		}
		if m.Index == nil || len(m.Index.WeightMap) == 0 {
			yield(TensorWithName{}, errors.New("model not loaded, call LoadModel first"))
			return
		}

		// Group tensors by shard file for efficient reading
		shardToTensors := make(map[string][]string)
		for tensorName, filename := range m.Index.WeightMap {
			shardToTensors[filename] = append(shardToTensors[filename], tensorName)
		}

		// Process each shard file with one mmap
		for filename, tensorNames := range shardToTensors {
			// Download shard file
			localPath, err := m.Repo.DownloadFile(filename)
			if err != nil {
				yield(TensorWithName{}, errors.Wrapf(err, "failed to download %s", filename))
				return
			}

			// Parse header once
			header, dataOffset, err := m.parseHeader(localPath)
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

			// Create MMapReader
			mmapReader := &MMapReader{
				reader:     reader,
				dataOffset: dataOffset,
				header:     header,
			}

			// Sort tensors by file offset for sequential reading
			sortedTensors := sortTensorsByOffset(tensorNames, header)

			// Read all tensors from this shard
			for _, tensorName := range sortedTensors {
				tensor, err := mmapReader.ReadTensor(tensorName)
				if err != nil {
					reader.Close()
					yield(TensorWithName{}, err)
					return
				}

				if !yield(TensorWithName{Name: tensorName, Tensor: tensor}, nil) {
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
func sortTensorsByOffset(tensorNames []string, header *Header) []string {
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
