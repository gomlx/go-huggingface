package safetensors

import (
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

// Model represents a model (possibly split across multiple safetensor files).
// It contains a map of filename to a Header object, parsed from the safetensor file.
type Model struct {
	Repo      *hub.Repo
	IndexFile string
	Index     *ShardedModelIndex
	Headers   map[string]*Header // ".safetensor" filename -> parsed header
}

// ShardedModelIndex represents a model.safetensors.index.json file for sharded models.
type ShardedModelIndex struct {
	Metadata  map[string]any    `json:"metadata"`   // Model metadata
	WeightMap map[string]string `json:"weight_map"` // Tensor name -> filename
}

// New creates a new Model and loads the loads the headers from the repo safetensors file(s).
// If err is nil, it's ready to be used.
func New(repo *hub.Repo) (*Model, error) {
	m := NewEmpty(repo)
	err := m.Load()
	if err != nil {
		return nil, err
	}
	return m, nil
}

// NewEmpty creates an empty Model object, no headers are loaded.
//
// Call Model.Load() to load the model header(s).
func NewEmpty(repo *hub.Repo) *Model {
	return &Model{
		Repo: repo,
	}
}

// ListTensorNames returns all tensor names in the model.
func (m *Model) ListTensorNames() []string {
	names := make([]string, 0, len(m.Index.WeightMap))
	for name := range m.Index.WeightMap {
		names = append(names, name)
	}
	return names
}

// GetTensorFilename returns the filename containing a specific tensor.
func (m *Model) GetTensorFilename(tensorName string) (string, error) {
	filename, ok := m.Index.WeightMap[tensorName]
	if !ok {
		return "", errors.Errorf("tensor %s not found in weight map", tensorName)
	}
	return filename, nil
}

// GetTensorMetadata returns metadata for a specific tensor without loading data.
func (m *Model) GetTensorMetadata(tensorName string) (*TensorMetadata, error) {
	filename, err := m.GetTensorFilename(tensorName)
	if err != nil {
		return nil, err
	}

	st, err := m.GetSafetensor(filename)
	if err != nil {
		return nil, err
	}

	meta, ok := st.Header.Tensors[tensorName]
	if !ok {
		return nil, errors.Errorf("tensor %s not found in %s", tensorName, filename)
	}

	return meta, nil
}

// FileInfo holds information about a safetensor file.
type FileInfo struct {
	Filename string
	Header   *Header
}

// TensorMetadata represents metadata for a single tensor in a safetensors file.
type TensorMetadata struct {
	Name        string   `json:"-"`            // Tensor name (from map key)
	Dtype       string   `json:"dtype"`        // Data type: F32, F64, I32, I64, etc.
	Shape       []int    `json:"shape"`        // Tensor dimensions
	DataOffsets [2]int64 `json:"data_offsets"` // [start, end] byte offsets in file
}

// TensorAndName holds a tensor name and its GoMLX tensor data.
type TensorAndName struct {
	Name   string
	Tensor *tensors.Tensor
}
