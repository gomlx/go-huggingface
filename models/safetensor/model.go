package safetensor

import (
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

// Model represents a model (possibly split across multiple safetensor files).
type Model struct {
	Repo      *hub.Repo
	IndexFile string
	Index     *ShardedModelIndex
	headers   map[string]*Header // filename -> parsed header
}

// ShardedModelIndex represents a model.safetensors.index.json file for sharded models.
type ShardedModelIndex struct {
	Metadata  map[string]interface{} `json:"metadata"`   // Model metadata
	WeightMap map[string]string      `json:"weight_map"` // Tensor name -> filename
}

func New(repo *hub.Repo) (*Model, error) {
	return &Model{
		Repo: repo,
	}, nil
}

// ListTensors returns all tensor names in the model.
func (r *Model) ListTensors() []string {
	names := make([]string, 0, len(r.Index.WeightMap))
	for name := range r.Index.WeightMap {
		names = append(names, name)
	}
	return names
}

// GetTensorLocation returns the filename containing a specific tensor.
func (r *Model) GetTensorLocation(tensorName string) (string, error) {
	filename, ok := r.Index.WeightMap[tensorName]
	if !ok {
		return "", errors.Errorf("tensor %s not found in weight map", tensorName)
	}
	return filename, nil
}

// GetTensorMetadata returns metadata for a specific tensor without loading data.
func (r *Model) GetTensorMetadata(tensorName string) (*TensorMetadata, error) {
	filename, err := r.GetTensorLocation(tensorName)
	if err != nil {
		return nil, err
	}

	st, err := r.GetSafetensor(filename)
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

// TensorWithName holds a tensor name and its GoMLX tensor data.
type TensorWithName struct {
	Name   string
	Tensor *tensors.Tensor
}
