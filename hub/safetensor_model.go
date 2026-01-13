package hub

import (
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
	"golang.org/x/exp/mmap"
)

// Model represents a model (possibly split across multiple safetensor files).
type Model struct {
	repo      *Repo
	IndexFile string
	Index     *ShardedModelIndex
	headers   map[string]*SafetensorHeader // filename -> parsed header
}

// GetTensorLocation returns the filename containing a specific tensor.
func (sm *Model) GetTensorLocation(tensorName string) (string, error) {
	filename, ok := sm.Index.WeightMap[tensorName]
	if !ok {
		return "", errors.Errorf("tensor %s not found in weight map", tensorName)
	}
	return filename, nil
}

// ListTensors returns all tensor names in the sharded model.
func (sm *Model) ListTensors() []string {
	names := make([]string, 0, len(sm.Index.WeightMap))
	for name := range sm.Index.WeightMap {
		names = append(names, name)
	}
	return names
}

// LoadTensor loads a specific tensor from the appropriate shard file as a GoMLX tensor.
func (sm *Model) LoadTensor(tensorName string) (*tensors.Tensor, error) {
	filename, err := sm.GetTensorLocation(tensorName)
	if err != nil {
		return nil, err
	}

	return sm.repo.LoadSafetensor(filename, tensorName)
}

// LoadTensorStreaming loads a specific tensor using memory-mapped streaming as a GoMLX tensor.
func (sm *Model) LoadTensorStreaming(tensorName string) (*tensors.Tensor, error) {
	filename, err := sm.GetTensorLocation(tensorName)
	if err != nil {
		return nil, err
	}

	return sm.repo.LoadSafetensorStreaming(filename, tensorName)
}

// GetTensorMetadata returns metadata for a specific tensor without loading data.
func (sm *Model) GetTensorMetadata(tensorName string) (*TensorMetadata, error) {
	filename, err := sm.GetTensorLocation(tensorName)
	if err != nil {
		return nil, err
	}

	header, err := sm.repo.GetSafetensorHeader(filename)
	if err != nil {
		return nil, err
	}

	meta, ok := header.Tensors[tensorName]
	if !ok {
		return nil, errors.Errorf("tensor %s not found in %s", tensorName, filename)
	}

	return meta, nil
}

// SafetensorHeader represents the JSON header of a safetensors file.
type SafetensorHeader struct {
	Tensors  map[string]*TensorMetadata // Tensor name -> metadata
	Metadata map[string]interface{}     // Optional __metadata__ field
}

// ShardedModelIndex represents a model.safetensors.index.json file for sharded models.
type ShardedModelIndex struct {
	Metadata  map[string]interface{} `json:"metadata"`   // Model metadata
	WeightMap map[string]string      `json:"weight_map"` // Tensor name -> filename
}

// TensorMetadata represents metadata for a single tensor in a safetensors file.
type TensorMetadata struct {
	Name        string   `json:"-"`            // Tensor name (from map key)
	Dtype       string   `json:"dtype"`        // Data type: F32, F64, I32, I64, etc.
	Shape       []int    `json:"shape"`        // Tensor dimensions
	DataOffsets [2]int64 `json:"data_offsets"` // [start, end] byte offsets in file
}

// SizeBytes returns the size of the tensor data in bytes.
func (tm *TensorMetadata) SizeBytes() int64 {
	return tm.DataOffsets[1] - tm.DataOffsets[0]
}

// NumElements returns the total number of elements in a tensor based on its shape.
func (tm *TensorMetadata) NumElements() int64 {
	if len(tm.Shape) == 0 {
		return 1 // Scalar
	}
	prod := int64(1)
	for _, dim := range tm.Shape {
		prod *= int64(dim)
	}
	return prod
}

// SafetensorReader provides streaming access to tensor data via io.ReaderAt.
type SafetensorReader struct {
	reader     *mmap.ReaderAt
	dataOffset int64
	meta       *TensorMetadata
}

// ReadAt implements io.ReaderAt for the tensor data.
func (sr *SafetensorReader) ReadAt(p []byte, off int64) (n int, err error) {
	tensorOffset := sr.dataOffset + sr.meta.DataOffsets[0] + off
	return sr.reader.ReadAt(p, tensorOffset)
}

// Len returns the size of the tensor data in bytes.
func (sr *SafetensorReader) Len() int {
	return int(sr.meta.SizeBytes())
}

// Close closes the underlying memory-mapped file.
func (sr *SafetensorReader) Close() error {
	return sr.reader.Close()
}

// Metadata returns the tensor metadata.
func (sr *SafetensorReader) Metadata() *TensorMetadata {
	return sr.meta
}
