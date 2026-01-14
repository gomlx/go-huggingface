package safetensor

import (
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNewModelSafetensor tests creating a new ModelSafetensor instance.
func TestNewModelSafetensor(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	model, err := NewModelSafetensor(repo)
	require.NoError(t, err)
	assert.NotNil(t, model)
	assert.NotNil(t, model.Repo)
}

// TestListTensors tests listing all tensor names in a model.
func TestListTensors(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := NewModelSafetensor(repo)
	require.NoError(t, err)

	// Load model to populate index
	model, err := m.LoadModel()
	require.NoError(t, err)

	tensorNames := model.ListTensors()
	assert.Greater(t, len(tensorNames), 0)

	// Verify all names are non-empty
	for _, name := range tensorNames {
		assert.NotEmpty(t, name)
	}
}

// TestGetTensorLocation tests getting the filename containing a specific tensor.
func TestGetTensorLocation(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := NewModelSafetensor(repo)
	require.NoError(t, err)

	model, err := m.LoadModel()
	require.NoError(t, err)

	tensorNames := model.ListTensors()
	require.Greater(t, len(tensorNames), 0)

	filename, err := model.GetTensorLocation(tensorNames[0])
	require.NoError(t, err)
	assert.NotEmpty(t, filename)
	assert.Contains(t, filename, ".safetensors")

	// Test non-existent tensor
	_, err = model.GetTensorLocation("non_existent_tensor")
	assert.Error(t, err)
}

// TestGetTensorMetadata tests getting metadata for a specific tensor.
func TestGetTensorMetadata(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := NewModelSafetensor(repo)
	require.NoError(t, err)

	model, err := m.LoadModel()
	require.NoError(t, err)

	tensorNames := model.ListTensors()
	require.Greater(t, len(tensorNames), 0)

	// Use a known tensor name
	tensorName := "embeddings.position_embeddings.weight"
	meta, err := model.GetTensorMetadata(tensorName)
	require.NoError(t, err)
	assert.NotNil(t, meta)
	assert.NotEmpty(t, meta.Dtype)
	assert.NotNil(t, meta.Shape)
	assert.Greater(t, meta.SizeBytes(), int64(0))
}

// TestTensorMetadataSizeBytes tests calculating tensor size in bytes.
func TestTensorMetadataSizeBytes(t *testing.T) {
	meta := &TensorMetadata{
		DataOffsets: [2]int64{100, 500},
	}
	assert.Equal(t, int64(400), meta.SizeBytes())
}

// TestTensorMetadataNumElements tests calculating number of elements.
func TestTensorMetadataNumElements(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		expected int64
	}{
		{"scalar", []int{}, 1},
		{"vector", []int{10}, 10},
		{"matrix", []int{3, 4}, 12},
		{"3d", []int{2, 3, 4}, 24},
		{"4d", []int{2, 3, 4, 5}, 120},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			meta := &TensorMetadata{Shape: tt.shape}
			assert.Equal(t, tt.expected, meta.NumElements())
		})
	}
}
