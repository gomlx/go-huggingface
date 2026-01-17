package safetensor

import (
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNew tests creating a new Model instance.
func TestNew(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	model := NewEmpty(repo)
	assert.NotNil(t, model)
	assert.NotNil(t, model.Repo)
}

// TestListTensors tests listing all tensor names in a model.
func TestListTensors(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := New(repo)
	require.NoError(t, err)
	tensorNames := m.ListTensorNames()
	assert.Greater(t, len(tensorNames), 0)

	// Verify all names are non-empty
	for _, name := range tensorNames {
		assert.NotEmpty(t, name)
	}
}

// TestGetTensorFilename tests getting the filename containing a specific tensor.
func TestGetTensorFilename(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := New(repo)
	require.NoError(t, err)
	tensorNames := m.ListTensorNames()
	require.Greater(t, len(tensorNames), 0)

	filename, err := m.GetTensorFilename(tensorNames[0])
	require.NoError(t, err)
	assert.NotEmpty(t, filename)
	assert.Contains(t, filename, ".safetensors")

	// Test non-existent tensor
	_, err = m.GetTensorFilename("non_existent_tensor")
	assert.Error(t, err)
}

// TestGetTensorMetadata tests getting metadata for a specific tensor.
func TestGetTensorMetadata(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := New(repo)
	require.NoError(t, err)
	tensorNames := m.ListTensorNames()
	require.Greater(t, len(tensorNames), 0)

	// Use a known tensor name
	tensorName := "embeddings.position_embeddings.weight"
	meta, err := m.GetTensorMetadata(tensorName)
	require.NoError(t, err)
	assert.NotNil(t, meta)
	assert.NotEmpty(t, meta.Dtype)
	assert.NotNil(t, meta.Shape)
	assert.Greater(t, meta.DataOffsets[1]-meta.DataOffsets[0], int64(0))
}
