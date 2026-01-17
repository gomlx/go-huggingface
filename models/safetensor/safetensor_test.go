package safetensor

import (
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestLoadModel tests loading a model as a unified Model interface.
func TestLoadModel(t *testing.T) {
	// Test with single-file model
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m := NewEmpty(repo)
	err := m.Load()
	require.NoError(t, err)
	require.NotNil(t, m)
	assert.Greater(t, len(m.Index.WeightMap), 0, "should have tensors in weight map")

	// Verify we can access tensors through the model
	tensorNames := m.ListTensorNames()
	assert.Greater(t, len(tensorNames), 0)
}

// TestDetectShardedModel tests detecting sharded models.
func TestDetectShardedModel(t *testing.T) {
	// Test non-sharded model
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	model := NewEmpty(repo)
	indexFile, isSharded, err := model.DetectShardedModel()
	require.NoError(t, err)
	assert.False(t, isSharded)
	assert.Empty(t, indexFile)
}

// TestLoadSingleFileModel tests loading a single-file safetensors model.
func TestLoadSingleFileModel(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m := NewEmpty(repo)
	err := m.LoadSingleFileModel()
	require.NoError(t, err)
	require.NotNil(t, m)
	assert.Greater(t, len(m.Index.WeightMap), 0)

	// Verify tensors are mapped to the safetensors file
	for _, filename := range m.Index.WeightMap {
		assert.Contains(t, filename, ".safetensors")
	}
}

// TestLoadShardedModel tests loading a sharded model with index file.
func TestLoadShardedModel(t *testing.T) {
	t.Skip("Requires a sharded model")
}

// TestGetSafetensor tests getting safetensor file information.
func TestGetSafetensor(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	model := NewEmpty(repo)
	meta, err := model.GetSafetensor("model.safetensors")
	require.NoError(t, err)
	assert.NotNil(t, meta)

	require.Contains(t, meta.Header.Tensors, "embeddings.position_embeddings.weight")
	assert.Equal(t, "F32", meta.Header.Tensors["embeddings.position_embeddings.weight"].Dtype)
	assert.Equal(t, "embeddings.position_embeddings.weight", meta.Header.Tensors["embeddings.position_embeddings.weight"].Name)
	assert.Greater(t, len(meta.Header.Tensors["embeddings.position_embeddings.weight"].Shape), 0)
}

// TestIterSafetensors tests iterating over all safetensor files.
func TestIterSafetensors(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m := NewEmpty(repo)

	count := 0
	for safetensor, err := range m.IterSafetensors() {
		require.NoError(t, err)
		assert.NotEmpty(t, safetensor.Filename)
		assert.NotNil(t, safetensor.Header)
		assert.NotNil(t, safetensor.Header.Metadata)
		assert.Greater(t, len(safetensor.Header.Tensors), 0)
		count++
	}
	assert.Greater(t, count, 0, "should have at least one safetensor file")
}

// TestGetTensorFromFile tests loading a specific tensor as GoMLX tensor.
func TestGetTensorFromFile(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := New(repo)
	require.NoError(t, err)

	tensor, err := m.GetTensorFromFile("model.safetensors", "embeddings.position_embeddings.weight")
	require.NoError(t, err)
	assert.NotNil(t, tensor)
	assert.NotNil(t, tensor.Tensor)
	assert.Equal(t, "embeddings.position_embeddings.weight", tensor.Name)
	assert.Greater(t, tensor.Tensor.Shape().Size(), 0)
}

// TestGetTensor tests loading a specific tensor as GoMLX tensor.
func TestGetTensor(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := New(repo)
	require.NoError(t, err)

	tensor, err := m.GetTensor("embeddings.position_embeddings.weight")
	require.NoError(t, err)
	assert.NotNil(t, tensor)
	assert.NotNil(t, tensor.Tensor)
	assert.Equal(t, "embeddings.position_embeddings.weight", tensor.Name)
	assert.Greater(t, tensor.Tensor.Shape().Size(), 0)
}

// TestIterTensors tests iterating over all tensors.
func TestIterTensors(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := New(repo)
	require.NoError(t, err)
	count := 0
	for tensorWithName, err := range m.IterTensors() {
		require.NoError(t, err)
		assert.NotEmpty(t, tensorWithName.Name)
		assert.NotNil(t, tensorWithName.Tensor)
		assert.Greater(t, tensorWithName.Tensor.Shape().Size(), 0)
		count++
	}
	assert.Greater(t, count, 0, "should have at least one tensor")
}
