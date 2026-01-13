package hub

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestGetSafetensorMetadata tests getting tensor metadata without loading data.
func TestGetSafetensorMetadata(t *testing.T) {
	repo := New("sentence-transformers/all-MiniLM-L6-v2")
	meta, err := repo.GetSafetensorHeader("model.safetensors")
	require.NoError(t, err)
	assert.NotNil(t, meta)

	require.Contains(t, meta.Tensors, "embeddings.position_embeddings.weight")
	assert.Equal(t, "F32", meta.Tensors["embeddings.position_embeddings.weight"].Dtype)
	assert.Equal(t, "embeddings.position_embeddings.weight", meta.Tensors["embeddings.position_embeddings.weight"].Name)
	assert.Greater(t, len(meta.Tensors["embeddings.position_embeddings.weight"].Shape), 0)
}

// TestDetectShardedModel tests detecting sharded models.
func TestDetectShardedModel(t *testing.T) {
	// Test non-sharded model
	repo := New("sentence-transformers/all-MiniLM-L6-v2")
	indexFile, isSharded, err := repo.DetectShardedModel()
	require.NoError(t, err)
	assert.False(t, isSharded)
	assert.Empty(t, indexFile)
}

// TestLoadSafetensor tests GoMLX tensor loading.
func TestLoadSafetensor(t *testing.T) {
	repo := New("sentence-transformers/all-MiniLM-L6-v2")
	tensor, err := repo.LoadSafetensor("model.safetensors", "embeddings.position_embeddings.weight")
	require.NoError(t, err)
	assert.NotNil(t, tensor)
	assert.Greater(t, tensor.Shape().Size(), 0)
}

// TestLoadModel tests loading a model as a unified ShardedModel interface.
func TestLoadModel(t *testing.T) {
	// Test with single-file model
	repo := New("sentence-transformers/all-MiniLM-L6-v2")
	model, err := repo.LoadModel()
	require.NoError(t, err)
	require.NotNil(t, model)
	assert.Greater(t, len(model.Index.WeightMap), 0, "should have tensors in weight map")

	// Verify we can load tensors through the model
	tensorNames := model.ListTensors()
	assert.Greater(t, len(tensorNames), 0)

	tensor, err := model.LoadTensor(tensorNames[0])
	require.NoError(t, err)
	assert.NotNil(t, tensor)
}

// TestLoadSafetensorStreaming tests memory-mapped streaming access.
func TestLoadSafetensorStreaming(t *testing.T) {
	repo := New("sentence-transformers/all-MiniLM-L6-v2")
	tensor, err := repo.LoadSafetensorStreaming("model.safetensors", "embeddings.position_embeddings.weight")
	require.NoError(t, err)
	assert.NotNil(t, tensor)
	assert.Greater(t, tensor.Shape().Size(), 0)
}

func TestIterSafetensors(t *testing.T) {
	repo := New("sentence-transformers/all-MiniLM-L6-v2")
	for safetensor, err := range repo.IterSafetensors() {
		require.NoError(t, err)
		assert.NotEmpty(t, safetensor.Filename)
		assert.NotNil(t, safetensor.Header)
		assert.NotNil(t, safetensor.Header.Metadata)
		assert.Greater(t, len(safetensor.Header.Tensors), 0)
		assert.Greater(t, len(safetensor.Header.Metadata), 0)
	}
}

// TestIterAllTensors tests iterating over all tensors.
func TestIterAllTensors(t *testing.T) {
	repo := New("sentence-transformers/all-MiniLM-L6-v2")
	for tensorWithName, err := range repo.IterAllTensors() {
		require.NoError(t, err)
		assert.NotEmpty(t, tensorWithName.Name)
		assert.NotNil(t, tensorWithName.Tensor)
		assert.Greater(t, tensorWithName.Tensor.Shape().Size(), 0)
	}
}

// TestDtypeSize tests the dtype size helper function.
func TestDtypeSize(t *testing.T) {
	tests := []struct {
		dtype    string
		expected int
	}{
		{"F64", 8},
		{"F32", 4},
		{"F16", 2},
		{"BF16", 2},
		{"I64", 8},
		{"I32", 4},
		{"I16", 2},
		{"I8", 1},
		{"U64", 8},
		{"U32", 4},
		{"U16", 2},
		{"U8", 1},
		{"BOOL", 1},
	}

	for _, tt := range tests {
		t.Run(tt.dtype, func(t *testing.T) {
			size, err := dtypeSize(tt.dtype)
			require.NoError(t, err)
			assert.Equal(t, tt.expected, size)
		})
	}

	// Test unknown dtype
	_, err := dtypeSize("UNKNOWN")
	assert.Error(t, err)
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

// TestTensorMetadataSizeBytes tests calculating tensor size in bytes.
func TestTensorMetadataSizeBytes(t *testing.T) {
	meta := &TensorMetadata{
		DataOffsets: [2]int64{100, 500},
	}
	assert.Equal(t, int64(400), meta.SizeBytes())
}

// ExampleRepo_LoadShardedModel shows how to work with sharded models.
func ExampleRepo_LoadShardedModel() {
	repo := New("google/gemma-2-2b-it").WithAuth(os.Getenv("HF_TOKEN"))

	sharded, err := repo.LoadShardedModel("model.safetensors.index.json")
	if err != nil {
		// Handle error
		return
	}

	// List all tensors
	tensorNames := sharded.ListTensors()

	// Load a specific tensor as GoMLX tensor
	tensor, err := sharded.LoadTensor(tensorNames[0])
	if err != nil {
		// Handle error
		return
	}

	_ = tensor
}

// ExampleRepo_IterAllTensors shows how to iterate over all tensors.
func ExampleRepo_IterAllTensors() {
	repo := New("sentence-transformers/all-MiniLM-L6-v2")

	for tensorWithName, err := range repo.IterAllTensors() {
		if err != nil {
			// Handle error
			break
		}

		// Process each tensor
		_ = tensorWithName.Name
		_ = tensorWithName.Tensor.DType()
		_ = tensorWithName.Tensor.Shape()
	}
}

// ExampleRepo_LoadSafetensor shows GoMLX integration.
func ExampleRepo_LoadSafetensor() {
	repo := New("sentence-transformers/all-MiniLM-L6-v2")

	tensor, err := repo.LoadSafetensor("model.safetensors", "embeddings.word_embeddings.weight")
	if err != nil {
		// Handle error
		return
	}

	// Use tensor with GoMLX
	// graph.ConstTensor(g, tensor)
	_ = tensor
}
