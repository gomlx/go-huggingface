package safetensors

import (
	"fmt"
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestTensorReaderReadTensor tests reading a tensor using TensorReader.
func TestTensorReaderReadTensor(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m := NewEmpty(repo)

	tensorReader, err := m.NewTensorReader("model.safetensors")
	require.NoError(t, err)
	defer tensorReader.Close()

	// Read a known tensor
	tensorName := "embeddings.position_embeddings.weight"
	tensor, err := tensorReader.ReadTensor(nil, tensorName)
	require.NoError(t, err)
	assert.NotNil(t, tensor)
	assert.Greater(t, tensor.Shape().Size(), 0)
}

// TestTensorReaderReadTensorNotFound tests reading a non-existent tensor.
func TestTensorReaderReadTensorNotFound(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := New(repo)
	require.NoError(t, err)

	tensorReader, err := m.NewTensorReader("model.safetensors")
	require.NoError(t, err)
	defer tensorReader.Close()

	// Try to read non-existent tensor
	_, err = tensorReader.ReadTensor(nil, "non_existent_tensor")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

// TestTensorReaderMetadata tests the Metadata method.
func TestTensorReaderMetadata(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := New(repo)
	require.NoError(t, err)
	reader, err := m.NewTensorReader("model.safetensors")
	require.NoError(t, err)
	defer reader.Close()

	// Get a tensor metadata
	tensorName := "embeddings.position_embeddings.weight"
	_, ok := reader.Header.Tensors[tensorName]
	require.True(t, ok)
}

func TestTensorReaderTensor(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := New(repo)
	require.NoError(t, err)
	reader, err := m.NewTensorReader("model.safetensors")
	require.NoError(t, err)

	// Read a known tensor
	tensorName := "embeddings.position_embeddings.weight"
	tensor, err := reader.ReadTensor(nil, tensorName)
	require.NoError(t, err)
	assert.NotNil(t, tensor)
	assert.Greater(t, tensor.Shape().Size(), 0)
	fmt.Printf("- Tensor shape: %s\n", tensor.Shape())
	wantShape := shapes.Make(dtypes.Float32, 512, 384)
	if !tensor.Shape().Equal(wantShape) {
		t.Errorf("tensor shape %s read from %q does not match expected shape %s", tensor.Shape(), tensorName, wantShape)
	}

	// Close the reader
	err = reader.Close()
	require.NoError(t, err)
}
