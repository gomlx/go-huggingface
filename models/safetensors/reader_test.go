package safetensors

import (
	"fmt"
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/exp/mmap"
)

// TestMMapReaderReadTensor tests reading a tensor using MMapReader.
func TestMMapReaderReadTensor(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m := NewEmpty(repo)

	// Download file
	localPath, err := repo.DownloadFile("model.safetensors")
	require.NoError(t, err)

	// Parse header
	header, dataOffset, err := m.parseHeader(localPath)
	require.NoError(t, err)

	// Open mmap
	reader, err := mmap.Open(localPath)
	require.NoError(t, err)
	defer reader.Close()

	// Create MMapReader
	mmapReader := &MMapReader{
		reader:     reader,
		dataOffset: dataOffset,
		Header:     header,
	}

	// Read a known tensor
	tensorName := "embeddings.position_embeddings.weight"
	tensor, err := mmapReader.ReadTensor(tensorName)
	require.NoError(t, err)
	assert.NotNil(t, tensor)
	assert.Greater(t, tensor.Shape().Size(), 0)
}

// TestMMapReaderReadTensorNotFound tests reading a non-existent tensor.
func TestMMapReaderReadTensorNotFound(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := New(repo)
	require.NoError(t, err)

	localPath, err := repo.DownloadFile("model.safetensors")
	require.NoError(t, err)

	header, dataOffset, err := m.parseHeader(localPath)
	require.NoError(t, err)

	reader, err := mmap.Open(localPath)
	require.NoError(t, err)
	defer reader.Close()

	mmapReader := &MMapReader{
		reader:     reader,
		dataOffset: dataOffset,
		Header:     header,
	}

	// Try to read non-existent tensor
	_, err = mmapReader.ReadTensor("non_existent_tensor")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

// TestMMapReaderMetadata tests the Metadata method.
func TestMMapReaderMetadata(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := New(repo)
	require.NoError(t, err)
	reader, err := m.NewMMapReader("model.safetensors")
	require.NoError(t, err)
	defer reader.Close()

	// Get a tensor metadata
	tensorName := "embeddings.position_embeddings.weight"
	_, ok := reader.Header.Tensors[tensorName]
	require.True(t, ok)
}

func TestMMapReaderTensor(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := New(repo)
	require.NoError(t, err)
	reader, err := m.NewMMapReader("model.safetensors")
	require.NoError(t, err)

	// Read a known tensor
	tensorName := "embeddings.position_embeddings.weight"
	tensor, err := reader.ReadTensor(tensorName)
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
