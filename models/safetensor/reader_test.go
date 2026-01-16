package safetensor

import (
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/exp/mmap"
)

// TestMMapReaderReadTensor tests reading a tensor using MMapReader.
func TestMMapReaderReadTensor(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	m, err := New(repo)
	require.NoError(t, err)

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
		header:     header,
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
		header:     header,
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

	localPath, err := repo.DownloadFile("model.safetensors")
	require.NoError(t, err)

	header, dataOffset, err := m.parseHeader(localPath)
	require.NoError(t, err)

	reader, err := mmap.Open(localPath)
	require.NoError(t, err)
	defer reader.Close()

	// Get a tensor metadata
	tensorName := "embeddings.position_embeddings.weight"
	meta, ok := header.Tensors[tensorName]
	require.True(t, ok)

	mmapReader := &MMapReader{
		reader:     reader,
		dataOffset: dataOffset,
		header:     header,
		meta:       meta,
	}

	// Test Metadata method
	retrievedMeta := mmapReader.Metadata()
	assert.Equal(t, meta, retrievedMeta)
}

// TestMMapReaderLen tests the Len method.
func TestMMapReaderLen(t *testing.T) {
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

	// Get a tensor metadata
	tensorName := "embeddings.position_embeddings.weight"
	meta, ok := header.Tensors[tensorName]
	require.True(t, ok)

	mmapReader := &MMapReader{
		reader:     reader,
		dataOffset: dataOffset,
		header:     header,
		meta:       meta,
	}

	// Test Len method
	length := mmapReader.Len()
	expectedLength := int(meta.DataOffsets[1] - meta.DataOffsets[0])
	assert.Equal(t, expectedLength, length)
	assert.Greater(t, length, 0)
}

// TestMMapReaderReadAt tests the ReadAt method.
func TestMMapReaderReadAt(t *testing.T) {
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

	// Get a tensor metadata
	tensorName := "embeddings.position_embeddings.weight"
	meta, ok := header.Tensors[tensorName]
	require.True(t, ok)

	mmapReader := &MMapReader{
		reader:     reader,
		dataOffset: dataOffset,
		header:     header,
		meta:       meta,
	}

	// Test ReadAt method
	buffer := make([]byte, 16)
	n, err := mmapReader.ReadAt(buffer, 0)
	assert.NoError(t, err)
	assert.Equal(t, 16, n)

	// Verify we read something
	hasNonZero := false
	for _, b := range buffer {
		if b != 0 {
			hasNonZero = true
			break
		}
	}
	assert.True(t, hasNonZero, "expected to read some non-zero data")
}
