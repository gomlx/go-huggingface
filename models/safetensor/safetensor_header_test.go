package safetensor

import (
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestParseSafetensorHeader tests parsing safetensor headers.
func TestParseSafetensorHeader(t *testing.T) {
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2")
	model, err := NewModelSafetensor(repo)
	require.NoError(t, err)

	localPath, err := repo.DownloadFile("model.safetensors")
	require.NoError(t, err)

	header, dataOffset, err := model.ParseSafetensorHeader(localPath)
	require.NoError(t, err)
	assert.NotNil(t, header)
	assert.Greater(t, dataOffset, int64(0))
	assert.Greater(t, len(header.Tensors), 0)

	// Verify header structure
	for tensorName, meta := range header.Tensors {
		assert.NotEmpty(t, tensorName)
		assert.NotEmpty(t, meta.Dtype)
		assert.NotNil(t, meta.Shape)
		assert.Equal(t, tensorName, meta.Name)
	}
}

// TestSafetensorDtypeToGoMLX tests dtype conversion using GoMLX's DType.Size() method.
func TestSafetensorDtypeToGoMLX(t *testing.T) {
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
			// Convert safetensor dtype name to GoMLX dtype
			gomlxDtype, err := safetensorDtypeToGoMLX(tt.dtype)
			require.NoError(t, err)
			assert.Equal(t, tt.expected, gomlxDtype.Size())
		})
	}

	// Test unknown dtype
	_, err := safetensorDtypeToGoMLX("UNKNOWN")
	assert.Error(t, err)
}
