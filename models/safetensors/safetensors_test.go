package safetensors

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestLoadModel tests loading a model as a unified Model interface.
func TestLoadModel(t *testing.T) {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		t.Skip("skipping test; HF_TOKEN not set")
	}
	// Test with single-file model
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2").WithAuth(token)
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
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		t.Skip("skipping test; HF_TOKEN not set")
	}
	// Test non-sharded model
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2").WithAuth(token)
	model := NewEmpty(repo)
	indexFile, isSharded, err := model.DetectShardedModel()
	require.NoError(t, err)
	assert.False(t, isSharded)
	assert.Empty(t, indexFile)
}

// TestLoadSingleFileModel tests loading a single-file safetensors model.
func TestLoadSingleFileModel(t *testing.T) {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		t.Skip("skipping test; HF_TOKEN not set")
	}
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2").WithAuth(token)
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
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		t.Skip("skipping test; HF_TOKEN not set")
	}
	t.Skip("Requires a sharded model")
}

// TestGetSafetensor tests getting safetensor file information.
func TestGetSafetensor(t *testing.T) {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		t.Skip("skipping test; HF_TOKEN not set")
	}
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2").WithAuth(token)
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
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		t.Skip("skipping test; HF_TOKEN not set")
	}
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2").WithAuth(token)
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
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		t.Skip("skipping test; HF_TOKEN not set")
	}
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2").WithAuth(token)
	m, err := New(repo)
	require.NoError(t, err)

	tensor, err := m.GetTensorFromFile(nil, "model.safetensors", "embeddings.position_embeddings.weight")
	require.NoError(t, err)
	assert.NotNil(t, tensor)
	assert.NotNil(t, tensor.Tensor)
	assert.Equal(t, "embeddings.position_embeddings.weight", tensor.Name)
	assert.Greater(t, tensor.Tensor.Shape().Size(), 0)
}

// TestGetTensor tests loading a specific tensor as GoMLX tensor.
func TestGetTensor(t *testing.T) {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		t.Skip("skipping test; HF_TOKEN not set")
	}
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2").WithAuth(token)
	m, err := New(repo)
	require.NoError(t, err)

	tensor, err := m.GetTensor(nil, "embeddings.position_embeddings.weight")
	require.NoError(t, err)
	assert.NotNil(t, tensor)
	assert.NotNil(t, tensor.Tensor)
	assert.Equal(t, "embeddings.position_embeddings.weight", tensor.Name)
	assert.Greater(t, tensor.Tensor.Shape().Size(), 0)
}

var allMiniVariablesToShape = map[string]string{
	"embeddings.position_ids":                           "(Int64)[1, 512]",
	"embeddings.LayerNorm.bias":                         "(Float32)[384]",
	"embeddings.LayerNorm.weight":                       "(Float32)[384]",
	"embeddings.position_embeddings.weight":             "(Float32)[512, 384]",
	"embeddings.token_type_embeddings.weight":           "(Float32)[2, 384]",
	"embeddings.word_embeddings.weight":                 "(Float32)[30522, 384]",
	"encoder.layer.0.attention.output.LayerNorm.bias":   "(Float32)[384]",
	"encoder.layer.0.attention.output.LayerNorm.weight": "(Float32)[384]",
	"encoder.layer.0.attention.output.dense.bias":       "(Float32)[384]",
	"encoder.layer.0.attention.output.dense.weight":     "(Float32)[384, 384]",
	"encoder.layer.0.attention.self.key.bias":           "(Float32)[384]",
	"encoder.layer.0.attention.self.key.weight":         "(Float32)[384, 384]",
	"encoder.layer.0.attention.self.query.bias":         "(Float32)[384]",
	"encoder.layer.0.attention.self.query.weight":       "(Float32)[384, 384]",
	"encoder.layer.0.attention.self.value.bias":         "(Float32)[384]",
	"encoder.layer.0.attention.self.value.weight":       "(Float32)[384, 384]",
	"encoder.layer.0.intermediate.dense.bias":           "(Float32)[1536]",
	"encoder.layer.0.intermediate.dense.weight":         "(Float32)[1536, 384]",
	"encoder.layer.0.output.LayerNorm.bias":             "(Float32)[384]",
	"encoder.layer.0.output.LayerNorm.weight":           "(Float32)[384]",
	"encoder.layer.0.output.dense.bias":                 "(Float32)[384]",
	"encoder.layer.0.output.dense.weight":               "(Float32)[384, 1536]",
	"encoder.layer.1.attention.output.LayerNorm.bias":   "(Float32)[384]",
	"encoder.layer.1.attention.output.LayerNorm.weight": "(Float32)[384]",
	"encoder.layer.1.attention.output.dense.bias":       "(Float32)[384]",
	"encoder.layer.1.attention.output.dense.weight":     "(Float32)[384, 384]",
	"encoder.layer.1.attention.self.key.bias":           "(Float32)[384]",
	"encoder.layer.1.attention.self.key.weight":         "(Float32)[384, 384]",
	"encoder.layer.1.attention.self.query.bias":         "(Float32)[384]",
	"encoder.layer.1.attention.self.query.weight":       "(Float32)[384, 384]",
	"encoder.layer.1.attention.self.value.bias":         "(Float32)[384]",
	"encoder.layer.1.attention.self.value.weight":       "(Float32)[384, 384]",
	"encoder.layer.1.intermediate.dense.bias":           "(Float32)[1536]",
	"encoder.layer.1.intermediate.dense.weight":         "(Float32)[1536, 384]",
	"encoder.layer.1.output.LayerNorm.bias":             "(Float32)[384]",
	"encoder.layer.1.output.LayerNorm.weight":           "(Float32)[384]",
	"encoder.layer.1.output.dense.bias":                 "(Float32)[384]",
	"encoder.layer.1.output.dense.weight":               "(Float32)[384, 1536]",
	"encoder.layer.2.attention.output.LayerNorm.bias":   "(Float32)[384]",
	"encoder.layer.2.attention.output.LayerNorm.weight": "(Float32)[384]",
	"encoder.layer.2.attention.output.dense.bias":       "(Float32)[384]",
	"encoder.layer.2.attention.output.dense.weight":     "(Float32)[384, 384]",
	"encoder.layer.2.attention.self.key.bias":           "(Float32)[384]",
	"encoder.layer.2.attention.self.key.weight":         "(Float32)[384, 384]",
	"encoder.layer.2.attention.self.query.bias":         "(Float32)[384]",
	"encoder.layer.2.attention.self.query.weight":       "(Float32)[384, 384]",
	"encoder.layer.2.attention.self.value.bias":         "(Float32)[384]",
	"encoder.layer.2.attention.self.value.weight":       "(Float32)[384, 384]",
	"encoder.layer.2.intermediate.dense.bias":           "(Float32)[1536]",
	"encoder.layer.2.intermediate.dense.weight":         "(Float32)[1536, 384]",
	"encoder.layer.2.output.LayerNorm.bias":             "(Float32)[384]",
	"encoder.layer.2.output.LayerNorm.weight":           "(Float32)[384]",
	"encoder.layer.2.output.dense.bias":                 "(Float32)[384]",
	"encoder.layer.2.output.dense.weight":               "(Float32)[384, 1536]",
	"encoder.layer.3.attention.output.LayerNorm.bias":   "(Float32)[384]",
	"encoder.layer.3.attention.output.LayerNorm.weight": "(Float32)[384]",
	"encoder.layer.3.attention.output.dense.bias":       "(Float32)[384]",
	"encoder.layer.3.attention.output.dense.weight":     "(Float32)[384, 384]",
	"encoder.layer.3.attention.self.key.bias":           "(Float32)[384]",
	"encoder.layer.3.attention.self.key.weight":         "(Float32)[384, 384]",
	"encoder.layer.3.attention.self.query.bias":         "(Float32)[384]",
	"encoder.layer.3.attention.self.query.weight":       "(Float32)[384, 384]",
	"encoder.layer.3.attention.self.value.bias":         "(Float32)[384]",
	"encoder.layer.3.attention.self.value.weight":       "(Float32)[384, 384]",
	"encoder.layer.3.intermediate.dense.bias":           "(Float32)[1536]",
	"encoder.layer.3.intermediate.dense.weight":         "(Float32)[1536, 384]",
	"encoder.layer.3.output.LayerNorm.bias":             "(Float32)[384]",
	"encoder.layer.3.output.LayerNorm.weight":           "(Float32)[384]",
	"encoder.layer.3.output.dense.bias":                 "(Float32)[384]",
	"encoder.layer.3.output.dense.weight":               "(Float32)[384, 1536]",
	"encoder.layer.4.attention.output.LayerNorm.bias":   "(Float32)[384]",
	"encoder.layer.4.attention.output.LayerNorm.weight": "(Float32)[384]",
	"encoder.layer.4.attention.output.dense.bias":       "(Float32)[384]",
	"encoder.layer.4.attention.output.dense.weight":     "(Float32)[384, 384]",
	"encoder.layer.4.attention.self.key.bias":           "(Float32)[384]",
	"encoder.layer.4.attention.self.key.weight":         "(Float32)[384, 384]",
	"encoder.layer.4.attention.self.query.bias":         "(Float32)[384]",
	"encoder.layer.4.attention.self.query.weight":       "(Float32)[384, 384]",
	"encoder.layer.4.attention.self.value.bias":         "(Float32)[384]",
	"encoder.layer.4.attention.self.value.weight":       "(Float32)[384, 384]",
	"encoder.layer.4.intermediate.dense.bias":           "(Float32)[1536]",
	"encoder.layer.4.intermediate.dense.weight":         "(Float32)[1536, 384]",
	"encoder.layer.4.output.LayerNorm.bias":             "(Float32)[384]",
	"encoder.layer.4.output.LayerNorm.weight":           "(Float32)[384]",
	"encoder.layer.4.output.dense.bias":                 "(Float32)[384]",
	"encoder.layer.4.output.dense.weight":               "(Float32)[384, 1536]",
	"encoder.layer.5.attention.output.LayerNorm.bias":   "(Float32)[384]",
	"encoder.layer.5.attention.output.LayerNorm.weight": "(Float32)[384]",
	"encoder.layer.5.attention.output.dense.bias":       "(Float32)[384]",
	"encoder.layer.5.attention.output.dense.weight":     "(Float32)[384, 384]",
	"encoder.layer.5.attention.self.key.bias":           "(Float32)[384]",
	"encoder.layer.5.attention.self.key.weight":         "(Float32)[384, 384]",
	"encoder.layer.5.attention.self.query.bias":         "(Float32)[384]",
	"encoder.layer.5.attention.self.query.weight":       "(Float32)[384, 384]",
	"encoder.layer.5.attention.self.value.bias":         "(Float32)[384]",
	"encoder.layer.5.attention.self.value.weight":       "(Float32)[384, 384]",
	"encoder.layer.5.intermediate.dense.bias":           "(Float32)[1536]",
	"encoder.layer.5.intermediate.dense.weight":         "(Float32)[1536, 384]",
	"encoder.layer.5.output.LayerNorm.bias":             "(Float32)[384]",
	"encoder.layer.5.output.LayerNorm.weight":           "(Float32)[384]",
	"encoder.layer.5.output.dense.bias":                 "(Float32)[384]",
	"encoder.layer.5.output.dense.weight":               "(Float32)[384, 1536]",
	"pooler.dense.bias":                                 "(Float32)[384]",
	"pooler.dense.weight":                               "(Float32)[384, 384]",
}

// TestIterTensors tests iterating over all tensors.
func TestIterTensors(t *testing.T) {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		t.Skip("skipping test; HF_TOKEN not set")
	}
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2").WithAuth(token)
	m, err := New(repo)
	require.NoError(t, err)
	count := 0
	for tensorAndName, err := range m.IterTensors(nil) {
		require.NoError(t, err)
		wantShapeStr, found := allMiniVariablesToShape[tensorAndName.Name]
		require.True(t, found, "tensor %q not expected", tensorAndName.Name)
		assert.Equal(t, wantShapeStr, tensorAndName.Tensor.Shape().String())
		count++
	}
	assert.Equal(t, len(allMiniVariablesToShape), count)
}

func TestIterTensorsFromRepo(t *testing.T) {
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		t.Skip("skipping test; HF_TOKEN not set")
	}
	repo := hub.New("sentence-transformers/all-MiniLM-L6-v2").WithAuth(token)
	count := 0
	for tensorAndName, err := range IterTensorsFromRepo(nil, repo) {
		require.NoError(t, err)
		wantShapeStr, found := allMiniVariablesToShape[tensorAndName.Name]
		require.True(t, found, "tensor %q not expected", tensorAndName.Name)
		assert.Equal(t, wantShapeStr, tensorAndName.Tensor.Shape().String())
		count++
	}
	assert.Equal(t, len(allMiniVariablesToShape), count)
}

func TestWithMMapLocal(t *testing.T) {
	dir := t.TempDir()
	headerJSON := `{"w1":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"b1":{"dtype":"F32","shape":[3],"data_offsets":[16,28]}}`

	var buf bytes.Buffer
	require.NoError(t, binary.Write(&buf, binary.LittleEndian, uint64(len(headerJSON))))
	buf.WriteString(headerJSON)
	wantValues := map[string][]float32{
		"w1": {1.0, 2.0, 3.0, 4.0},
		"b1": {-0.5, 0.25, 1.5},
	}
	for _, v := range append(wantValues["w1"], wantValues["b1"]...) {
		require.NoError(t, binary.Write(&buf, binary.LittleEndian, math.Float32bits(v)))
	}
	require.NoError(t, os.WriteFile(filepath.Join(dir, "model.safetensors"), buf.Bytes(), 0644))

	repo := hub.NewLocal(dir)

	for _, useMMap := range []bool{true, false} {
		t.Run(fmt.Sprintf("mmap=%v", useMMap), func(t *testing.T) {
			m := NewEmpty(repo).WithMMap(useMMap)
			assert.Equal(t, useMMap && DefaultUseMMap(), m.UseMMap())
			require.NoError(t, m.Load())

			// 1. Check TensorReader directly
			tr, err := m.NewTensorReader("model.safetensors")
			require.NoError(t, err)
			if m.UseMMap() {
				assert.NotNil(t, tr.mmapBuf)
			} else {
				assert.Nil(t, tr.mmapBuf)
				assert.NotNil(t, tr.file)
			}
			tW1, err := tr.ReadTensor(nil, "w1")
			require.NoError(t, err)
			gotW1, err := tensors.CopyFlatData[float32](tW1)
			require.NoError(t, err)
			assert.Equal(t, wantValues["w1"], gotW1)
			require.NoError(t, tr.Close())

			// 2. Check Model.GetTensor
			tanB1, err := m.GetTensor(nil, "b1")
			require.NoError(t, err)
			gotB1, err := tensors.CopyFlatData[float32](tanB1.Tensor)
			require.NoError(t, err)
			assert.Equal(t, wantValues["b1"], gotB1)

			// 3. Check Model.IterTensors
			seen := make(map[string][]float32)
			for tan, err := range m.IterTensors(nil) {
				require.NoError(t, err)
				vals, err := tensors.CopyFlatData[float32](tan.Tensor)
				require.NoError(t, err)
				seen[tan.Name] = vals
			}
			assert.Equal(t, wantValues, seen)

			// 4. Check IterTensorsFromRepoWithMMap
			seenRepo := make(map[string][]float32)
			for tan, err := range IterTensorsFromRepoWithMMap(nil, repo, useMMap) {
				require.NoError(t, err)
				vals, err := tensors.CopyFlatData[float32](tan.Tensor)
				require.NoError(t, err)
				seenRepo[tan.Name] = vals
			}
			assert.Equal(t, wantValues, seenRepo)
		})
	}
}
