package transformer

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
	"testing/fstest"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSaveConfigsAndLoadModelFromCheckpoint(t *testing.T) {
	// Create a fake HuggingFace repo in a temp directory
	repoDir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(repoDir, "config.json"), []byte(`{
		"model_type": "bert",
		"hidden_size": 256,
		"num_hidden_layers": 4,
		"num_attention_heads": 4,
		"vocab_size": 1000
	}`), 0644))
	require.NoError(t, os.WriteFile(filepath.Join(repoDir, "tokenizer.json"), []byte(`{"version": "1.0"}`), 0644))
	require.NoError(t, os.WriteFile(filepath.Join(repoDir, "vocab.txt"), []byte("hello\nworld\n"), 0644))
	require.NoError(t, os.WriteFile(filepath.Join(repoDir, "model.safetensors"), []byte("weights dummy data"), 0644))

	repo := hub.NewLocal(repoDir)
	m, err := LoadModel(repo)
	require.NoError(t, err)

	checkpointDir := t.TempDir()

	// 1. Test SaveToCheckpoint
	err = m.SaveToCheckpoint(checkpointDir)
	require.NoError(t, err)

	// Verify files in checkpointDir/hf
	hfDir := filepath.Join(checkpointDir, CheckpointHFSubDir)
	assert.FileExists(t, filepath.Join(hfDir, "config.json"))
	assert.FileExists(t, filepath.Join(hfDir, "tokenizer.json"))
	assert.FileExists(t, filepath.Join(hfDir, "vocab.txt"))
	// Model weights should NOT be copied
	assert.NoFileExists(t, filepath.Join(hfDir, "model.safetensors"))

	// 2. Test LoadModelFromCheckpoint
	loadedModel, err := LoadModelFromCheckpoint(checkpointDir)
	require.NoError(t, err)
	assert.Equal(t, "bert", loadedModel.Config.ModelType)
	assert.Equal(t, 256, loadedModel.Config.HiddenSize)
	assert.Equal(t, 4, loadedModel.Config.NumHiddenLayers)
	assert.False(t, loadedModel.UseCausalMask())

	// 3. Test SaveConfigs directly to a custom directory
	customDir := t.TempDir()
	err = m.SaveConfigs(customDir)
	require.NoError(t, err)
	assert.FileExists(t, filepath.Join(customDir, "config.json"))
	assert.FileExists(t, filepath.Join(customDir, "tokenizer.json"))
	assert.FileExists(t, filepath.Join(customDir, "vocab.txt"))
}

func TestLoadModelFromFS(t *testing.T) {
	configJSON := []byte(`{
		"model_type": "gemma",
		"hidden_size": 512,
		"num_hidden_layers": 6,
		"num_attention_heads": 8,
		"vocab_size": 2000
	}`)
	tokenizerJSON := []byte(`{"version": "1.0"}`)

	// 1. Test with in-memory fs.FS (fstest.MapFS)
	mapFS := fstest.MapFS{
		"my_checkpoint/hf/config.json":    &fstest.MapFile{Data: configJSON},
		"my_checkpoint/hf/tokenizer.json": &fstest.MapFile{Data: tokenizerJSON},
	}

	m, err := LoadModelFromFS(mapFS, "my_checkpoint")
	require.NoError(t, err)
	assert.Equal(t, "gemma", m.Config.ModelType)
	assert.Equal(t, 512, m.Config.HiddenSize)
	assert.True(t, m.UseCausalMask())

	// 2. Test with Zip archive as fs.FS
	var zipBuf bytes.Buffer
	zw := zip.NewWriter(&zipBuf)

	cf, err := zw.Create("checkpoint/hf/config.json")
	require.NoError(t, err)
	_, err = cf.Write(configJSON)
	require.NoError(t, err)

	tf, err := zw.Create("checkpoint/hf/tokenizer.json")
	require.NoError(t, err)
	_, err = tf.Write(tokenizerJSON)
	require.NoError(t, err)

	require.NoError(t, zw.Close())

	zipReader, err := zip.NewReader(bytes.NewReader(zipBuf.Bytes()), int64(zipBuf.Len()))
	require.NoError(t, err)

	zipModel, err := LoadModelFromFS(zipReader, "checkpoint")
	require.NoError(t, err)
	assert.Equal(t, "gemma", zipModel.Config.ModelType)
	assert.Equal(t, 512, zipModel.Config.HiddenSize)
	assert.True(t, zipModel.UseCausalMask())
}

// buildDummySafetensors creates a minimal valid .safetensors file with a single F32 tensor.
func buildDummySafetensors(tensorName string, values []float32) []byte {
	var dataBuf bytes.Buffer
	for _, v := range values {
		_ = binary.Write(&dataBuf, binary.LittleEndian, math.Float32bits(v))
	}
	dataBytes := dataBuf.Bytes()

	headerJSON := []byte(`{"` + tensorName + `": {"dtype": "F32", "shape": [` +
		string(rune('0'+len(values))) + `], "data_offsets": [0, ` +
		string(rune('0'+len(dataBytes))) + `]}}`)

	var out bytes.Buffer
	_ = binary.Write(&out, binary.LittleEndian, uint64(len(headerJSON)))
	out.Write(headerJSON)
	out.Write(dataBytes)
	return out.Bytes()
}

func TestLoadStoreFiltered_PreservesExistingVariables(t *testing.T) {
	// Create repo with minimal config and dummy safetensors
	repoDir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(repoDir, "config.json"), []byte(`{
		"model_type": "bert",
		"hidden_size": 2,
		"num_hidden_layers": 1,
		"num_attention_heads": 1,
		"vocab_size": 2
	}`), 0644))

	// Tensor "embeddings.word_embeddings.weight" maps to "/token_embed/embeddings"
	stBytes := buildDummySafetensors("embeddings.word_embeddings.weight", []float32{1.0, 2.0})
	require.NoError(t, os.WriteFile(filepath.Join(repoDir, "model.safetensors"), stBytes, 0644))

	repo := hub.NewLocal(repoDir)
	m, err := LoadModel(repo)
	require.NoError(t, err)

	// 1. Test that when variable ALREADY exists in store, LoadStore preserves it:
	storeWithCheckpoint := model.NewStore()
	checkpointVal := tensors.FromValue([]float32{42.0, 99.0})
	v, err := storeWithCheckpoint.CreateVariable("/token_embed/embeddings", checkpointVal)
	require.NoError(t, err)

	err = m.LoadStore(nil, storeWithCheckpoint)
	require.NoError(t, err)

	// Value in store should remain the checkpoint value: 42.0, 99.0
	actualTensor, err := v.Value()
	require.NoError(t, err)
	assert.Equal(t, []float32{42.0, 99.0}, actualTensor.Value())

	// 2. Test that in a fresh store without previous checkpoint, LoadStore loads 1.0, 2.0:
	freshStore := model.NewStore()
	err = m.LoadStore(nil, freshStore)
	require.NoError(t, err)

	freshVar := freshStore.GetVariable("/token_embed/embeddings")
	require.NotNil(t, freshVar)
	freshTensor, err := freshVar.Value()
	require.NoError(t, err)
	assert.Equal(t, []float32{1.0, 2.0}, freshTensor.Value())
}
