package gguf

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReadTensorF32(t *testing.T) {
	// Create a GGUF file with one F32 tensor [4] containing [1.0, 2.0, 3.0, 4.0].
	tensorData := make([]byte, 16)
	binary.LittleEndian.PutUint32(tensorData[0:4], math.Float32bits(1.0))
	binary.LittleEndian.PutUint32(tensorData[4:8], math.Float32bits(2.0))
	binary.LittleEndian.PutUint32(tensorData[8:12], math.Float32bits(3.0))
	binary.LittleEndian.PutUint32(tensorData[12:16], math.Float32bits(4.0))

	path := buildMinimalGGUF(t, 1, 1,
		func(b *ggufBuilder) {
			b.writeKVString("general.architecture", "test")
		},
		func(b *ggufBuilder) {
			b.writeTensorInfo("weights", []uint64{4}, TensorTypeF32, 0)
		},
		tensorData)

	f, err := Open(path)
	require.NoError(t, err)

	reader, err := NewReader(f)
	require.NoError(t, err)
	defer reader.Close()

	tensor, err := reader.ReadTensor(nil, "weights")
	require.NoError(t, err)

	assert.Equal(t, []int{4}, tensor.Shape().Dimensions)

	// Read back the float32 values.
	var got [4]float32
	tensor.MutableBytes(func(data []byte) {
		for i := range 4 {
			got[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4 : i*4+4]))
		}
	})
	assert.Equal(t, [4]float32{1.0, 2.0, 3.0, 4.0}, got)
}

func TestReadTensorF32_2D(t *testing.T) {
	// Create a GGUF file with one F32 tensor [3, 2] in GGUF order (innermost first).
	// GoMLX shape should be [2, 3] (reversed).
	// 6 floats = 24 bytes.
	tensorData := make([]byte, 24)
	for i := range 6 {
		binary.LittleEndian.PutUint32(tensorData[i*4:i*4+4], math.Float32bits(float32(i+1)))
	}

	path := buildMinimalGGUF(t, 1, 1,
		func(b *ggufBuilder) {
			b.writeKVString("general.architecture", "test")
		},
		func(b *ggufBuilder) {
			b.writeTensorInfo("matrix", []uint64{3, 2}, TensorTypeF32, 0)
		},
		tensorData)

	f, err := Open(path)
	require.NoError(t, err)

	reader, err := NewReader(f)
	require.NoError(t, err)
	defer reader.Close()

	tensor, err := reader.ReadTensor(nil, "matrix")
	require.NoError(t, err)

	// Dimensions reversed: GGUF [3, 2] → GoMLX [2, 3].
	assert.Equal(t, []int{2, 3}, tensor.Shape().Dimensions)
}

func TestReadTensorNotFound(t *testing.T) {
	path := buildMinimalGGUF(t, 1, 0,
		func(b *ggufBuilder) {
			b.writeKVString("general.architecture", "test")
		},
		nil, nil)

	f, err := Open(path)
	require.NoError(t, err)

	reader, err := NewReader(f)
	require.NoError(t, err)
	defer reader.Close()

	_, err = reader.ReadTensor(nil, "nonexistent")
	assert.ErrorContains(t, err, "not found")
}

func TestReadTensorRawAndBytes(t *testing.T) {
	// Create a 2D Q8_0 tensor with GGUF shape [32, 2] (GoMLX logical [2, 32], raw [2, 34] Uint8).
	tensorData := make([]byte, 2*34)
	for i := range tensorData {
		tensorData[i] = byte(i)
	}

	path := buildMinimalGGUF(t, 1, 1,
		func(b *ggufBuilder) {
			b.writeKVString("general.architecture", "test")
		},
		func(b *ggufBuilder) {
			b.writeTensorInfo("raw", []uint64{32, 2}, TensorTypeQ8_0, 0)
		},
		tensorData)

	f, err := Open(path)
	require.NoError(t, err)

	reader, err := NewReader(f)
	require.NoError(t, err)
	defer reader.Close()

	// Test ReadTensorBytes: returns []byte and *TensorInfo.
	raw, info, err := reader.ReadTensorBytes("raw")
	require.NoError(t, err)
	assert.Equal(t, "raw", info.Name)
	assert.Equal(t, tensorData, raw)

	// Test ReadTensorRaw: returns a dtypes.Uint8 GoMLX tensor with shape [2, 34].
	rawTensor, err := reader.ReadTensorRaw(nil, "raw")
	require.NoError(t, err)
	assert.Equal(t, dtypes.Uint8, rawTensor.Shape().DType)
	assert.Equal(t, []int{2, 34}, rawTensor.Shape().Dimensions)
	rawTensor.ConstBytes(func(data []byte) {
		assert.Equal(t, tensorData, data)
	})
}

func TestModelMultiFileAndRaw(t *testing.T) {
	dir := t.TempDir()

	// Primary GGUF file: 1 Q4_0 tensor [32, 2] (2 blocks = 36 bytes).
	primaryData := make([]byte, 36)
	for i := range primaryData {
		primaryData[i] = byte(i + 1)
	}
	primaryPath := buildMinimalGGUF(t, 1, 1,
		func(b *ggufBuilder) {
			b.writeKVString("general.architecture", "llama")
		},
		func(b *ggufBuilder) {
			b.writeTensorInfo("blk.0.attn_q.weight", []uint64{32, 2}, TensorTypeQ4_0, 0)
		},
		primaryData)
	primaryBytes, err := os.ReadFile(primaryPath)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(dir, "01_model.gguf"), primaryBytes, 0644))

	// Extra GGUF file (mmproj): 1 F32 tensor [4] (16 bytes).
	extraData := make([]byte, 16)
	for i := range 4 {
		binary.LittleEndian.PutUint32(extraData[i*4:i*4+4], math.Float32bits(float32(i+10)))
	}
	extraPath := buildMinimalGGUF(t, 1, 1,
		func(b *ggufBuilder) {
			b.writeKVString("general.architecture", "clip")
		},
		func(b *ggufBuilder) {
			b.writeTensorInfo("v.blk.0.weight", []uint64{4}, TensorTypeF32, 0)
		},
		extraData)
	extraBytes, err := os.ReadFile(extraPath)
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(dir, "02_mmproj.gguf"), extraBytes, 0644))

	repo := hub.NewLocal(dir)

	m := NewEmpty(repo)
	ggufFiles, err := m.ListGGUFFiles()
	require.NoError(t, err)
	assert.Len(t, ggufFiles, 2)

	require.NoError(t, m.LoadAll())
	defer m.Close()

	assert.Len(t, m.ExtraFiles(), 1)
	assert.ElementsMatch(t, []string{"blk.0.attn_q.weight", "v.blk.0.weight"}, m.ListTensorNames())

	// Lookup tensor info in both primary and extra files.
	infoQ, ok := m.GetTensorInfo("blk.0.attn_q.weight")
	assert.True(t, ok)
	assert.Equal(t, TensorTypeQ4_0, infoQ.Type)

	infoV, ok := m.GetTensorInfo("v.blk.0.weight")
	assert.True(t, ok)
	assert.Equal(t, TensorTypeF32, infoV.Type)

	// ReadTensorBytes across files.
	gotBytes, gotInfo, err := m.ReadTensorBytes("blk.0.attn_q.weight")
	require.NoError(t, err)
	assert.Equal(t, primaryData, gotBytes)
	assert.Equal(t, "blk.0.attn_q.weight", gotInfo.Name)

	// ReadTensorRaw (Uint8 GoMLX tensor with shape [2, 18]).
	tnRaw, err := m.ReadTensorRaw(nil, "blk.0.attn_q.weight")
	require.NoError(t, err)
	assert.Equal(t, dtypes.Uint8, tnRaw.Tensor.Shape().DType)
	assert.Equal(t, []int{2, 18}, tnRaw.Tensor.Shape().Dimensions)
	tnRaw.Tensor.ConstBytes(func(data []byte) {
		assert.Equal(t, primaryData, data)
	})

	// ReadTensor from extra file.
	tnV, err := m.ReadTensor(nil, "v.blk.0.weight")
	require.NoError(t, err)
	assert.Equal(t, dtypes.Float32, tnV.Tensor.Shape().DType)
	assert.Equal(t, []int{4}, tnV.Tensor.Shape().Dimensions)
}

func TestReadTensorQ8_0(t *testing.T) {
	// Create a Q8_0 tensor with 32 elements (1 block).
	// Block: 2 bytes f16 scale + 32 bytes int8 = 34 bytes.
	// scale = 1.0, values = [0, 1, 2, ..., 31]
	tensorData := make([]byte, 34)
	binary.LittleEndian.PutUint16(tensorData[0:2], float32ToFloat16Bits(1.0))
	for i := range 32 {
		tensorData[2+i] = byte(i)
	}

	path := buildMinimalGGUF(t, 1, 1,
		func(b *ggufBuilder) {
			b.writeKVString("general.architecture", "test")
		},
		func(b *ggufBuilder) {
			b.writeTensorInfo("q8", []uint64{32}, TensorTypeQ8_0, 0)
		},
		tensorData)

	f, err := Open(path)
	require.NoError(t, err)

	reader, err := NewReader(f)
	require.NoError(t, err)
	defer reader.Close()

	// Quantized tensors are dequantized to Float32.
	tensor, err := reader.ReadTensor(nil, "q8")
	require.NoError(t, err)

	assert.Equal(t, []int{32}, tensor.Shape().Dimensions)

	// Read back the dequantized float32 values.
	tensor.MutableBytes(func(data []byte) {
		for i := range 32 {
			got := math.Float32frombits(binary.LittleEndian.Uint32(data[i*4 : i*4+4]))
			assert.InDelta(t, float32(i), got, 0.01, "Q8_0 read index %d", i)
		}
	})
}

func TestReadMultipleTensors(t *testing.T) {
	// Two F32 tensors: [4] at offset 0, [2] at offset 16.
	tensorData := make([]byte, 24)
	for i := range 4 {
		binary.LittleEndian.PutUint32(tensorData[i*4:i*4+4], math.Float32bits(float32(i+1)))
	}
	for i := range 2 {
		binary.LittleEndian.PutUint32(tensorData[16+i*4:16+i*4+4], math.Float32bits(float32(10+i)))
	}

	path := buildMinimalGGUF(t, 1, 2,
		func(b *ggufBuilder) {
			b.writeKVString("general.architecture", "test")
		},
		func(b *ggufBuilder) {
			b.writeTensorInfo("first", []uint64{4}, TensorTypeF32, 0)
			b.writeTensorInfo("second", []uint64{2}, TensorTypeF32, 16)
		},
		tensorData)

	f, err := Open(path)
	require.NoError(t, err)

	reader, err := NewReader(f)
	require.NoError(t, err)
	defer reader.Close()

	t1, err := reader.ReadTensor(nil, "first")
	require.NoError(t, err)
	assert.Equal(t, []int{4}, t1.Shape().Dimensions)

	t2, err := reader.ReadTensor(nil, "second")
	require.NoError(t, err)
	assert.Equal(t, []int{2}, t2.Shape().Dimensions)

	// Verify second tensor values.
	t2.MutableBytes(func(data []byte) {
		v0 := math.Float32frombits(binary.LittleEndian.Uint32(data[0:4]))
		v1 := math.Float32frombits(binary.LittleEndian.Uint32(data[4:8]))
		assert.InDelta(t, 10.0, v0, 0.01)
		assert.InDelta(t, 11.0, v1, 0.01)
	})
}
