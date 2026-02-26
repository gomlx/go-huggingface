package gguf

import (
	"encoding/binary"
	"math"
	"testing"

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

	reader, err := NewMMapReader(path, f)
	require.NoError(t, err)
	defer reader.Close()

	tensor, err := reader.ReadTensor("weights")
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

	reader, err := NewMMapReader(path, f)
	require.NoError(t, err)
	defer reader.Close()

	tensor, err := reader.ReadTensor("matrix")
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

	reader, err := NewMMapReader(path, f)
	require.NoError(t, err)
	defer reader.Close()

	_, err = reader.ReadTensor("nonexistent")
	assert.ErrorContains(t, err, "not found")
}

func TestReadTensorRaw(t *testing.T) {
	// Create tensor data with known bytes.
	tensorData := make([]byte, 16)
	for i := range 16 {
		tensorData[i] = byte(i)
	}

	path := buildMinimalGGUF(t, 1, 1,
		func(b *ggufBuilder) {
			b.writeKVString("general.architecture", "test")
		},
		func(b *ggufBuilder) {
			b.writeTensorInfo("raw", []uint64{4}, TensorTypeF32, 0)
		},
		tensorData)

	f, err := Open(path)
	require.NoError(t, err)

	reader, err := NewMMapReader(path, f)
	require.NoError(t, err)
	defer reader.Close()

	raw, info, err := reader.ReadTensorRaw("raw")
	require.NoError(t, err)
	assert.Equal(t, "raw", info.Name)
	assert.Equal(t, tensorData, raw)
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

	reader, err := NewMMapReader(path, f)
	require.NoError(t, err)
	defer reader.Close()

	// Quantized tensors are dequantized to Float32.
	tensor, err := reader.ReadTensor("q8")
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

	reader, err := NewMMapReader(path, f)
	require.NoError(t, err)
	defer reader.Close()

	t1, err := reader.ReadTensor("first")
	require.NoError(t, err)
	assert.Equal(t, []int{4}, t1.Shape().Dimensions)

	t2, err := reader.ReadTensor("second")
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
