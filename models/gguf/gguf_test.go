package gguf

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ggufBuilder constructs a minimal valid GGUF binary for testing.
type ggufBuilder struct {
	buf []byte
}

func newGGUFBuilder() *ggufBuilder {
	return &ggufBuilder{}
}

func (b *ggufBuilder) writeUint8(v uint8)   { b.buf = append(b.buf, v) }
func (b *ggufBuilder) writeUint16(v uint16) { b.buf = binary.LittleEndian.AppendUint16(b.buf, v) }
func (b *ggufBuilder) writeUint32(v uint32) { b.buf = binary.LittleEndian.AppendUint32(b.buf, v) }
func (b *ggufBuilder) writeUint64(v uint64) { b.buf = binary.LittleEndian.AppendUint64(b.buf, v) }
func (b *ggufBuilder) writeInt32(v int32)   { b.writeUint32(uint32(v)) }
func (b *ggufBuilder) writeFloat32(v float32) {
	b.writeUint32(math.Float32bits(v))
}

func (b *ggufBuilder) writeString(s string) {
	b.writeUint64(uint64(len(s)))
	b.buf = append(b.buf, s...)
}

func (b *ggufBuilder) writeKVString(key, value string) {
	b.writeString(key)
	b.writeUint32(uint32(valueTypeString))
	b.writeString(value)
}

func (b *ggufBuilder) writeKVUint32(key string, value uint32) {
	b.writeString(key)
	b.writeUint32(uint32(valueTypeUint32))
	b.writeUint32(value)
}

func (b *ggufBuilder) writeKVUint64(key string, value uint64) {
	b.writeString(key)
	b.writeUint32(uint32(valueTypeUint64))
	b.writeUint64(value)
}

func (b *ggufBuilder) writeKVFloat32(key string, value float32) {
	b.writeString(key)
	b.writeUint32(uint32(valueTypeFloat32))
	b.writeFloat32(value)
}

func (b *ggufBuilder) writeKVBool(key string, value bool) {
	b.writeString(key)
	b.writeUint32(uint32(valueTypeBool))
	if value {
		b.writeUint8(1)
	} else {
		b.writeUint8(0)
	}
}

func (b *ggufBuilder) writeKVStringArray(key string, values []string) {
	b.writeString(key)
	b.writeUint32(uint32(valueTypeArray))
	b.writeUint32(uint32(valueTypeString))
	b.writeUint64(uint64(len(values)))
	for _, v := range values {
		b.writeString(v)
	}
}

func (b *ggufBuilder) writeKVInt32Array(key string, values []int32) {
	b.writeString(key)
	b.writeUint32(uint32(valueTypeArray))
	b.writeUint32(uint32(valueTypeInt32))
	b.writeUint64(uint64(len(values)))
	for _, v := range values {
		b.writeInt32(v)
	}
}

func (b *ggufBuilder) writeTensorInfo(name string, shape []uint64, ttype TensorType, offset uint64) {
	b.writeString(name)
	b.writeUint32(uint32(len(shape)))
	for _, d := range shape {
		b.writeUint64(d)
	}
	b.writeUint32(uint32(ttype))
	b.writeUint64(offset)
}

func (b *ggufBuilder) bytes() []byte { return b.buf }

// buildMinimalGGUF creates a minimal valid GGUF v3 file in a temp directory.
func buildMinimalGGUF(t *testing.T, kvCount, tensorCount int, writeKVs func(*ggufBuilder), writeTensors func(*ggufBuilder), tensorData []byte) string {
	t.Helper()

	b := newGGUFBuilder()

	// Magic.
	b.buf = append(b.buf, "GGUF"...)
	// Version 3.
	b.writeUint32(3)
	// Tensor count.
	b.writeUint64(uint64(tensorCount))
	// KV count.
	b.writeUint64(uint64(kvCount))

	// Key-value pairs.
	if writeKVs != nil {
		writeKVs(b)
	}

	// Tensor infos.
	if writeTensors != nil {
		writeTensors(b)
	}

	// Pad to default alignment (32 bytes).
	for len(b.buf)%32 != 0 {
		b.buf = append(b.buf, 0)
	}

	// Tensor data.
	if tensorData != nil {
		b.buf = append(b.buf, tensorData...)
	}

	path := filepath.Join(t.TempDir(), "test.gguf")
	require.NoError(t, os.WriteFile(path, b.bytes(), 0644))
	return path
}

func TestOpenValidFile(t *testing.T) {
	path := buildMinimalGGUF(t, 1, 0,
		func(b *ggufBuilder) {
			b.writeKVString("general.architecture", "llama")
		},
		nil, nil)

	f, err := Open(path)
	require.NoError(t, err)
	assert.Equal(t, uint32(3), f.Version)
	assert.Len(t, f.KeyValues, 1)
	assert.Len(t, f.TensorInfos, 0)
	assert.Equal(t, "llama", f.Architecture())
}

func TestOpenInvalidMagic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.gguf")
	require.NoError(t, os.WriteFile(path, []byte("BADx"), 0644))

	_, err := Open(path)
	assert.ErrorContains(t, err, "invalid magic")
}

func TestOpenUnsupportedVersion(t *testing.T) {
	b := newGGUFBuilder()
	b.buf = append(b.buf, "GGUF"...)
	b.writeUint32(1) // Version 1, unsupported.
	b.writeUint64(0) // tensor count
	b.writeUint64(0) // kv count

	path := filepath.Join(t.TempDir(), "old.gguf")
	require.NoError(t, os.WriteFile(path, b.bytes(), 0644))

	_, err := Open(path)
	assert.ErrorContains(t, err, "unsupported version")
}

func TestMetadataTypes(t *testing.T) {
	path := buildMinimalGGUF(t, 4, 0,
		func(b *ggufBuilder) {
			b.writeKVString("general.architecture", "llama")
			b.writeKVUint32("llama.block_count", 32)
			b.writeKVBool("llama.use_parallel_residual", true)
			b.writeKVStringArray("tokenizer.ggml.tokens", []string{"hello", "world", "!"})
		},
		nil, nil)

	f, err := Open(path)
	require.NoError(t, err)

	kv, ok := f.GetKeyValue("general.architecture")
	assert.True(t, ok)
	assert.Equal(t, "llama", kv.String())

	kv, ok = f.GetKeyValue("llama.block_count")
	assert.True(t, ok)
	assert.Equal(t, uint64(32), kv.Uint())
	assert.Equal(t, int64(32), kv.Int())

	kv, ok = f.GetKeyValue("llama.use_parallel_residual")
	assert.True(t, ok)
	assert.True(t, kv.Bool())

	kv, ok = f.GetKeyValue("tokenizer.ggml.tokens")
	assert.True(t, ok)
	assert.Equal(t, []string{"hello", "world", "!"}, kv.Strings())

	_, ok = f.GetKeyValue("does.not.exist")
	assert.False(t, ok)
}

func TestTensorInfoParsing(t *testing.T) {
	// Create 2 F32 tensors: [3, 4] and [5].
	// Tensor data: 12 floats (48 bytes) + 5 floats (20 bytes) = 68 bytes.
	tensorData := make([]byte, 68)

	path := buildMinimalGGUF(t, 1, 2,
		func(b *ggufBuilder) {
			b.writeKVString("general.architecture", "test")
		},
		func(b *ggufBuilder) {
			// GGUF stores dims innermost-first, so [3, 4] in GGUF means
			// the tensor has 4 rows and 3 columns (HF convention [4, 3]).
			b.writeTensorInfo("weight1", []uint64{3, 4}, TensorTypeF32, 0)
			b.writeTensorInfo("weight2", []uint64{5}, TensorTypeF32, 48)
		},
		tensorData)

	f, err := Open(path)
	require.NoError(t, err)
	assert.Len(t, f.TensorInfos, 2)

	info1, ok := f.GetTensorInfo("weight1")
	assert.True(t, ok)
	assert.Equal(t, "weight1", info1.Name)
	assert.Equal(t, []uint64{3, 4}, info1.Shape)
	assert.Equal(t, TensorTypeF32, info1.Type)
	assert.Equal(t, uint64(0), info1.Offset)
	assert.Equal(t, uint64(12), info1.NumElements())
	assert.Equal(t, int64(48), info1.NumBytes())

	// GoMLXShape reverses dimensions.
	_, dims := info1.GoMLXShape()
	assert.Equal(t, []int{4, 3}, dims)

	info2, ok := f.GetTensorInfo("weight2")
	assert.True(t, ok)
	assert.Equal(t, uint64(48), info2.Offset)
	assert.Equal(t, uint64(5), info2.NumElements())
}

func TestListTensorNames(t *testing.T) {
	path := buildMinimalGGUF(t, 1, 2,
		func(b *ggufBuilder) {
			b.writeKVString("general.architecture", "test")
		},
		func(b *ggufBuilder) {
			b.writeTensorInfo("a.weight", []uint64{4}, TensorTypeF32, 0)
			b.writeTensorInfo("b.weight", []uint64{4}, TensorTypeF32, 16)
		},
		make([]byte, 32))

	f, err := Open(path)
	require.NoError(t, err)

	names := f.ListTensorNames()
	assert.Len(t, names, 2)
	assert.Contains(t, names, "a.weight")
	assert.Contains(t, names, "b.weight")
}

func TestTensorTypeProperties(t *testing.T) {
	tests := []struct {
		tt        TensorType
		blockSize int
		typeSize  int
		quantized bool
		name      string
	}{
		{TensorTypeF32, 1, 4, false, "F32"},
		{TensorTypeF16, 1, 2, false, "F16"},
		{TensorTypeBF16, 1, 2, false, "BF16"},
		{TensorTypeQ4_0, 32, 18, true, "Q4_0"},
		{TensorTypeQ8_0, 32, 34, true, "Q8_0"},
		{TensorTypeQ4_K, 256, 144, true, "Q4_K"},
		{TensorTypeQ6_K, 256, 210, true, "Q6_K"},
		{TensorTypeI8, 1, 1, false, "I8"},
		{TensorTypeI32, 1, 4, false, "I32"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.blockSize, tt.tt.BlockSize(), "BlockSize")
			assert.Equal(t, tt.typeSize, tt.tt.TypeSize(), "TypeSize")
			assert.Equal(t, tt.quantized, tt.tt.IsQuantized(), "IsQuantized")
			assert.Equal(t, tt.name, tt.tt.String(), "String")
		})
	}
}
