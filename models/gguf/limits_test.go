package gguf

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestOpenRejectsOversizedTensorDims verifies the GHSA-v8r9-hqm6-68cf regression:
// a minimal GGUF file declaring a tensor with huge dimensions (e.g. [2^60]) must
// fail cleanly with an error instead of panicking with "makeslice: len out of range".
func TestOpenRejectsOversizedTensorDims(t *testing.T) {
	// Exact PoC structure from GHSA-v8r9-hqm6-68cf: 1 F32 tensor "x" with dims = [2^60], no tensor data.
	b := newGGUFBuilder()
	b.buf = append(b.buf, "GGUF"...)
	b.writeUint32(3)
	b.writeUint64(1) // tensorCount
	b.writeUint64(0) // kvCount
	b.writeTensorInfo("x", []uint64{1 << 60}, TensorTypeF32, 0)

	path := filepath.Join(t.TempDir(), "poc.gguf")
	require.NoError(t, os.WriteFile(path, b.bytes(), 0644))

	assert.NotPanics(t, func() {
		_, err := Open(path)
		require.Error(t, err)
	})
}

func TestOpenHeaderLimits(t *testing.T) {
	writeRaw := func(t *testing.T, data []byte) string {
		t.Helper()
		p := filepath.Join(t.TempDir(), "malformed.gguf")
		require.NoError(t, os.WriteFile(p, data, 0644))
		return p
	}

	t.Run("tensor_count_exceeds_file", func(t *testing.T) {
		b := newGGUFBuilder()
		b.buf = append(b.buf, "GGUF"...)
		b.writeUint32(3)
		b.writeUint64(100_000) // within maxTensorCount, way larger than 24-byte file
		b.writeUint64(0)
		_, err := Open(writeRaw(t, b.bytes()))
		require.Error(t, err)
		assert.ErrorContains(t, err, "exceeding remaining file size")
	})

	t.Run("kv_count_exceeds_file", func(t *testing.T) {
		b := newGGUFBuilder()
		b.buf = append(b.buf, "GGUF"...)
		b.writeUint32(3)
		b.writeUint64(0)
		b.writeUint64(50_000) // within maxKVCount, way larger than 24-byte file
		_, err := Open(writeRaw(t, b.bytes()))
		require.Error(t, err)
		assert.ErrorContains(t, err, "exceeding remaining file size")
	})

	t.Run("string_len_exceeds_file", func(t *testing.T) {
		b := newGGUFBuilder()
		b.buf = append(b.buf, "GGUF"...)
		b.writeUint32(3)
		b.writeUint64(0)
		b.writeUint64(1)
		b.writeUint64(1 << 20) // 1 MiB string length in a tiny file
		b.buf = append(b.buf, make([]byte, 16)...)
		_, err := Open(writeRaw(t, b.bytes()))
		require.Error(t, err)
		assert.ErrorContains(t, err, "exceeds remaining file size")
	})

	t.Run("numeric_array_exceeds_file", func(t *testing.T) {
		b := newGGUFBuilder()
		b.buf = append(b.buf, "GGUF"...)
		b.writeUint32(3)
		b.writeUint64(0)
		b.writeUint64(1)
		b.writeString("arr")
		b.writeUint32(uint32(valueTypeArray))
		b.writeUint32(uint32(valueTypeFloat64))
		b.writeUint64(1 << 20) // 8 MiB array in tiny file
		_, err := Open(writeRaw(t, b.bytes()))
		require.Error(t, err)
		assert.ErrorContains(t, err, "exceeds remaining file size")
	})

	t.Run("bool_array_exceeds_file", func(t *testing.T) {
		b := newGGUFBuilder()
		b.buf = append(b.buf, "GGUF"...)
		b.writeUint32(3)
		b.writeUint64(0)
		b.writeUint64(1)
		b.writeString("arr")
		b.writeUint32(uint32(valueTypeArray))
		b.writeUint32(uint32(valueTypeBool))
		b.writeUint64(1 << 20)
		_, err := Open(writeRaw(t, b.bytes()))
		require.Error(t, err)
		assert.ErrorContains(t, err, "exceeds remaining file size")
	})

	t.Run("string_array_exceeds_file", func(t *testing.T) {
		b := newGGUFBuilder()
		b.buf = append(b.buf, "GGUF"...)
		b.writeUint32(3)
		b.writeUint64(0)
		b.writeUint64(1)
		b.writeString("arr")
		b.writeUint32(uint32(valueTypeArray))
		b.writeUint32(uint32(valueTypeString))
		b.writeUint64(1 << 20)
		_, err := Open(writeRaw(t, b.bytes()))
		require.Error(t, err)
		assert.ErrorContains(t, err, "exceeds remaining file size")
	})

	t.Run("negative_alignment", func(t *testing.T) {
		path := buildMinimalGGUF(t, 1, 0,
			func(b *ggufBuilder) {
				b.writeString(KeyGeneralAlignment)
				b.writeUint32(uint32(valueTypeInt32))
				b.writeUint32(0xFFFFFFE0) // int32(-32)
			}, nil, nil)
		_, err := Open(path)
		require.Error(t, err)
		assert.ErrorContains(t, err, "negative alignment")
	})

	t.Run("oversized_alignment", func(t *testing.T) {
		path := buildMinimalGGUF(t, 1, 0,
			func(b *ggufBuilder) {
				b.writeString(KeyGeneralAlignment)
				b.writeUint32(uint32(valueTypeUint64))
				b.writeUint64(1 << 30)
			}, nil, nil)
		_, err := Open(path)
		require.Error(t, err)
		assert.ErrorContains(t, err, "alignment")
	})
}

func TestOpenTensorValidation(t *testing.T) {
	cases := []struct {
		name       string
		shape      []uint64
		ttype      TensorType
		offset     uint64
		tensorData []byte
		errSubstr  string
	}{
		{
			name:      "zero_dims",
			shape:     []uint64{},
			ttype:     TensorTypeF32,
			errSubstr: "0 dimensions",
		},
		{
			name:      "dim_exceeds_max_int",
			shape:     []uint64{1 << 63},
			ttype:     TensorTypeF32,
			errSubstr: "exceeds max int",
		},
		{
			name:      "element_count_overflow_uint64",
			shape:     []uint64{1 << 32, 1 << 32},
			ttype:     TensorTypeF32,
			errSubstr: "overflows",
		},
		{
			name:      "byte_size_overflow_int64",
			shape:     []uint64{1 << 62},
			ttype:     TensorTypeF32,
			errSubstr: "exceeds limit",
		},
		{
			name:      "quantized_dim_not_divisible_by_block_size",
			shape:     []uint64{33},
			ttype:     TensorTypeQ4_0,
			errSubstr: "not divisible by block size",
		},
		{
			name:      "zero_outer_dim_raw_inner_overflow",
			shape:     []uint64{1 << 61, 0},
			ttype:     TensorTypeF64,
			errSubstr: "overflows int",
		},
		{
			name:       "tensor_size_exceeds_file_data_section",
			shape:      []uint64{1024},
			ttype:      TensorTypeF32,
			offset:     0,
			tensorData: make([]byte, 16), // needs 4096 bytes
			errSubstr:  "exceeding",
		},
		{
			name:       "tensor_offset_exceeds_file_data_section",
			shape:      []uint64{4},
			ttype:      TensorTypeF32,
			offset:     1024,
			tensorData: make([]byte, 16),
			errSubstr:  "exceeds",
		},
		{
			name:       "tensor_offset_uint64_max",
			shape:      []uint64{4},
			ttype:      TensorTypeF32,
			offset:     math.MaxUint64,
			tensorData: make([]byte, 16),
			errSubstr:  "exceeds",
		},
		{
			name:       "unknown_type_huge_dims",
			shape:      []uint64{1 << 60},
			ttype:      TensorType(9999),
			offset:     0,
			tensorData: make([]byte, 16),
			errSubstr:  "exceeds limit",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := buildMinimalGGUF(t, 0, 1, nil,
				func(b *ggufBuilder) {
					b.writeTensorInfo("bad", tc.shape, tc.ttype, tc.offset)
				},
				tc.tensorData)
			assert.NotPanics(t, func() {
				_, err := Open(path)
				require.Error(t, err)
				assert.ErrorContains(t, err, tc.errSubstr)
			})
		})
	}
}

// TestReaderRejectsInvalidTensorDefenseInDepth ensures Reader methods validate tensor
// bounds before allocating even if a File/TensorInfo was constructed or mutated outside Open.
func TestReaderRejectsInvalidTensorDefenseInDepth(t *testing.T) {
	tmpPath := filepath.Join(t.TempDir(), "tiny.gguf")
	require.NoError(t, os.WriteFile(tmpPath, make([]byte, 64), 0644))

	cases := []struct {
		name string
		info TensorInfo
	}{
		{
			name: "f32_huge_dim_2_60",
			info: TensorInfo{Name: "x", Shape: []uint64{1 << 60}, Type: TensorTypeF32, Offset: 0},
		},
		{
			name: "f32_dim_exceeds_max_int",
			info: TensorInfo{Name: "x", Shape: []uint64{1 << 63}, Type: TensorTypeF32, Offset: 0},
		},
		{
			name: "f32_product_overflow_to_zero",
			info: TensorInfo{Name: "x", Shape: []uint64{1 << 32, 1 << 32}, Type: TensorTypeF32, Offset: 0},
		},
		{
			name: "f32_exceeds_file_size",
			info: TensorInfo{Name: "x", Shape: []uint64{1024}, Type: TensorTypeF32, Offset: 0},
		},
		{
			name: "q4_0_huge_dim_2_60",
			info: TensorInfo{Name: "x", Shape: []uint64{1 << 60}, Type: TensorTypeQ4_0, Offset: 0},
		},
		{
			name: "q4_0_exceeds_file_size",
			info: TensorInfo{Name: "x", Shape: []uint64{1024}, Type: TensorTypeQ4_0, Offset: 0},
		},
		{
			name: "q4_0_misaligned_dim",
			info: TensorInfo{Name: "x", Shape: []uint64{33}, Type: TensorTypeQ4_0, Offset: 0},
		},
		{
			name: "offset_wraparound",
			info: TensorInfo{Name: "x", Shape: []uint64{4}, Type: TensorTypeF32, Offset: math.MaxUint64},
		},
		{
			name: "unsupported_type",
			info: TensorInfo{Name: "x", Shape: []uint64{4}, Type: TensorType(9999), Offset: 0},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := &File{
				Version:      3,
				Alignment:    32,
				TensorInfos:  []TensorInfo{tc.info},
				tensorByName: map[string]TensorInfo{tc.info.Name: tc.info},
				path:         tmpPath,
				dataOffset:   32,
				fileSize:     64,
			}
			r, err := NewReader(f)
			require.NoError(t, err)
			defer r.Close()

			assert.NotPanics(t, func() {
				_, _, err := r.ReadTensorBytes(tc.info.Name)
				assert.Error(t, err, "ReadTensorBytes should fail")
			})
			assert.NotPanics(t, func() {
				_, err := r.ReadTensor(nil, tc.info.Name)
				assert.Error(t, err, "ReadTensor should fail")
			})
			assert.NotPanics(t, func() {
				_, err := r.ReadTensorRaw(nil, tc.info.Name)
				assert.Error(t, err, "ReadTensorRaw should fail")
			})
		})
	}
}

func TestTensorInfoOverflowHelpers(t *testing.T) {
	tiOverflow := TensorInfo{Shape: []uint64{1 << 32, 1 << 32}, Type: TensorTypeF32}
	assert.Equal(t, uint64(0), tiOverflow.NumElements())
	assert.Equal(t, int64(0), tiOverflow.NumBytes())

	tiByteOverflow := TensorInfo{Shape: []uint64{1 << 62}, Type: TensorTypeF32}
	assert.Equal(t, uint64(1<<62), tiByteOverflow.NumElements())
	assert.Equal(t, int64(0), tiByteOverflow.NumBytes())
}
