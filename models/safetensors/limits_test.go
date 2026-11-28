package safetensors

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"math/bits"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func writeSafetensorsFile(t *testing.T, dir, name, headerJSON string, tensorData []byte) string {
	t.Helper()
	var buf bytes.Buffer
	require.NoError(t, binary.Write(&buf, binary.LittleEndian, uint64(len(headerJSON))))
	buf.WriteString(headerJSON)
	buf.Write(tensorData)
	p := filepath.Join(dir, name)
	require.NoError(t, os.WriteFile(p, buf.Bytes(), 0644))
	return p
}

func TestParseHeaderRejectsInvalidLimitsAndMetadata(t *testing.T) {
	t.Run("file_too_small", func(t *testing.T) {
		p := filepath.Join(t.TempDir(), "tiny.safetensors")
		require.NoError(t, os.WriteFile(p, []byte{1, 2, 3}, 0644))
		f, err := os.Open(p)
		require.NoError(t, err)
		defer f.Close()
		_, _, err = parseHeaderFromReader(f)
		require.Error(t, err)
		assert.ErrorContains(t, err, "too small")
	})

	t.Run("zero_header_size", func(t *testing.T) {
		p := filepath.Join(t.TempDir(), "zero.safetensors")
		var buf [8]byte
		require.NoError(t, os.WriteFile(p, buf[:], 0644))
		f, err := os.Open(p)
		require.NoError(t, err)
		defer f.Close()
		_, _, err = parseHeaderFromReader(f)
		require.Error(t, err)
		assert.ErrorContains(t, err, "0 bytes")
	})

	t.Run("header_size_exceeds_file_size", func(t *testing.T) {
		// Claims 50 MiB header (under 100 MiB cap) in a 16-byte file: must fail
		// immediately against file size without allocating 50 MiB.
		p := filepath.Join(t.TempDir(), "oversized_hdr.safetensors")
		var buf [16]byte
		binary.LittleEndian.PutUint64(buf[:8], 50*1024*1024)
		require.NoError(t, os.WriteFile(p, buf[:], 0644))
		f, err := os.Open(p)
		require.NoError(t, err)
		defer f.Close()
		_, _, err = parseHeaderFromReader(f)
		require.Error(t, err)
		assert.ErrorContains(t, err, "exceeds remaining file size")
	})

	cases := []struct {
		name       string
		headerJSON string
		dataLen    int
		errSubstr  string
	}{
		{
			name:       "negative_dim",
			headerJSON: `{"x":{"dtype":"F32","shape":[-1,4],"data_offsets":[0,16]}}`,
			dataLen:    16,
			errSubstr:  "negative dimension",
		},
		{
			name:       "excessive_rank",
			headerJSON: fmt.Sprintf(`{"x":{"dtype":"F32","shape":[%s],"data_offsets":[0,4]}}`, strings.Repeat("1,", maxTensorDims)+"1"),
			dataLen:    4,
			errSubstr:  "exceeds limit",
		},
		{
			name:       "dim_product_overflow_wraps_to_zero",
			headerJSON: `{"x":{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[0,0]}}`,
			dataLen:    0,
			errSubstr:  "overflows",
		},
		{
			name:       "byte_size_overflow_wraps_to_16",
			headerJSON: `{"x":{"dtype":"F32","shape":[1,4611686018427387908],"data_offsets":[0,16]}}`,
			dataLen:    16,
			errSubstr:  "overflows",
		},
		{
			name:       "huge_dim_exceeds_limit",
			headerJSON: `{"x":{"dtype":"F32","shape":[1152921504606846976],"data_offsets":[0,16]}}`,
			dataLen:    16,
			errSubstr:  "exceeds limit",
		},
		{
			name:       "invalid_dtype_name",
			headerJSON: `{"x":{"dtype":"INVALID","shape":[4],"data_offsets":[0,16]}}`,
			dataLen:    16,
			errSubstr:  "invalid dtype",
		},
		{
			name:       "empty_dtype_name",
			headerJSON: `{"x":{"dtype":"","shape":[4],"data_offsets":[0,16]}}`,
			dataLen:    16,
			errSubstr:  "empty dtype",
		},
		{
			name:       "negative_start_offset",
			headerJSON: `{"x":{"dtype":"F32","shape":[4],"data_offsets":[-16,0]}}`,
			dataLen:    16,
			errSubstr:  "invalid data offsets",
		},
		{
			name:       "reversed_offsets",
			headerJSON: `{"x":{"dtype":"F32","shape":[4],"data_offsets":[16,0]}}`,
			dataLen:    16,
			errSubstr:  "invalid data offsets",
		},
		{
			name:       "offsets_overflow_int64_with_data_offset",
			headerJSON: fmt.Sprintf(`{"x":{"dtype":"F32","shape":[4],"data_offsets":[%d,%d]}}`, int64(math.MaxInt64-16), int64(math.MaxInt64)),
			dataLen:    16,
			errSubstr:  "overflow",
		},
		{
			name:       "offsets_beyond_data_section",
			headerJSON: `{"x":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}`,
			dataLen:    8,
			errSubstr:  "out of range",
		},
		{
			name:       "offsets_size_mismatch_shape",
			headerJSON: `{"x":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}`,
			dataLen:    16,
			errSubstr:  "expected 16 bytes",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			writeSafetensorsFile(t, dir, "model.safetensors", tc.headerJSON, make([]byte, tc.dataLen))
			repo := hub.NewLocal(dir)

			// 1. Model.Load rejects eagerly at header parse time.
			assert.NotPanics(t, func() {
				_, err := New(repo)
				require.Error(t, err)
				assert.ErrorContains(t, err, tc.errSubstr)
			})

			// 2. IterTensorsFromRepoWithMMap rejects eagerly for both mmap and file I/O.
			for _, useMMap := range []bool{true, false} {
				assert.NotPanics(t, func() {
					var gotErr error
					for _, err := range IterTensorsFromRepoWithMMap(nil, repo, useMMap) {
						if err != nil {
							gotErr = err
							break
						}
					}
					require.Error(t, gotErr, "useMMap=%v", useMMap)
					assert.ErrorContains(t, gotErr, tc.errSubstr)
				})
			}
		})
	}
}

func TestParseHeaderAllowsValidEdgeCases(t *testing.T) {
	dir := t.TempDir()
	// Scalar F32 (4 bytes), empty tensor [0, 4] (0 bytes), and an unsupported FP8 tensor
	// whose metadata/offsets are still valid for header inspection.
	headerJSON := `{"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"empty":{"dtype":"F32","shape":[0,4],"data_offsets":[4,4]},"fp8":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[4,8]}}`
	writeSafetensorsFile(t, dir, "model.safetensors", headerJSON, make([]byte, 8))

	repo := hub.NewLocal(dir)
	m, err := New(repo)
	require.NoError(t, err)

	tr, err := m.NewTensorReader("model.safetensors")
	require.NoError(t, err)
	defer tr.Close()

	scalarT, err := tr.ReadTensor(nil, "scalar")
	require.NoError(t, err)
	assert.True(t, scalarT.Shape().IsScalar())

	emptyT, err := tr.ReadTensor(nil, "empty")
	require.NoError(t, err)
	assert.Equal(t, []int{0, 4}, emptyT.Shape().Dimensions)

	// Reading the unsupported FP8 tensor returns a clean error without panicking.
	_, err = tr.ReadTensor(nil, "fp8")
	require.Error(t, err)
	assert.ErrorContains(t, err, "not supported")
}

func TestTensorReaderDefenseInDepthAgainstMalformedMetadata(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "dummy.safetensors")
	require.NoError(t, os.WriteFile(tmpFile, make([]byte, 64), 0600))
	f, err := os.Open(tmpFile)
	require.NoError(t, err)
	defer f.Close()

	halfBitDim := 1 << (bits.UintSize / 2)
	quarterWrapDim := 1<<(bits.UintSize-2) + 4
	cases := []struct {
		name string
		meta *TensorMetadata
	}{
		{
			name: "negative_dim",
			meta: &TensorMetadata{Name: "x", Dtype: "F32", Shape: []int{-1, 4}, DataOffsets: [2]int64{0, 16}},
		},
		{
			name: "overflow_dim_product_to_zero",
			meta: &TensorMetadata{Name: "x", Dtype: "F32", Shape: []int{halfBitDim, halfBitDim}, DataOffsets: [2]int64{0, 0}},
		},
		{
			name: "overflow_byte_size_to_16",
			meta: &TensorMetadata{Name: "x", Dtype: "F32", Shape: []int{1, quarterWrapDim}, DataOffsets: [2]int64{0, 16}},
		},
		{
			name: "invalid_dtype",
			meta: &TensorMetadata{Name: "x", Dtype: "INVALID", Shape: []int{4}, DataOffsets: [2]int64{0, 16}},
		},
		{
			name: "negative_relative_offset_into_header",
			// dataOffset=32, DataOffsets=[-16, 0]: previously dataOffset+(-16)=16 >= 0 passed!
			meta: &TensorMetadata{Name: "x", Dtype: "F32", Shape: []int{4}, DataOffsets: [2]int64{-16, 0}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			readers := map[string]*TensorReader{
				"mmap": {
					mmapBuf:    make([]byte, 64),
					dataOffset: 32,
					Header:     &Header{Tensors: map[string]*TensorMetadata{"x": tc.meta}},
				},
				"file": {
					file:       f,
					fileSize:   64,
					dataOffset: 32,
					Header:     &Header{Tensors: map[string]*TensorMetadata{"x": tc.meta}},
				},
			}
			for mode, tr := range readers {
				assert.NotPanics(t, func() {
					_, err := tr.ReadTensor(nil, "x")
					assert.Error(t, err, "ReadTensor (%s) should fail", mode)
				})
				assert.NotPanics(t, func() {
					var iterErr error
					for _, err := range tr.IterTensors(nil, []string{"x"}) {
						if err != nil {
							iterErr = err
							break
						}
					}
					assert.Error(t, iterErr, "IterTensors (%s) should fail", mode)
				})
			}
		})
	}
}
