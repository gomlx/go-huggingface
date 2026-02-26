package gguf

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// float32ToFloat16Bits converts a float32 to its IEEE 754 half-precision representation.
// Used only in tests to construct known test vectors.
func float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := (bits >> 16) & 0x8000
	exp := int((bits>>23)&0xFF) - 127 + 15
	mant := bits & 0x7FFFFF

	switch {
	case exp <= 0:
		return uint16(sign)
	case exp >= 31:
		return uint16(sign | 0x7C00) // Inf
	default:
		return uint16(sign | uint32(exp)<<10 | (mant >> 13))
	}
}

func leUint16(v uint16) []byte {
	b := make([]byte, 2)
	binary.LittleEndian.PutUint16(b, v)
	return b
}

func TestFloat16ToFloat32(t *testing.T) {
	tests := []struct {
		name string
		bits uint16
		want float32
	}{
		{"positive zero", 0x0000, 0.0},
		{"negative zero", 0x8000, float32(math.Copysign(0, -1))},
		{"one", 0x3C00, 1.0},
		{"negative one", 0xBC00, -1.0},
		{"half", 0x3800, 0.5},
		{"two", 0x4000, 2.0},
		{"inf", 0x7C00, float32(math.Inf(1))},
		{"neg inf", 0xFC00, float32(math.Inf(-1))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := float16ToFloat32(tt.bits)
			if math.IsInf(float64(tt.want), 0) {
				assert.True(t, math.IsInf(float64(got), int(math.Copysign(1, float64(tt.want)))))
			} else {
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

func TestFloat16Roundtrip(t *testing.T) {
	// Verify our test helper produces correct fp16 bits that round-trip through dequant.
	values := []float32{0.0, 1.0, -1.0, 0.5, 2.0, 0.25, 100.0}
	for _, v := range values {
		bits := float32ToFloat16Bits(v)
		got := float16ToFloat32(bits)
		assert.InDelta(t, v, got, float64(math.Abs(float64(v))*0.001+1e-6),
			"roundtrip failed for %v (bits=0x%04X, got=%v)", v, bits, got)
	}
}

func TestDequantQ8_0(t *testing.T) {
	// Q8_0 block: 2 bytes f16 scale + 32 bytes int8 values = 34 bytes.
	// scale = 2.0, values = [0, 1, 2, ..., 31]
	// Expected: [0.0, 2.0, 4.0, ..., 62.0]
	src := make([]byte, 34)
	binary.LittleEndian.PutUint16(src[0:2], float32ToFloat16Bits(2.0))
	for i := range 32 {
		src[2+i] = byte(int8(i))
	}

	dst := make([]float32, 32)
	dequantQ8_0(src, dst)

	for i := range 32 {
		assert.InDelta(t, float32(i)*2.0, dst[i], 0.01, "Q8_0 index %d", i)
	}
}

func TestDequantQ8_0_Negative(t *testing.T) {
	// Test with negative int8 values.
	// scale = 1.0, values = [-128, -1, 0, 1, 127, ...]
	src := make([]byte, 34)
	binary.LittleEndian.PutUint16(src[0:2], float32ToFloat16Bits(1.0))
	src[2] = 0x80 // int8(-128)
	src[3] = 0xFF // int8(-1)
	src[4] = 0x00 // int8(0)
	src[5] = 0x01 // int8(1)
	src[6] = 0x7F // int8(127)

	dst := make([]float32, 32)
	dequantQ8_0(src, dst)

	assert.InDelta(t, -128.0, dst[0], 0.01)
	assert.InDelta(t, -1.0, dst[1], 0.01)
	assert.InDelta(t, 0.0, dst[2], 0.01)
	assert.InDelta(t, 1.0, dst[3], 0.01)
	assert.InDelta(t, 127.0, dst[4], 0.01)
}

func TestDequantQ4_0(t *testing.T) {
	// Q4_0 block: 2 bytes f16 scale + 16 bytes nibbles = 18 bytes.
	// scale = 0.5
	// Each byte encodes two 4-bit values, each offset by -8.
	// Byte[0] = 0x80 → low nibble = 0, high nibble = 8
	// Offset: low → 0 - 8 = -8, high → 8 - 8 = 0
	// Result: dst[0] = -8 * 0.5 = -4.0, dst[16] = 0 * 0.5 = 0.0
	src := make([]byte, 18)
	binary.LittleEndian.PutUint16(src[0:2], float32ToFloat16Bits(0.5))
	src[2] = 0x80 // low=0, high=8

	dst := make([]float32, 32)
	dequantQ4_0(src, dst)

	assert.InDelta(t, -4.0, dst[0], 0.01, "Q4_0 low nibble")
	assert.InDelta(t, 0.0, dst[16], 0.01, "Q4_0 high nibble")

	// Byte[1] = 0xFF → low = 15, high = 15
	// Offset: low → 15 - 8 = 7, high → 15 - 8 = 7
	// Result: dst[1] = 7 * 0.5 = 3.5, dst[17] = 7 * 0.5 = 3.5
	src[3] = 0xFF
	dequantQ4_0(src, dst)
	assert.InDelta(t, 3.5, dst[1], 0.01, "Q4_0 low nibble 0xF")
	assert.InDelta(t, 3.5, dst[17], 0.01, "Q4_0 high nibble 0xF")
}

func TestDequantQ4_1(t *testing.T) {
	// Q4_1 block: 2 bytes f16 scale + 2 bytes f16 min + 16 bytes nibbles = 20 bytes.
	// scale = 1.0, min = 2.0
	// Byte[0] = 0x31 → low = 1, high = 3
	// Result: dst[0] = 1 * 1.0 + 2.0 = 3.0, dst[16] = 3 * 1.0 + 2.0 = 5.0
	src := make([]byte, 20)
	binary.LittleEndian.PutUint16(src[0:2], float32ToFloat16Bits(1.0))
	binary.LittleEndian.PutUint16(src[2:4], float32ToFloat16Bits(2.0))
	src[4] = 0x31

	dst := make([]float32, 32)
	dequantQ4_1(src, dst)

	assert.InDelta(t, 3.0, dst[0], 0.01, "Q4_1 low nibble")
	assert.InDelta(t, 5.0, dst[16], 0.01, "Q4_1 high nibble")
}

func TestDequantQ5_0(t *testing.T) {
	// Q5_0 block: 2 bytes f16 scale + 4 bytes qh + 16 bytes qs = 22 bytes.
	// scale = 1.0
	// All zero qs and qh: 5-bit value = 0, offset by -16 → -16.
	src := make([]byte, 22)
	binary.LittleEndian.PutUint16(src[0:2], float32ToFloat16Bits(1.0))

	dst := make([]float32, 32)
	dequantQ5_0(src, dst)

	for i := range 32 {
		assert.InDelta(t, -16.0, dst[i], 0.01, "Q5_0 zero index %d", i)
	}

	// Set qh bit 0 → element 0 gets high bit = 16, so 5-bit value = 0 | 16 = 16, offset: 16-16=0.
	binary.LittleEndian.PutUint32(src[2:6], 1) // bit 0 set
	dequantQ5_0(src, dst)
	assert.InDelta(t, 0.0, dst[0], 0.01, "Q5_0 with high bit set")
	assert.InDelta(t, -16.0, dst[1], 0.01, "Q5_0 without high bit")
}

func TestDequantQ5_1(t *testing.T) {
	// Q5_1 block: 2 bytes f16 scale + 2 bytes f16 min + 4 bytes qh + 16 bytes qs = 24 bytes.
	// scale = 1.0, min = 0.0, all zero → 5-bit value = 0, result = 0 * 1.0 + 0.0 = 0.0
	src := make([]byte, 24)
	binary.LittleEndian.PutUint16(src[0:2], float32ToFloat16Bits(1.0))
	binary.LittleEndian.PutUint16(src[2:4], float32ToFloat16Bits(0.0))

	dst := make([]float32, 32)
	dequantQ5_1(src, dst)

	for i := range 32 {
		assert.InDelta(t, 0.0, dst[i], 0.01, "Q5_1 zero index %d", i)
	}

	// With min = 5.0 → all values become 5.0.
	binary.LittleEndian.PutUint16(src[2:4], float32ToFloat16Bits(5.0))
	dequantQ5_1(src, dst)
	for i := range 32 {
		assert.InDelta(t, 5.0, dst[i], 0.01, "Q5_1 min=5 index %d", i)
	}
}

func TestDequantQ2_K(t *testing.T) {
	// Q2_K: 16 bytes scales + 64 bytes qs + 2 bytes d + 2 bytes dmin = 84 bytes.
	// Set d = 1.0, dmin = 0.0, all scales low nibble = 1, high nibble = 0.
	// All qs = 0 → 2-bit values are 0.
	// dl = 1.0 * 1 = 1.0, ml = 0.0 * 0 = 0.0.
	// dst[i] = 1.0 * 0 - 0.0 = 0.0
	src := make([]byte, 84)
	for i := 0; i < 16; i++ {
		src[i] = 0x01 // scale low nibble = 1, min high nibble = 0
	}
	binary.LittleEndian.PutUint16(src[80:82], float32ToFloat16Bits(1.0))
	binary.LittleEndian.PutUint16(src[82:84], float32ToFloat16Bits(0.0))

	dst := make([]float32, 256)
	dequantQ2_K(src, dst)

	for i := range 256 {
		assert.InDelta(t, 0.0, dst[i], 0.01, "Q2_K zero index %d", i)
	}
}

func TestDequantQ3_K(t *testing.T) {
	// Q3_K: 32 bytes hmask + 64 bytes qs + 12 bytes scales + 2 bytes d = 110 bytes.
	// All zeros: d = 0 → everything is 0.
	src := make([]byte, 110)

	dst := make([]float32, 256)
	dequantQ3_K(src, dst)

	for i := range 256 {
		assert.InDelta(t, 0.0, dst[i], 0.01, "Q3_K zero index %d", i)
	}
}

func TestDequantQ4_K(t *testing.T) {
	// Q4_K: 2 bytes d + 2 bytes dmin + 12 bytes scales + 128 bytes qs = 144 bytes.
	// d = 1.0, dmin = 0.0, all scales = 1 (scale=1, min=0 for sub-blocks 0..3).
	// All qs = 0 → 4-bit values are 0.
	// dst[i] = 1.0 * 1 * 0 - 0.0 * 0 = 0.0
	src := make([]byte, 144)
	binary.LittleEndian.PutUint16(src[0:2], float32ToFloat16Bits(1.0))
	binary.LittleEndian.PutUint16(src[2:4], float32ToFloat16Bits(0.0))
	// scales[0..3] = 1 → getScaleMinK4(0..3) returns sc=1, m=0
	// scales[4..7] = 0 → min=0 for sub-blocks 0..3
	for i := 0; i < 4; i++ {
		src[4+i] = 1
	}

	dst := make([]float32, 256)
	dequantQ4_K(src, dst)

	// First 128 values (sub-blocks 0..3) should be 0.
	for i := 0; i < 128; i++ {
		assert.InDelta(t, 0.0, dst[i], 0.01, "Q4_K zero index %d", i)
	}
}

func TestDequantQ4_K_NonZero(t *testing.T) {
	// More interesting test: d = 2.0, dmin = 1.0, first sub-block scale=3, min=2.
	// qs[0] = 0x54 → low nibble = 4, high nibble = 5
	// Sub-block 0: d1 = 2.0 * 3 = 6.0, min1 = 1.0 * 2 = 2.0
	// Sub-block 1: d2 = 2.0 * scale1, min2 = 1.0 * min1
	// dst[0] = 6.0 * 4 - 2.0 = 22.0
	// dst[32] = d2 * 5 - min2
	src := make([]byte, 144)
	binary.LittleEndian.PutUint16(src[0:2], float32ToFloat16Bits(2.0))
	binary.LittleEndian.PutUint16(src[2:4], float32ToFloat16Bits(1.0))
	// For sub-blocks 0-3 (j < 4): sc = scales[j] & 63, m = scales[j+4] & 63
	src[4+0] = 3 // scale for sub-block 0 = 3
	src[4+4] = 2 // min for sub-block 0 = 2
	src[4+1] = 5 // scale for sub-block 1 = 5
	src[4+5] = 1 // min for sub-block 1 = 1

	// qs[0] = 0x54: low = 4, high = 5
	src[16] = 0x54

	dst := make([]float32, 256)
	dequantQ4_K(src, dst)

	// dst[0] = d1 * low - min1 = 2.0*3*4 - 1.0*2 = 24 - 2 = 22
	assert.InDelta(t, 22.0, dst[0], 0.1, "Q4_K non-zero low nibble")
	// dst[32] = d2 * high - min2 = 2.0*5*5 - 1.0*1 = 50 - 1 = 49
	assert.InDelta(t, 49.0, dst[32], 0.1, "Q4_K non-zero high nibble")
}

func TestDequantQ6_K(t *testing.T) {
	// Q6_K: 128 bytes ql + 64 bytes qh + 16 bytes scales + 2 bytes d = 210 bytes.
	// All zeros → d = 0, so all outputs are 0.
	src := make([]byte, 210)

	dst := make([]float32, 256)
	dequantQ6_K(src, dst)

	for i := range 256 {
		assert.InDelta(t, 0.0, dst[i], 0.01, "Q6_K zero index %d", i)
	}
}

func TestGetDequantFunc(t *testing.T) {
	// Supported quantized types should return a function.
	supported := []TensorType{
		TensorTypeQ8_0, TensorTypeQ4_0, TensorTypeQ4_1,
		TensorTypeQ5_0, TensorTypeQ5_1,
		TensorTypeQ2_K, TensorTypeQ3_K, TensorTypeQ4_K,
		TensorTypeQ5_K, TensorTypeQ6_K,
	}
	for _, tt := range supported {
		fn, err := getDequantFunc(tt)
		require.NoError(t, err, "getDequantFunc(%s)", tt)
		assert.NotNil(t, fn, "getDequantFunc(%s)", tt)
	}

	// Non-quantized types should error.
	_, err := getDequantFunc(TensorTypeF32)
	assert.Error(t, err)

	_, err = getDequantFunc(TensorTypeF16)
	assert.Error(t, err)
}

func TestDequantQ5_K(t *testing.T) {
	// Q5_K: 2 bytes d + 2 bytes dmin + 12 bytes scales + 32 bytes qh + 128 bytes qs = 176 bytes.
	// All zeros → all outputs are 0 (d = 0).
	src := make([]byte, 176)

	dst := make([]float32, 256)
	dequantQ5_K(src, dst)

	for i := range 256 {
		assert.InDelta(t, 0.0, dst[i], 0.01, "Q5_K zero index %d", i)
	}
}

func TestDequantQ5_K_WithData(t *testing.T) {
	// Verify Q5_K with d = 1.0, dmin = 0.0, scale = 1 for sub-block 0, all qs/qh = 0.
	// 5-bit value = 0 (no high bit) → dst = 1.0 * 1 * 0 - 0 = 0.
	src := make([]byte, 176)
	binary.LittleEndian.PutUint16(src[0:2], float32ToFloat16Bits(1.0))
	binary.LittleEndian.PutUint16(src[2:4], float32ToFloat16Bits(0.0))
	src[4+0] = 1 // scale for sub-block 0

	dst := make([]float32, 256)
	dequantQ5_K(src, dst)

	for i := 0; i < 32; i++ {
		assert.InDelta(t, 0.0, dst[i], 0.01, "Q5_K data index %d", i)
	}

	// Set qs[0] = 0x03 → low nibble = 3 for element 0.
	// With qh all zero, 5-bit value = 3.
	// dst[0] = 1.0 * 1 * 3 - 0 = 3.0
	src[48] = 0x03
	dequantQ5_K(src, dst)
	assert.InDelta(t, 3.0, dst[0], 0.01, "Q5_K non-zero qs")
}
