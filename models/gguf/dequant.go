package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
)

// dequantFunc dequantizes a single block of quantized data.
// src contains the raw block bytes, dst receives the float32 output values.
// len(dst) must equal the block size for the quantization type.
type dequantFunc func(src []byte, dst []float32)

// getDequantFunc returns the dequantization function for the given tensor type,
// or an error if the type is unsupported or not quantized.
func getDequantFunc(t TensorType) (dequantFunc, error) {
	switch t {
	case TensorTypeQ8_0:
		return dequantQ8_0, nil
	case TensorTypeQ4_0:
		return dequantQ4_0, nil
	case TensorTypeQ4_1:
		return dequantQ4_1, nil
	case TensorTypeQ5_0:
		return dequantQ5_0, nil
	case TensorTypeQ5_1:
		return dequantQ5_1, nil
	case TensorTypeQ2_K:
		return dequantQ2_K, nil
	case TensorTypeQ3_K:
		return dequantQ3_K, nil
	case TensorTypeQ4_K:
		return dequantQ4_K, nil
	case TensorTypeQ5_K:
		return dequantQ5_K, nil
	case TensorTypeQ6_K:
		return dequantQ6_K, nil
	default:
		return nil, fmt.Errorf("unsupported quantization type %s (%d)", t, t)
	}
}

// float16ToFloat32 converts a half-precision float (stored as uint16) to float32.
func float16ToFloat32(bits uint16) float32 {
	// IEEE 754 half-precision to single-precision conversion.
	sign := uint32(bits>>15) & 1
	exp := uint32(bits>>10) & 0x1F
	mant := uint32(bits) & 0x3FF

	var f uint32
	switch {
	case exp == 0:
		if mant == 0 {
			// Zero (positive or negative).
			f = sign << 31
		} else {
			// Subnormal: normalize.
			exp = 1
			for mant&0x400 == 0 {
				mant <<= 1
				exp--
			}
			mant &= 0x3FF
			f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
		}
	case exp == 0x1F:
		// Inf or NaN.
		f = (sign << 31) | (0xFF << 23) | (mant << 13)
	default:
		// Normal number.
		f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
	}
	return math.Float32frombits(f)
}

// dequantQ8_0 dequantizes a Q8_0 block (34 bytes → 32 float32 values).
// Format: f16 scale (2 bytes) + 32 int8 quant values.
// Math: dst[i] = scale * int8(qs[i])
func dequantQ8_0(src []byte, dst []float32) {
	d := float16ToFloat32(binary.LittleEndian.Uint16(src[0:2]))
	for j := range 32 {
		dst[j] = d * float32(int8(src[2+j]))
	}
}

// dequantQ4_0 dequantizes a Q4_0 block (18 bytes → 32 float32 values).
// Format: f16 scale (2 bytes) + 16 bytes of packed nibbles.
// Math: low nibble → first 16 values, high nibble → last 16, each offset by -8.
func dequantQ4_0(src []byte, dst []float32) {
	d := float16ToFloat32(binary.LittleEndian.Uint16(src[0:2]))
	qs := src[2:]
	for j := range 16 {
		x0 := int(qs[j]&0x0F) - 8
		x1 := int(qs[j]>>4) - 8
		dst[j] = float32(x0) * d
		dst[j+16] = float32(x1) * d
	}
}

// dequantQ4_1 dequantizes a Q4_1 block (20 bytes → 32 float32 values).
// Format: f16 scale (2) + f16 min (2) + 16 bytes of packed nibbles.
// Math: dst[i] = nibble * scale + min (asymmetric, no offset).
func dequantQ4_1(src []byte, dst []float32) {
	d := float16ToFloat32(binary.LittleEndian.Uint16(src[0:2]))
	m := float16ToFloat32(binary.LittleEndian.Uint16(src[2:4]))
	qs := src[4:]
	for j := range 16 {
		x0 := int(qs[j] & 0x0F)
		x1 := int(qs[j] >> 4)
		dst[j] = float32(x0)*d + m
		dst[j+16] = float32(x1)*d + m
	}
}

// dequantQ5_0 dequantizes a Q5_0 block (22 bytes → 32 float32 values).
// Format: f16 scale (2) + 4 bytes high bits (qh) + 16 bytes nibbles (qs).
// Math: 5-bit value = low_nibble | (high_bit << 4), offset by -16.
func dequantQ5_0(src []byte, dst []float32) {
	d := float16ToFloat32(binary.LittleEndian.Uint16(src[0:2]))
	qh := binary.LittleEndian.Uint32(src[2:6])
	qs := src[6:]
	for j := range 16 {
		xh0 := ((qh >> uint(j)) << 4) & 0x10
		xh1 := ((qh >> uint(j+12))) & 0x10
		x0 := int32((uint32(qs[j]&0x0F) | xh0)) - 16
		x1 := int32((uint32(qs[j]>>4) | xh1)) - 16
		dst[j] = float32(x0) * d
		dst[j+16] = float32(x1) * d
	}
}

// dequantQ5_1 dequantizes a Q5_1 block (24 bytes → 32 float32 values).
// Format: f16 scale (2) + f16 min (2) + 4 bytes high bits (qh) + 16 bytes nibbles (qs).
// Math: 5-bit value = low_nibble | (high_bit << 4), then * scale + min.
func dequantQ5_1(src []byte, dst []float32) {
	d := float16ToFloat32(binary.LittleEndian.Uint16(src[0:2]))
	m := float16ToFloat32(binary.LittleEndian.Uint16(src[2:4]))
	qh := binary.LittleEndian.Uint32(src[4:8])
	qs := src[8:]
	for j := range 16 {
		xh0 := ((qh >> uint(j)) << 4) & 0x10
		xh1 := ((qh >> uint(j+12))) & 0x10
		x0 := uint32(qs[j]&0x0F) | xh0
		x1 := uint32(qs[j]>>4) | xh1
		dst[j] = float32(x0)*d + m
		dst[j+16] = float32(x1)*d + m
	}
}

// dequantQ2_K dequantizes a Q2_K block (84 bytes → 256 float32 values).
// Format: 16 bytes scales + 64 bytes quants + f16 d + f16 dmin.
// Each scales byte: low 4 bits = sub-block scale, high 4 bits = sub-block min.
func dequantQ2_K(src []byte, dst []float32) {
	scales := src[0:16]
	qs := src[16:80]
	d := float16ToFloat32(binary.LittleEndian.Uint16(src[80:82]))
	dmin := float16ToFloat32(binary.LittleEndian.Uint16(src[82:84]))

	var idx int
	var is int
	for n := 0; n < 256; n += 128 {
		shift := uint(0)
		for j := 0; j < 4; j++ {
			sc := scales[is]
			is++
			dl := d * float32(sc&0xF)
			ml := dmin * float32(sc>>4)
			for l := 0; l < 16; l++ {
				dst[idx] = dl*float32((qs[n/4+l]>>shift)&3) - ml
				idx++
			}

			sc = scales[is]
			is++
			dl = d * float32(sc&0xF)
			ml = dmin * float32(sc>>4)
			for l := 0; l < 16; l++ {
				dst[idx] = dl*float32((qs[n/4+16+l]>>shift)&3) - ml
				idx++
			}

			shift += 2
		}
	}
}

// dequantQ3_K dequantizes a Q3_K block (110 bytes → 256 float32 values).
// Format: 32 bytes hmask + 64 bytes qs + 12 bytes scales + f16 d.
// Each value is 3 bits: 2 from qs + 1 from hmask. Scales are 6-bit, biased by 32.
func dequantQ3_K(src []byte, dst []float32) {
	hmask := src[0:32]
	qs := src[32:96]
	scaleBytes := src[96:108]
	dAll := float16ToFloat32(binary.LittleEndian.Uint16(src[108:110]))

	// Unpack 6-bit scales from 12-byte packed representation into 16 int8 values.
	var aux [4]uint32
	aux[0] = binary.LittleEndian.Uint32(scaleBytes[0:4])
	aux[1] = binary.LittleEndian.Uint32(scaleBytes[4:8])
	aux[2] = binary.LittleEndian.Uint32(scaleBytes[8:12])

	kmask1 := uint32(0x03030303)
	kmask2 := uint32(0x0f0f0f0f)

	tmp := aux[2]
	aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
	aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
	aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
	aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)

	// Read unpacked scales as signed int8.
	var scales [16]int8
	for i := 0; i < 4; i++ {
		scales[i*4+0] = int8(aux[i])
		scales[i*4+1] = int8(aux[i] >> 8)
		scales[i*4+2] = int8(aux[i] >> 16)
		scales[i*4+3] = int8(aux[i] >> 24)
	}

	var idx int
	var is int
	var m uint8 = 1
	var qOff int
	for range 2 { // Two groups of 128 values.
		shift := uint(0)
		for range 4 { // Four sub-groups per group, each extracting 2 bits at a different shift.
			dl := dAll * float32(scales[is]-32)
			is++
			for l := 0; l < 16; l++ {
				q := int8((qs[qOff+l] >> shift) & 3)
				if hmask[l]&m == 0 {
					q -= 4
				}
				dst[idx] = dl * float32(q)
				idx++
			}

			dl = dAll * float32(scales[is]-32)
			is++
			for l := 0; l < 16; l++ {
				q := int8((qs[qOff+16+l] >> shift) & 3)
				if hmask[16+l]&m == 0 {
					q -= 4
				}
				dst[idx] = dl * float32(q)
				idx++
			}

			shift += 2
			m <<= 1
		}
		qOff += 32
	}
}

// getScaleMinK4 extracts a 6-bit scale and min value from the Q4_K/Q5_K
// 12-byte packed scales array. j is the sub-block index (0..7).
func getScaleMinK4(j int, scales []byte) (sc, m uint8) {
	if j < 4 {
		sc = scales[j] & 63
		m = scales[j+4] & 63
	} else {
		sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
		m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
	}
	return
}

// dequantQ4_K dequantizes a Q4_K block (144 bytes → 256 float32 values).
// Format: f16 d (2) + f16 dmin (2) + 12 bytes scales + 128 bytes nibbles.
// 8 sub-blocks of 32 values, each with 6-bit packed scale and min.
func dequantQ4_K(src []byte, dst []float32) {
	d := float16ToFloat32(binary.LittleEndian.Uint16(src[0:2]))
	dmin := float16ToFloat32(binary.LittleEndian.Uint16(src[2:4]))
	scales := src[4:16]
	qs := src[16:]

	var idx int
	var is int
	for j := 0; j < 256; j += 64 {
		sc1, m1 := getScaleMinK4(is, scales)
		d1 := d * float32(sc1)
		min1 := dmin * float32(m1)

		sc2, m2 := getScaleMinK4(is+1, scales)
		d2 := d * float32(sc2)
		min2 := dmin * float32(m2)

		qoff := j / 2
		for l := 0; l < 32; l++ {
			dst[idx] = d1*float32(qs[qoff+l]&0xF) - min1
			idx++
		}
		for l := 0; l < 32; l++ {
			dst[idx] = d2*float32(qs[qoff+l]>>4) - min2
			idx++
		}
		is += 2
	}
}

// dequantQ5_K dequantizes a Q5_K block (176 bytes → 256 float32 values).
// Format: f16 d (2) + f16 dmin (2) + 12 bytes scales + 32 bytes qh + 128 bytes qs.
// Same as Q4_K but with a 5th bit per value stored in qh.
func dequantQ5_K(src []byte, dst []float32) {
	d := float16ToFloat32(binary.LittleEndian.Uint16(src[0:2]))
	dmin := float16ToFloat32(binary.LittleEndian.Uint16(src[2:4]))
	scales := src[4:16]
	qh := src[16:48]
	qs := src[48:]

	var idx int
	var is int
	var u1, u2 uint8 = 1, 2
	var qlOff int
	for range 4 { // Four groups of 64 values.
		sc1, m1 := getScaleMinK4(is, scales)
		d1 := d * float32(sc1)
		min1 := dmin * float32(m1)

		sc2, m2 := getScaleMinK4(is+1, scales)
		d2 := d * float32(sc2)
		min2 := dmin * float32(m2)

		// qh is 32 bytes; we always index [0..31] but use rotating bitmasks u1/u2.
		for l := 0; l < 32; l++ {
			hbit := uint8(0)
			if qh[l]&u1 != 0 {
				hbit = 16
			}
			dst[idx] = d1*float32(uint8(qs[qlOff+l]&0xF)+hbit) - min1
			idx++
		}
		for l := 0; l < 32; l++ {
			hbit := uint8(0)
			if qh[l]&u2 != 0 {
				hbit = 16
			}
			dst[idx] = d2*float32(uint8(qs[qlOff+l]>>4)+hbit) - min2
			idx++
		}
		qlOff += 32
		is += 2
		u1 <<= 2
		u2 <<= 2
	}
}

// dequantQ6_K dequantizes a Q6_K block (210 bytes → 256 float32 values).
// Format: 128 bytes ql + 64 bytes qh + 16 bytes scales + f16 d.
// 6-bit values: 4 bits from ql + 2 bits from qh, centered by -32.
func dequantQ6_K(src []byte, dst []float32) {
	ql := src[0:128]
	qh := src[128:192]
	sc := src[192:208]
	d := float16ToFloat32(binary.LittleEndian.Uint16(src[208:210]))

	var idx int
	var qlOff, qhOff, scOff int
	for n := 0; n < 256; n += 128 {
		for l := 0; l < 32; l++ {
			is := l / 16
			q1 := int8((uint8(ql[qlOff+l])&0xF)|((qh[qhOff+l]>>0)&3)<<4) - 32
			q2 := int8((uint8(ql[qlOff+l+32])&0xF)|((qh[qhOff+l]>>2)&3)<<4) - 32
			q3 := int8((uint8(ql[qlOff+l])>>4)|((qh[qhOff+l]>>4)&3)<<4) - 32
			q4 := int8((uint8(ql[qlOff+l+32])>>4)|((qh[qhOff+l]>>6)&3)<<4) - 32
			dst[idx+l] = d * float32(int8(sc[scOff+is])) * float32(q1)
			dst[idx+l+32] = d * float32(int8(sc[scOff+is+2])) * float32(q2)
			dst[idx+l+64] = d * float32(int8(sc[scOff+is+4])) * float32(q3)
			dst[idx+l+96] = d * float32(int8(sc[scOff+is+6])) * float32(q4)
		}
		idx += 128
		qlOff += 64
		qhOff += 32
		scOff += 8
	}
}
