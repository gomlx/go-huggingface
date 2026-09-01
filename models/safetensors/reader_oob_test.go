package safetensors

import (
	"strings"
	"testing"
)

// A safetensors file header is untrusted input, and each tensor's data_offsets
// come straight from it. ReadTensor sliced mr.mmapBuf[offset:end] using those
// offsets after only checking that (end-start) matched the shape's byte size,
// never that the offsets fell within the mapped data. A crafted header could
// therefore point outside the buffer (or use a negative start) and panic the
// reader with a slice-out-of-range. These tests ensure such offsets are
// rejected with an error instead.
func TestReadTensorRejectsOutOfRangeOffsets(t *testing.T) {
	newReader := func(off [2]int64) *TensorReader {
		return &TensorReader{
			mmapBuf:    make([]byte, 16),
			dataOffset: 0,
			Header: &Header{Tensors: map[string]*TensorMetadata{
				"evil": {Name: "evil", Dtype: "F32", Shape: []int{4}, DataOffsets: off},
			}},
		}
	}
	cases := map[string][2]int64{
		"beyond-buffer": {1 << 40, 1<<40 + 16},
		"negative":      {-16, 0},
	}
	for name, off := range cases {
		t.Run(name, func(t *testing.T) {
			tr := newReader(off)
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("ReadTensor panicked on %s offsets: %v", name, r)
				}
			}()
			_, err := tr.ReadTensor(nil, "evil")
			if err == nil || !strings.Contains(err.Error(), "out of range") {
				t.Fatalf("expected out-of-range error, got %v", err)
			}
		})
	}
}
