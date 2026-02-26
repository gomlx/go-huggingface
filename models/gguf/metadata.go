package gguf

// ggufValueType represents the type tag of a GGUF metadata value in the binary format.
type ggufValueType uint32

const (
	valueTypeUint8   ggufValueType = 0
	valueTypeInt8    ggufValueType = 1
	valueTypeUint16  ggufValueType = 2
	valueTypeInt16   ggufValueType = 3
	valueTypeUint32  ggufValueType = 4
	valueTypeInt32   ggufValueType = 5
	valueTypeFloat32 ggufValueType = 6
	valueTypeBool    ggufValueType = 7
	valueTypeString  ggufValueType = 8
	valueTypeArray   ggufValueType = 9
	valueTypeUint64  ggufValueType = 10
	valueTypeInt64   ggufValueType = 11
	valueTypeFloat64 ggufValueType = 12
)

// KeyValue represents a metadata key-value pair from a GGUF file.
type KeyValue struct {
	Key string
	Value
}

// Value wraps a GGUF metadata value with typed accessors.
// Accessors return zero values when the underlying type doesn't match,
// rather than returning errors.
type Value struct {
	data any
}

// Raw returns the underlying value without type conversion.
func (v Value) Raw() any {
	return v.data
}

// String returns the value as a string, or "" if it is not a string.
func (v Value) String() string {
	s, _ := v.data.(string)
	return s
}

// Strings returns the value as a string slice, or nil if it is not one.
func (v Value) Strings() []string {
	if s, ok := v.data.([]string); ok {
		return s
	}
	return nil
}

// Int returns the value as an int64. Works for any signed or unsigned integer type.
// Returns 0 if the value is not an integer.
func (v Value) Int() int64 {
	switch n := v.data.(type) {
	case int8:
		return int64(n)
	case int16:
		return int64(n)
	case int32:
		return int64(n)
	case int64:
		return n
	case uint8:
		return int64(n)
	case uint16:
		return int64(n)
	case uint32:
		return int64(n)
	case uint64:
		return int64(n)
	default:
		return 0
	}
}

// Uint returns the value as a uint64. Works for any unsigned or signed integer type.
// Returns 0 if the value is not an integer.
func (v Value) Uint() uint64 {
	switch n := v.data.(type) {
	case uint8:
		return uint64(n)
	case uint16:
		return uint64(n)
	case uint32:
		return uint64(n)
	case uint64:
		return n
	case int8:
		return uint64(n)
	case int16:
		return uint64(n)
	case int32:
		return uint64(n)
	case int64:
		return uint64(n)
	default:
		return 0
	}
}

// Float returns the value as a float64. Works for float32 and float64.
// Returns 0 if the value is not a float.
func (v Value) Float() float64 {
	switch n := v.data.(type) {
	case float32:
		return float64(n)
	case float64:
		return n
	default:
		return 0
	}
}

// Floats returns the value as a float64 slice, or nil if it is not one.
func (v Value) Floats() []float64 {
	switch s := v.data.(type) {
	case []float64:
		return s
	case []float32:
		out := make([]float64, len(s))
		for i, f := range s {
			out[i] = float64(f)
		}
		return out
	default:
		return nil
	}
}

// Bool returns the value as a bool, or false if it is not a bool.
func (v Value) Bool() bool {
	b, _ := v.data.(bool)
	return b
}

// Ints returns the value as an int64 slice, or nil if it is not an integer array.
func (v Value) Ints() []int64 {
	switch s := v.data.(type) {
	case []int64:
		return s
	case []int32:
		out := make([]int64, len(s))
		for i, n := range s {
			out[i] = int64(n)
		}
		return out
	case []int16:
		out := make([]int64, len(s))
		for i, n := range s {
			out[i] = int64(n)
		}
		return out
	case []int8:
		out := make([]int64, len(s))
		for i, n := range s {
			out[i] = int64(n)
		}
		return out
	case []uint64:
		out := make([]int64, len(s))
		for i, n := range s {
			out[i] = int64(n)
		}
		return out
	case []uint32:
		out := make([]int64, len(s))
		for i, n := range s {
			out[i] = int64(n)
		}
		return out
	case []uint16:
		out := make([]int64, len(s))
		for i, n := range s {
			out[i] = int64(n)
		}
		return out
	case []uint8:
		out := make([]int64, len(s))
		for i, n := range s {
			out[i] = int64(n)
		}
		return out
	default:
		return nil
	}
}

// Uints returns the value as a uint64 slice, or nil if it is not an integer array.
func (v Value) Uints() []uint64 {
	switch s := v.data.(type) {
	case []uint64:
		return s
	case []uint32:
		out := make([]uint64, len(s))
		for i, n := range s {
			out[i] = uint64(n)
		}
		return out
	case []uint16:
		out := make([]uint64, len(s))
		for i, n := range s {
			out[i] = uint64(n)
		}
		return out
	case []uint8:
		out := make([]uint64, len(s))
		for i, n := range s {
			out[i] = uint64(n)
		}
		return out
	case []int64:
		out := make([]uint64, len(s))
		for i, n := range s {
			out[i] = uint64(n)
		}
		return out
	case []int32:
		out := make([]uint64, len(s))
		for i, n := range s {
			out[i] = uint64(n)
		}
		return out
	default:
		return nil
	}
}
