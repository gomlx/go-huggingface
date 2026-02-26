package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

const (
	ggufMagic          = "GGUF"
	defaultAlignment   = 32
	minSupportedVersion = 2
)

// File represents a parsed GGUF file. Create one with Open.
type File struct {
	// Version is the GGUF format version (2 or 3).
	Version uint32
	// Alignment is the byte alignment for tensor data (default 32).
	Alignment uint64
	// KeyValues holds all metadata key-value pairs from the file header.
	KeyValues []KeyValue
	// TensorInfos holds parsed information about every tensor in the file.
	TensorInfos []TensorInfo

	kvByKey      map[string]*KeyValue
	tensorByName map[string]*TensorInfo
	path         string
	dataOffset   int64
}

// Open opens and parses a GGUF file, reading all metadata and tensor info.
// The returned File can be used to look up metadata and read tensor data.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("gguf: open %s: %w", path, err)
	}
	defer f.Close()

	file := &File{path: path}
	r := &countingReader{r: f}

	// Read and validate magic number.
	var magic [4]byte
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("gguf: read magic: %w", err)
	}
	if string(magic[:]) != ggufMagic {
		return nil, fmt.Errorf("gguf: invalid magic %q, expected %q", magic[:], ggufMagic)
	}

	// Read version.
	if err := binary.Read(r, binary.LittleEndian, &file.Version); err != nil {
		return nil, fmt.Errorf("gguf: read version: %w", err)
	}
	if file.Version < minSupportedVersion {
		return nil, fmt.Errorf("gguf: unsupported version %d (minimum %d)", file.Version, minSupportedVersion)
	}

	// Read counts.
	var tensorCount, kvCount uint64
	if err := binary.Read(r, binary.LittleEndian, &tensorCount); err != nil {
		return nil, fmt.Errorf("gguf: read tensor count: %w", err)
	}
	if err := binary.Read(r, binary.LittleEndian, &kvCount); err != nil {
		return nil, fmt.Errorf("gguf: read kv count: %w", err)
	}

	// Read all key-value pairs.
	file.KeyValues = make([]KeyValue, 0, kvCount)
	for range kvCount {
		kv, err := readKeyValue(r)
		if err != nil {
			return nil, fmt.Errorf("gguf: read kv pair %d/%d: %w", len(file.KeyValues), kvCount, err)
		}
		file.KeyValues = append(file.KeyValues, kv)
	}

	// Read all tensor info entries.
	file.TensorInfos = make([]TensorInfo, 0, tensorCount)
	for range tensorCount {
		ti, err := readTensorInfo(r)
		if err != nil {
			return nil, fmt.Errorf("gguf: read tensor info %d/%d: %w", len(file.TensorInfos), tensorCount, err)
		}
		file.TensorInfos = append(file.TensorInfos, ti)
	}

	// Compute aligned data offset.
	file.Alignment = defaultAlignment
	if kv, ok := file.getKV("general.alignment"); ok {
		if a := kv.Uint(); a > 0 {
			file.Alignment = a
		}
	}
	offset := uint64(r.n)
	alignment := file.Alignment
	file.dataOffset = int64(offset + (alignment-offset%alignment)%alignment)

	// Build indexes.
	file.kvByKey = make(map[string]*KeyValue, len(file.KeyValues))
	for i := range file.KeyValues {
		file.kvByKey[file.KeyValues[i].Key] = &file.KeyValues[i]
	}
	file.tensorByName = make(map[string]*TensorInfo, len(file.TensorInfos))
	for i := range file.TensorInfos {
		file.tensorByName[file.TensorInfos[i].Name] = &file.TensorInfos[i]
	}

	return file, nil
}

// Path returns the local file path of the GGUF file.
func (f *File) Path() string {
	return f.path
}

// DataOffset returns the byte offset where tensor data begins in the file.
func (f *File) DataOffset() int64 {
	return f.dataOffset
}

// GetKeyValue looks up a metadata key-value pair by its key.
func (f *File) GetKeyValue(key string) (KeyValue, bool) {
	return f.getKV(key)
}

func (f *File) getKV(key string) (KeyValue, bool) {
	kv, ok := f.kvByKey[key]
	if !ok {
		return KeyValue{}, false
	}
	return *kv, true
}

// GetTensorInfo looks up a tensor by name.
func (f *File) GetTensorInfo(name string) (TensorInfo, bool) {
	ti, ok := f.tensorByName[name]
	if !ok {
		return TensorInfo{}, false
	}
	return *ti, true
}

// Architecture returns the model architecture string (e.g., "llama", "gemma"),
// or "" if the metadata key "general.architecture" is not present.
func (f *File) Architecture() string {
	kv, ok := f.getKV("general.architecture")
	if !ok {
		return ""
	}
	return kv.String()
}

// ListTensorNames returns the names of all tensors in the file.
func (f *File) ListTensorNames() []string {
	names := make([]string, len(f.TensorInfos))
	for i, ti := range f.TensorInfos {
		names[i] = ti.Name
	}
	return names
}

// Binary reading helpers.

// countingReader wraps an io.Reader and counts bytes read.
type countingReader struct {
	r io.Reader
	n int64
}

func (cr *countingReader) Read(p []byte) (int, error) {
	n, err := cr.r.Read(p)
	cr.n += int64(n)
	return n, err
}

// readString reads a GGUF string: uint64 length prefix followed by that many bytes.
func readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", fmt.Errorf("read string length: %w", err)
	}
	if length > 1<<20 { // 1MB sanity check for a single string.
		return "", fmt.Errorf("string length %d exceeds 1MB limit", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", fmt.Errorf("read string data: %w", err)
	}
	return string(buf), nil
}

// readKeyValue reads a single GGUF key-value pair from the stream.
func readKeyValue(r io.Reader) (KeyValue, error) {
	key, err := readString(r)
	if err != nil {
		return KeyValue{}, fmt.Errorf("read key: %w", err)
	}

	var typeTag uint32
	if err := binary.Read(r, binary.LittleEndian, &typeTag); err != nil {
		return KeyValue{}, fmt.Errorf("read value type for %q: %w", key, err)
	}

	val, err := readValue(r, ggufValueType(typeTag))
	if err != nil {
		return KeyValue{}, fmt.Errorf("read value for %q (type %d): %w", key, typeTag, err)
	}

	return KeyValue{Key: key, Value: val}, nil
}

// readValue reads a GGUF value of the given type.
func readValue(r io.Reader, vtype ggufValueType) (Value, error) {
	switch vtype {
	case valueTypeUint8:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return Value{data: v}, err
	case valueTypeInt8:
		var v int8
		err := binary.Read(r, binary.LittleEndian, &v)
		return Value{data: v}, err
	case valueTypeUint16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return Value{data: v}, err
	case valueTypeInt16:
		var v int16
		err := binary.Read(r, binary.LittleEndian, &v)
		return Value{data: v}, err
	case valueTypeUint32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return Value{data: v}, err
	case valueTypeInt32:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return Value{data: v}, err
	case valueTypeFloat32:
		var v float32
		err := binary.Read(r, binary.LittleEndian, &v)
		return Value{data: v}, err
	case valueTypeBool:
		var v uint8
		if err := binary.Read(r, binary.LittleEndian, &v); err != nil {
			return Value{}, err
		}
		return Value{data: v != 0}, nil
	case valueTypeString:
		s, err := readString(r)
		return Value{data: s}, err
	case valueTypeUint64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return Value{data: v}, err
	case valueTypeInt64:
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return Value{data: v}, err
	case valueTypeFloat64:
		var v float64
		err := binary.Read(r, binary.LittleEndian, &v)
		return Value{data: v}, err
	case valueTypeArray:
		return readArray(r)
	default:
		return Value{}, fmt.Errorf("unknown value type %d", vtype)
	}
}

// readArray reads a GGUF typed array: uint32 element type, uint64 count, then elements.
func readArray(r io.Reader) (Value, error) {
	var elemType uint32
	if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
		return Value{}, fmt.Errorf("read array element type: %w", err)
	}
	var count uint64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return Value{}, fmt.Errorf("read array count: %w", err)
	}

	switch ggufValueType(elemType) {
	case valueTypeUint8:
		return readArrayOf[uint8](r, count)
	case valueTypeInt8:
		return readArrayOf[int8](r, count)
	case valueTypeUint16:
		return readArrayOf[uint16](r, count)
	case valueTypeInt16:
		return readArrayOf[int16](r, count)
	case valueTypeUint32:
		return readArrayOf[uint32](r, count)
	case valueTypeInt32:
		return readArrayOf[int32](r, count)
	case valueTypeFloat32:
		return readArrayOf[float32](r, count)
	case valueTypeUint64:
		return readArrayOf[uint64](r, count)
	case valueTypeInt64:
		return readArrayOf[int64](r, count)
	case valueTypeFloat64:
		return readArrayOf[float64](r, count)
	case valueTypeBool:
		return readBoolArray(r, count)
	case valueTypeString:
		return readStringArray(r, count)
	default:
		return Value{}, fmt.Errorf("unsupported array element type %d", elemType)
	}
}

// readArrayOf reads a typed numeric array using generics.
func readArrayOf[T any](r io.Reader, count uint64) (Value, error) {
	vals := make([]T, count)
	for i := range count {
		if err := binary.Read(r, binary.LittleEndian, &vals[i]); err != nil {
			return Value{}, fmt.Errorf("read array element %d: %w", i, err)
		}
	}
	return Value{data: vals}, nil
}

// readBoolArray reads an array of bools (each stored as a single byte).
func readBoolArray(r io.Reader, count uint64) (Value, error) {
	vals := make([]bool, count)
	for i := range count {
		var b uint8
		if err := binary.Read(r, binary.LittleEndian, &b); err != nil {
			return Value{}, fmt.Errorf("read bool array element %d: %w", i, err)
		}
		vals[i] = b != 0
	}
	return Value{data: vals}, nil
}

// readStringArray reads an array of GGUF strings.
func readStringArray(r io.Reader, count uint64) (Value, error) {
	vals := make([]string, count)
	for i := range count {
		s, err := readString(r)
		if err != nil {
			return Value{}, fmt.Errorf("read string array element %d: %w", i, err)
		}
		vals[i] = s
	}
	return Value{data: vals}, nil
}

// readTensorInfo reads a single tensor info entry from the stream.
func readTensorInfo(r io.Reader) (TensorInfo, error) {
	name, err := readString(r)
	if err != nil {
		return TensorInfo{}, fmt.Errorf("read tensor name: %w", err)
	}

	var nDims uint32
	if err := binary.Read(r, binary.LittleEndian, &nDims); err != nil {
		return TensorInfo{}, fmt.Errorf("read tensor dims count for %q: %w", name, err)
	}

	shape := make([]uint64, nDims)
	for i := range nDims {
		if err := binary.Read(r, binary.LittleEndian, &shape[i]); err != nil {
			return TensorInfo{}, fmt.Errorf("read tensor dim %d for %q: %w", i, name, err)
		}
	}

	var ttype uint32
	if err := binary.Read(r, binary.LittleEndian, &ttype); err != nil {
		return TensorInfo{}, fmt.Errorf("read tensor type for %q: %w", name, err)
	}

	var offset uint64
	if err := binary.Read(r, binary.LittleEndian, &offset); err != nil {
		return TensorInfo{}, fmt.Errorf("read tensor offset for %q: %w", name, err)
	}

	return TensorInfo{
		Name:   name,
		Shape:  shape,
		Type:   TensorType(ttype),
		Offset: offset,
	}, nil
}
