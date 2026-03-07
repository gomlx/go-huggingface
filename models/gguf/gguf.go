package gguf

import (
	"bufio"
	"cmp"
	"encoding/binary"
	"io"
	"os"
	"slices"

	"github.com/pkg/errors"
)

const (
	ggufMagic           = "GGUF"
	defaultAlignment    = 32
	minSupportedVersion = 2

	// Well-known GGUF metadata keys from the specification.
	KeyGeneralArchitecture = "general.architecture"
	KeyGeneralAlignment    = "general.alignment"
)

// Sanity limits to prevent excessive allocations from malicious files.
const (
	maxKVCount     = 1 << 16 // 65536 metadata pairs.
	maxTensorCount = 1 << 20 // ~1 million tensors.
	maxArrayCount  = 1 << 24 // ~16 million array elements.
	maxTensorDims  = 8       // Maximum number of tensor dimensions.
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
		return nil, errors.Wrapf(err, "gguf: open %s", path)
	}
	defer f.Close()

	file := &File{path: path}
	r := &countingReader{r: bufio.NewReaderSize(f, 64*1024)}

	// Read and validate magic number.
	var magic [4]byte
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return nil, errors.Wrapf(err, "gguf: read magic")
	}
	if string(magic[:]) != ggufMagic {
		return nil, errors.Errorf("gguf: invalid magic %q, expected %q", magic[:], ggufMagic)
	}

	// Read version.
	if err := binary.Read(r, binary.LittleEndian, &file.Version); err != nil {
		return nil, errors.Wrapf(err, "gguf: read version")
	}
	if file.Version < minSupportedVersion {
		return nil, errors.Errorf("gguf: unsupported version %d (minimum %d)", file.Version, minSupportedVersion)
	}

	// Read counts.
	var tensorCount, kvCount uint64
	if err := binary.Read(r, binary.LittleEndian, &tensorCount); err != nil {
		return nil, errors.Wrapf(err, "gguf: read tensor count")
	}
	if err := binary.Read(r, binary.LittleEndian, &kvCount); err != nil {
		return nil, errors.Wrapf(err, "gguf: read kv count")
	}
	if tensorCount > maxTensorCount {
		return nil, errors.Errorf("gguf: tensor count %d exceeds limit %d", tensorCount, maxTensorCount)
	}
	if kvCount > maxKVCount {
		return nil, errors.Errorf("gguf: kv count %d exceeds limit %d", kvCount, maxKVCount)
	}

	// Read all key-value pairs.
	file.KeyValues = make([]KeyValue, 0, kvCount)
	for range kvCount {
		kv, err := readKeyValue(r)
		if err != nil {
			return nil, errors.Wrapf(err, "gguf: read kv pair %d/%d", len(file.KeyValues), kvCount)
		}
		file.KeyValues = append(file.KeyValues, kv)
	}

	// Read all tensor info entries.
	file.TensorInfos = make([]TensorInfo, 0, tensorCount)
	for range tensorCount {
		ti, err := readTensorInfo(r)
		if err != nil {
			return nil, errors.Wrapf(err, "gguf: read tensor info %d/%d", len(file.TensorInfos), tensorCount)
		}
		file.TensorInfos = append(file.TensorInfos, ti)
	}

	// Build indexes (needed before alignment lookup).
	file.kvByKey = make(map[string]*KeyValue, len(file.KeyValues))
	for i := range file.KeyValues {
		file.kvByKey[file.KeyValues[i].Key] = &file.KeyValues[i]
	}
	file.tensorByName = make(map[string]*TensorInfo, len(file.TensorInfos))
	for i := range file.TensorInfos {
		file.tensorByName[file.TensorInfos[i].Name] = &file.TensorInfos[i]
	}

	// Sort tensors by offset for optimal sequential I/O.
	slices.SortFunc(file.TensorInfos, func(a, b TensorInfo) int {
		return cmp.Compare(a.Offset, b.Offset)
	})

	// Compute aligned data offset.
	file.Alignment = defaultAlignment
	if kv, ok := file.GetKeyValue(KeyGeneralAlignment); ok {
		if a := kv.Uint64(); a > 0 {
			file.Alignment = a
		}
	}
	offset := uint64(r.n)
	alignment := file.Alignment
	file.dataOffset = int64(offset + (alignment-offset%alignment)%alignment)

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
	kv, ok := f.GetKeyValue(KeyGeneralArchitecture)
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
		return "", errors.Wrapf(err, "read string length")
	}
	if length > 1<<20 { // 1MB sanity check for a single string.
		return "", errors.Errorf("string length %d exceeds 1MB limit", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", errors.Wrapf(err, "read string data")
	}
	return string(buf), nil
}

// readKeyValue reads a single GGUF key-value pair from the stream.
func readKeyValue(r io.Reader) (KeyValue, error) {
	key, err := readString(r)
	if err != nil {
		return KeyValue{}, errors.Wrapf(err, "read key")
	}

	var typeTag uint32
	if err := binary.Read(r, binary.LittleEndian, &typeTag); err != nil {
		return KeyValue{}, errors.Wrapf(err, "read value type for %q", key)
	}

	val, err := readValue(r, ggufValueType(typeTag))
	if err != nil {
		return KeyValue{}, errors.Wrapf(err, "read value for %q (type %d)", key, typeTag)
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
		return Value{}, errors.Errorf("unknown value type %d", vtype)
	}
}

// readArray reads a GGUF typed array: uint32 element type, uint64 count, then elements.
func readArray(r io.Reader) (Value, error) {
	var elemType uint32
	if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
		return Value{}, errors.Wrapf(err, "read array element type")
	}
	var count uint64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return Value{}, errors.Wrapf(err, "read array count")
	}
	if count > maxArrayCount {
		return Value{}, errors.Errorf("array count %d exceeds limit %d", count, maxArrayCount)
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
		return Value{}, errors.Errorf("unsupported array element type %d", elemType)
	}
}

// readArrayOf reads a typed numeric array in a single binary.Read call.
func readArrayOf[T any](r io.Reader, count uint64) (Value, error) {
	vals := make([]T, count)
	if err := binary.Read(r, binary.LittleEndian, vals); err != nil {
		return Value{}, errors.Wrapf(err, "read array (%d elements)", count)
	}
	return Value{data: vals}, nil
}

// readBoolArray reads an array of bools (each stored as a single byte).
func readBoolArray(r io.Reader, count uint64) (Value, error) {
	buf := make([]byte, count)
	if _, err := io.ReadFull(r, buf); err != nil {
		return Value{}, errors.Wrapf(err, "read bool array")
	}

	vals := make([]bool, count)
	for i, b := range buf {
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
			return Value{}, errors.Wrapf(err, "read string array element %d", i)
		}
		vals[i] = s
	}
	return Value{data: vals}, nil
}

// readTensorInfo reads a single tensor info entry from the stream.
func readTensorInfo(r io.Reader) (TensorInfo, error) {
	name, err := readString(r)
	if err != nil {
		return TensorInfo{}, errors.Wrapf(err, "read tensor name")
	}

	var nDims uint32
	if err := binary.Read(r, binary.LittleEndian, &nDims); err != nil {
		return TensorInfo{}, errors.Wrapf(err, "read tensor dims count for %q", name)
	}
	if nDims > maxTensorDims {
		return TensorInfo{}, errors.Errorf("tensor %q has %d dimensions, exceeds limit %d", name, nDims, maxTensorDims)
	}

	shape := make([]uint64, nDims)
	for i := range nDims {
		if err := binary.Read(r, binary.LittleEndian, &shape[i]); err != nil {
			return TensorInfo{}, errors.Wrapf(err, "read tensor dim %d for %q", i, name)
		}
	}

	var ttype uint32
	if err := binary.Read(r, binary.LittleEndian, &ttype); err != nil {
		return TensorInfo{}, errors.Wrapf(err, "read tensor type for %q", name)
	}

	var offset uint64
	if err := binary.Read(r, binary.LittleEndian, &offset); err != nil {
		return TensorInfo{}, errors.Wrapf(err, "read tensor offset for %q", name)
	}

	return TensorInfo{
		Name:   name,
		Shape:  shape,
		Type:   TensorType(ttype),
		Offset: offset,
	}, nil
}
