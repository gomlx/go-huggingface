package hub

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
	"golang.org/x/exp/mmap"
)

// GetSafetensorHeader returns the parsed safetensor header for a specific tensor.
func (r *Repo) GetSafetensorHeader(filename string) (*SafetensorHeader, error) {
	localPath, err := r.DownloadFile(filename)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to download %s", filename)
	}

	header, _, err := parseSafetensorHeader(localPath)
	if err != nil {
		return nil, err
	}

	return header, nil
}

// LoadSafetensor loads a tensor from a safetensors file and converts it to a GoMLX tensor.
// The returned tensor can be used with graph.ConstTensor().
func (r *Repo) LoadSafetensor(filename, tensorName string) (*tensors.Tensor, error) {
	localPath, err := r.DownloadFile(filename)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to download %s", filename)
	}

	header, dataOffset, err := parseSafetensorHeader(localPath)
	if err != nil {
		return nil, err
	}

	meta, ok := header.Tensors[tensorName]
	if !ok {
		return nil, errors.Errorf("tensor %s not found in %s", tensorName, filename)
	}

	// Open file and seek to tensor data
	f, err := os.Open(localPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open %s", localPath)
	}
	defer f.Close()

	// Seek to tensor data position
	offset := dataOffset + meta.DataOffsets[0]
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return nil, errors.Wrap(err, "failed to seek to tensor data")
	}

	// Read tensor data
	data := make([]byte, meta.SizeBytes())
	if _, err := io.ReadFull(f, data); err != nil {
		return nil, errors.Wrap(err, "failed to read tensor data")
	}

	// Convert dtype
	dtype, err := safetensorDtypeToGoMLX(meta.Dtype)
	if err != nil {
		return nil, err
	}

	// Convert shape
	dims := make([]int, len(meta.Shape))
	copy(dims, meta.Shape)

	// Convert bytes to Go slice
	numElements := meta.NumElements()
	goSlice, err := bytesToGoSlice(data, dtype, numElements)
	if err != nil {
		return nil, err
	}

	// Create tensor from flat data
	return createTensorFromGoSlice(goSlice, dtype, dims)
}

// LoadSafetensorStreaming loads a tensor using memory-mapped streaming.
// This is more memory-efficient for large tensors.
func (r *Repo) LoadSafetensorStreaming(filename, tensorName string) (*tensors.Tensor, error) {
	localPath, err := r.DownloadFile(filename)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to download %s", filename)
	}

	header, dataOffset, err := parseSafetensorHeader(localPath)
	if err != nil {
		return nil, err
	}

	meta, ok := header.Tensors[tensorName]
	if !ok {
		return nil, errors.Errorf("tensor %s not found in %s", tensorName, filename)
	}

	// Open file with mmap
	reader, err := mmap.Open(localPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to mmap %s", localPath)
	}
	defer reader.Close()

	// Convert dtype
	dtype, err := safetensorDtypeToGoMLX(meta.Dtype)
	if err != nil {
		return nil, err
	}

	// Create shape
	dims := make([]int, len(meta.Shape))
	copy(dims, meta.Shape)

	// Read data from mmap
	tensorOffset := dataOffset + meta.DataOffsets[0]
	data := make([]byte, meta.SizeBytes())
	if _, err := reader.ReadAt(data, tensorOffset); err != nil && err != io.EOF {
		return nil, errors.Wrap(err, "failed to read tensor data")
	}

	// Convert bytes to Go slice
	numElements := meta.NumElements()
	goSlice, err := bytesToGoSlice(data, dtype, numElements)
	if err != nil {
		return nil, err
	}

	// Create tensor
	return createTensorFromGoSlice(goSlice, dtype, dims)
}

// DetectShardedModel checks if the repository contains a sharded model and returns the index filename.
func (r *Repo) DetectShardedModel() (string, bool, error) {
	// Look for model.safetensors.index.json or pytorch_model.bin.index.json
	commonIndexFiles := []string{
		"model.safetensors.index.json",
		"pytorch_model.safetensors.index.json",
	}

	for filename, err := range r.IterFileNames() {
		if err != nil {
			return "", false, err
		}

		for _, indexName := range commonIndexFiles {
			if filename == indexName || filepath.Base(filename) == indexName {
				return filename, true, nil
			}
		}
	}

	return "", false, nil
}

// LoadModel loads a model as a Model, whether it's sharded or a single file.
// This provides a unified interface for loading any safetensors model.
// It automatically detects sharded models via index files, otherwise treats the first
// .safetensors file as a single-file model.
func (r *Repo) LoadModel() (*Model, error) {
	// First check if this is a sharded model
	indexFile, isSharded, err := r.DetectShardedModel()
	if err != nil {
		return nil, err
	}

	if isSharded {
		return r.LoadShardedModel(indexFile)
	}

	// Not sharded - find the first .safetensors file and create a single-file Model
	for filename, err := range r.IterFileNames() {
		if err != nil {
			return nil, err
		}

		if filepath.Ext(filename) == ".safetensors" {
			// Download and parse the file to get tensor names
			localPath, err := r.DownloadFile(filename)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to download %s", filename)
			}

			header, _, err := parseSafetensorHeader(localPath)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to parse header for %s", filename)
			}

			// Create a synthetic index with all tensors pointing to this one file
			weightMap := make(map[string]string)
			for tensorName := range header.Tensors {
				weightMap[tensorName] = filename
			}

			return &Model{
				repo:      r,
				IndexFile: filename, // Use the safetensors file itself as the "index"
				Index: &ShardedModelIndex{
					WeightMap: weightMap,
				},
				headers: map[string]*SafetensorHeader{
					filename: header,
				},
			}, nil
		}
	}

	return nil, errors.New("no .safetensors files found in repository")
}

// LoadShardedModel loads a sharded model index file (typically model.safetensors.index.json).
func (r *Repo) LoadShardedModel(indexFilename string) (*Model, error) {
	localPath, err := r.DownloadFile(indexFilename)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to download %s", indexFilename)
	}

	data, err := os.ReadFile(localPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read %s", localPath)
	}

	var index ShardedModelIndex
	if err := json.Unmarshal(data, &index); err != nil {
		return nil, errors.Wrap(err, "failed to parse sharded model index")
	}

	return &Model{
		repo:      r,
		IndexFile: indexFilename,
		Index:     &index,
		headers:   make(map[string]*SafetensorHeader),
	}, nil
}

// SafetensorFileInfo holds information about a safetensor file.
type SafetensorFileInfo struct {
	Filename string
	Header   *SafetensorHeader
}

// TensorWithName holds a tensor name and its GoMLX tensor data.
type TensorWithName struct {
	Name   string
	Tensor *tensors.Tensor
}

// IterSafetensors returns an iterator over all .safetensors files in the repository.
func (r *Repo) IterSafetensors() func(yield func(SafetensorFileInfo, error) bool) {
	return func(yield func(SafetensorFileInfo, error) bool) {
		for filename, err := range r.IterFileNames() {
			if err != nil {
				yield(SafetensorFileInfo{}, err)
				return
			}

			// Only process .safetensors files
			if !strings.HasSuffix(filename, ".safetensors") {
				continue
			}

			// Download and parse header
			localPath, err := r.DownloadFile(filename)
			if err != nil {
				yield(SafetensorFileInfo{}, errors.Wrapf(err, "failed to download %s", filename))
				return
			}

			header, _, err := parseSafetensorHeader(localPath)
			if err != nil {
				yield(SafetensorFileInfo{}, errors.Wrapf(err, "failed to parse header for %s", filename))
				return
			}

			if !yield(SafetensorFileInfo{Filename: filename, Header: header}, nil) {
				return
			}
		}
	}
}

// IterAllTensors returns an iterator over all tensors as GoMLX tensors.
// It uses LoadModel() which handles both sharded and single-file models uniformly.
func (r *Repo) IterAllTensors() func(yield func(TensorWithName, error) bool) {
	return func(yield func(TensorWithName, error) bool) {
		// Load model (handles both sharded and single-file)
		model, err := r.LoadModel()
		if err != nil {
			yield(TensorWithName{}, errors.Wrap(err, "failed to load model"))
			return
		}

		// Iterate over all tensors in the model
		for tensorName := range model.Index.WeightMap {
			tensor, err := model.LoadTensor(tensorName)
			if err != nil {
				yield(TensorWithName{}, err)
				return
			}

			if !yield(TensorWithName{Name: tensorName, Tensor: tensor}, nil) {
				return
			}
		}
	}
}

// parseSafetensorHeader reads and parses the header from a safetensors file.
// Safetensor format:
//
//	[8 bytes: header size as little-endian u64]
//	[header_size bytes: JSON header]
//	[remaining bytes: tensor data]
func parseSafetensorHeader(path string) (*SafetensorHeader, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, errors.Wrapf(err, "failed to open file %s", path)
	}
	defer f.Close()

	// Read header size (8 bytes, little-endian)
	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		return nil, 0, errors.Wrap(err, "failed to read header size")
	}

	if headerSize > 100*1024*1024 { // Sanity check: 100MB max header
		return nil, 0, errors.Errorf("header size too large: %d bytes", headerSize)
	}

	// Read JSON header
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return nil, 0, errors.Wrap(err, "failed to read header JSON")
	}

	// Parse JSON
	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, 0, errors.Wrap(err, "failed to parse header JSON")
	}

	header := &SafetensorHeader{
		Tensors:  make(map[string]*TensorMetadata),
		Metadata: make(map[string]interface{}),
	}

	// Parse each field
	for key, value := range rawHeader {
		if key == "__metadata__" {
			if err := json.Unmarshal(value, &header.Metadata); err != nil {
				return nil, 0, errors.Wrap(err, "failed to parse __metadata__")
			}
		} else {
			var tm TensorMetadata
			if err := json.Unmarshal(value, &tm); err != nil {
				return nil, 0, errors.Wrapf(err, "failed to parse tensor metadata for %s", key)
			}
			tm.Name = key
			header.Tensors[key] = &tm
		}
	}

	// Data offset is after the 8-byte size + header
	dataOffset := int64(8 + headerSize)
	return header, dataOffset, nil
}

// safetensorDtypeToGoMLX maps safetensor dtype strings to GoMLX dtypes.
// safetensorToGoMLXDtype maps safetensor dtype names to GoMLX dtype names.
// Safetensors uses naming like "I64", "F32", while GoMLX uses "Int64", "Float32".
var safetensorToGoMLXDtype = map[string]string{
	"I8":   "Int8",
	"I16":  "Int16",
	"I32":  "Int32",
	"I64":  "Int64",
	"U8":   "Uint8",
	"U16":  "Uint16",
	"U32":  "Uint32",
	"U64":  "Uint64",
	"F16":  "Float16",
	"F32":  "Float32",
	"F64":  "Float64",
	"BF16": "BFloat16",
	"BOOL": "Bool",
}

func safetensorDtypeToGoMLX(stDtype string) (dtypes.DType, error) {
	// First try direct mapping from safetensor naming to GoMLX naming
	if gomlxName, found := safetensorToGoMLXDtype[stDtype]; found {
		if dtype, found := dtypes.MapOfNames[gomlxName]; found {
			return dtype, nil
		}
	}

	// Fallback: try looking up stDtype directly in MapOfNames (for any aliases)
	if dtype, found := dtypes.MapOfNames[stDtype]; found {
		return dtype, nil
	}

	return dtypes.InvalidDType, fmt.Errorf("unsupported safetensor dtype: %s", stDtype)
}

// bytesToGoSlice converts raw bytes to a Go slice of the appropriate type.
func bytesToGoSlice(data []byte, dtype dtypes.DType, numElements int64) (interface{}, error) {
	switch dtype {
	case dtypes.Float32:
		if len(data) != int(numElements)*4 {
			return nil, fmt.Errorf("data size mismatch for float32: got %d bytes, expected %d", len(data), numElements*4)
		}
		slice := make([]float32, numElements)
		for i := int64(0); i < numElements; i++ {
			bits := binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
			slice[i] = math.Float32frombits(bits)
		}
		return slice, nil

	case dtypes.Float64:
		if len(data) != int(numElements)*8 {
			return nil, fmt.Errorf("data size mismatch for float64: got %d bytes, expected %d", len(data), numElements*8)
		}
		slice := make([]float64, numElements)
		for i := int64(0); i < numElements; i++ {
			bits := binary.LittleEndian.Uint64(data[i*8 : (i+1)*8])
			slice[i] = math.Float64frombits(bits)
		}
		return slice, nil

	case dtypes.Int32:
		if len(data) != int(numElements)*4 {
			return nil, fmt.Errorf("data size mismatch for int32: got %d bytes, expected %d", len(data), numElements*4)
		}
		slice := make([]int32, numElements)
		for i := int64(0); i < numElements; i++ {
			slice[i] = int32(binary.LittleEndian.Uint32(data[i*4 : (i+1)*4]))
		}
		return slice, nil

	case dtypes.Int64:
		if len(data) != int(numElements)*8 {
			return nil, fmt.Errorf("data size mismatch for int64: got %d bytes, expected %d", len(data), numElements*8)
		}
		slice := make([]int64, numElements)
		for i := int64(0); i < numElements; i++ {
			slice[i] = int64(binary.LittleEndian.Uint64(data[i*8 : (i+1)*8]))
		}
		return slice, nil

	case dtypes.Int16:
		if len(data) != int(numElements)*2 {
			return nil, fmt.Errorf("data size mismatch for int16: got %d bytes, expected %d", len(data), numElements*2)
		}
		slice := make([]int16, numElements)
		for i := int64(0); i < numElements; i++ {
			slice[i] = int16(binary.LittleEndian.Uint16(data[i*2 : (i+1)*2]))
		}
		return slice, nil

	case dtypes.Int8:
		if len(data) != int(numElements) {
			return nil, fmt.Errorf("data size mismatch for int8: got %d bytes, expected %d", len(data), numElements)
		}
		slice := make([]int8, numElements)
		for i := int64(0); i < numElements; i++ {
			slice[i] = int8(data[i])
		}
		return slice, nil

	case dtypes.Uint32:
		if len(data) != int(numElements)*4 {
			return nil, fmt.Errorf("data size mismatch for uint32: got %d bytes, expected %d", len(data), numElements*4)
		}
		slice := make([]uint32, numElements)
		for i := int64(0); i < numElements; i++ {
			slice[i] = binary.LittleEndian.Uint32(data[i*4 : (i+1)*4])
		}
		return slice, nil

	case dtypes.Uint64:
		if len(data) != int(numElements)*8 {
			return nil, fmt.Errorf("data size mismatch for uint64: got %d bytes, expected %d", len(data), numElements*8)
		}
		slice := make([]uint64, numElements)
		for i := int64(0); i < numElements; i++ {
			slice[i] = binary.LittleEndian.Uint64(data[i*8 : (i+1)*8])
		}
		return slice, nil

	case dtypes.Uint16:
		if len(data) != int(numElements)*2 {
			return nil, fmt.Errorf("data size mismatch for uint16: got %d bytes, expected %d", len(data), numElements*2)
		}
		slice := make([]uint16, numElements)
		for i := int64(0); i < numElements; i++ {
			slice[i] = binary.LittleEndian.Uint16(data[i*2 : (i+1)*2])
		}
		return slice, nil

	case dtypes.Uint8:
		if len(data) != int(numElements) {
			return nil, fmt.Errorf("data size mismatch for uint8: got %d bytes, expected %d", len(data), numElements)
		}
		slice := make([]uint8, numElements)
		copy(slice, data)
		return slice, nil

	case dtypes.Bool:
		if len(data) != int(numElements) {
			return nil, fmt.Errorf("data size mismatch for bool: got %d bytes, expected %d", len(data), numElements)
		}
		slice := make([]bool, numElements)
		for i := int64(0); i < numElements; i++ {
			slice[i] = data[i] != 0
		}
		return slice, nil

	case dtypes.Complex64:
		if len(data) != int(numElements)*8 {
			return nil, fmt.Errorf("data size mismatch for complex64: got %d bytes, expected %d", len(data), numElements*8)
		}
		slice := make([]complex64, numElements)
		for i := int64(0); i < numElements; i++ {
			realBits := binary.LittleEndian.Uint32(data[i*8 : i*8+4])
			imagBits := binary.LittleEndian.Uint32(data[i*8+4 : (i+1)*8])
			slice[i] = complex(math.Float32frombits(realBits), math.Float32frombits(imagBits))
		}
		return slice, nil

	case dtypes.Complex128:
		if len(data) != int(numElements)*16 {
			return nil, fmt.Errorf("data size mismatch for complex128: got %d bytes, expected %d", len(data), numElements*16)
		}
		slice := make([]complex128, numElements)
		for i := int64(0); i < numElements; i++ {
			realBits := binary.LittleEndian.Uint64(data[i*16 : i*16+8])
			imagBits := binary.LittleEndian.Uint64(data[i*16+8 : (i+1)*16])
			slice[i] = complex(math.Float64frombits(realBits), math.Float64frombits(imagBits))
		}
		return slice, nil

	case dtypes.Float16, dtypes.BFloat16:
		// F16 and BF16 stored as uint16
		if len(data) != int(numElements)*2 {
			return nil, fmt.Errorf("data size mismatch for float16/bfloat16: got %d bytes, expected %d", len(data), numElements*2)
		}
		slice := make([]uint16, numElements)
		for i := int64(0); i < numElements; i++ {
			slice[i] = binary.LittleEndian.Uint16(data[i*2 : (i+1)*2])
		}
		return slice, nil

	default:
		return nil, fmt.Errorf("unsupported dtype: %v", dtype)
	}
}

// createTensorFromGoSlice creates a GoMLX tensor from a Go slice with the given dtype and dimensions.
func createTensorFromGoSlice(goSlice interface{}, dtype dtypes.DType, dims []int) (*tensors.Tensor, error) {
	switch dtype {
	case dtypes.Float32:
		return tensors.FromFlatDataAndDimensions(goSlice.([]float32), dims...), nil
	case dtypes.Float64:
		return tensors.FromFlatDataAndDimensions(goSlice.([]float64), dims...), nil
	case dtypes.Int32:
		return tensors.FromFlatDataAndDimensions(goSlice.([]int32), dims...), nil
	case dtypes.Int64:
		return tensors.FromFlatDataAndDimensions(goSlice.([]int64), dims...), nil
	case dtypes.Int16:
		return tensors.FromFlatDataAndDimensions(goSlice.([]int16), dims...), nil
	case dtypes.Int8:
		return tensors.FromFlatDataAndDimensions(goSlice.([]int8), dims...), nil
	case dtypes.Uint32:
		return tensors.FromFlatDataAndDimensions(goSlice.([]uint32), dims...), nil
	case dtypes.Uint64:
		return tensors.FromFlatDataAndDimensions(goSlice.([]uint64), dims...), nil
	case dtypes.Uint16:
		return tensors.FromFlatDataAndDimensions(goSlice.([]uint16), dims...), nil
	case dtypes.Uint8:
		return tensors.FromFlatDataAndDimensions(goSlice.([]uint8), dims...), nil
	case dtypes.Bool:
		return tensors.FromFlatDataAndDimensions(goSlice.([]bool), dims...), nil
	case dtypes.Complex64:
		return tensors.FromFlatDataAndDimensions(goSlice.([]complex64), dims...), nil
	case dtypes.Complex128:
		return tensors.FromFlatDataAndDimensions(goSlice.([]complex128), dims...), nil
	case dtypes.Float16, dtypes.BFloat16:
		return tensors.FromFlatDataAndDimensions(goSlice.([]uint16), dims...), nil
	default:
		return nil, fmt.Errorf("unsupported dtype: %v", dtype)
	}
}

// dtypeSize returns the size in bytes of a single element of the given dtype.
func dtypeSize(dtype string) (int, error) {
	switch dtype {
	case "F64", "I64", "U64":
		return 8, nil
	case "F32", "I32", "U32":
		return 4, nil
	case "F16", "BF16", "I16", "U16":
		return 2, nil
	case "I8", "U8", "BOOL":
		return 1, nil
	default:
		return 0, fmt.Errorf("unknown dtype: %s", dtype)
	}
}
