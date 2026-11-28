package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"io/fs"
	"math"
	"math/bits"
	"os"
	"strings"

	"github.com/gomlx/compute/dtypes"
	"github.com/pkg/errors"
)

// Sanity limits to prevent excessive allocations from malicious or malformed files.
const (
	maxHeaderSize  = 100 * 1024 * 1024 // 100 MiB max JSON header.
	maxTensorCount = 1 << 20           // ~1 million tensors per file.
	maxTensorDims  = 64                // Maximum number of tensor dimensions.
	maxTensorBytes = 64 << 30          // 64 GiB max byte size for a single tensor.
)

// maxAllocBytes returns the maximum allowed single-tensor allocation size on the current platform.
func maxAllocBytes() int64 {
	if int64(math.MaxInt) < maxTensorBytes {
		return int64(math.MaxInt)
	}
	return maxTensorBytes
}

// Header represents the JSON header of a safetensors file.
type Header struct {
	Tensors  map[string]*TensorMetadata // Tensor name -> metadata
	Metadata map[string]interface{}     // Optional __metadata__ field
}

// parseHeaderFromReader reads and parses the header from an io.Reader.
// If r supports Stat() (such as *os.File or fs.File), file-size bounds are
// checked eagerly before allocating the header buffer and for every tensor entry.
func parseHeaderFromReader(r io.Reader) (*Header, int64, error) {
	return parseHeaderFromReaderWithSize(r, -1)
}

// parseHeaderFromReaderWithSize reads and parses the header from r.
// If fileSize >= 0, header size and tensor data_offsets are validated against fileSize.
// If fileSize < 0 and r implements Stat(), fileSize is queried automatically.
func parseHeaderFromReaderWithSize(r io.Reader, fileSize int64) (*Header, int64, error) {
	if fileSize < 0 {
		if statter, ok := r.(interface{ Stat() (fs.FileInfo, error) }); ok {
			if fi, err := statter.Stat(); err == nil {
				fileSize = fi.Size()
			}
		}
	}
	if fileSize >= 0 && fileSize < 8 {
		return nil, 0, errors.Errorf("file too small (%d bytes) to contain safetensors header", fileSize)
	}

	// Read header size (8 bytes, little-endian)
	var headerSize uint64
	if err := binary.Read(r, binary.LittleEndian, &headerSize); err != nil {
		return nil, 0, errors.Wrap(err, "failed to read header size")
	}

	if headerSize == 0 {
		return nil, 0, errors.Errorf("invalid safetensors header size: 0 bytes")
	}
	if headerSize > maxHeaderSize { // Sanity check: 100MB max header
		return nil, 0, errors.Errorf("header size too large: %d bytes (limit %d)", headerSize, maxHeaderSize)
	}
	if fileSize >= 0 && headerSize > uint64(fileSize-8) {
		return nil, 0, errors.Errorf("header size %d exceeds remaining file size %d", headerSize, fileSize-8)
	}

	// Read JSON header
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(r, headerBytes); err != nil {
		return nil, 0, errors.Wrap(err, "failed to read header JSON")
	}

	// Parse JSON
	var rawHeader map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &rawHeader); err != nil {
		return nil, 0, errors.Wrap(err, "failed to parse header JSON")
	}
	if len(rawHeader) > maxTensorCount+1 {
		return nil, 0, errors.Errorf("header declares %d entries, exceeds limit %d", len(rawHeader), maxTensorCount)
	}

	// Data offset is after the 8-byte size + header.
	dataOffset := int64(8 + headerSize)
	dataSectionSize := int64(-1)
	if fileSize >= 0 {
		dataSectionSize = fileSize - dataOffset
	}

	header := &Header{
		Tensors:  make(map[string]*TensorMetadata, len(rawHeader)),
		Metadata: make(map[string]interface{}),
	}

	// Parse and validate each field eagerly so invalid shapes/offsets fail immediately.
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
			if err := validateTensorMetadata(&tm, dataOffset, dataSectionSize); err != nil {
				return nil, 0, err
			}
			header.Tensors[key] = &tm
		}
	}

	return header, dataOffset, nil
}

// validateTensorMetadata checks a tensor entry's dimensions, offsets, and (for supported dtypes)
// exact byte size against the data section size (if >= 0) and platform allocation limits.
func validateTensorMetadata(tm *TensorMetadata, dataOffset, dataSectionSize int64) error {
	if tm.Dtype == "" {
		return errors.Errorf("tensor %q has empty dtype", tm.Name)
	}
	lowerDtype := strings.ToLower(tm.Dtype)
	if lowerDtype == "invalid" || lowerDtype == "invaliddtype" {
		return errors.Errorf("tensor %q has invalid dtype %q", tm.Name, tm.Dtype)
	}

	nElements, err := tm.validateDimensions()
	if err != nil {
		return err
	}

	start, end := tm.DataOffsets[0], tm.DataOffsets[1]
	if start < 0 || end < start {
		return errors.Errorf("tensor %q has invalid data offsets [%d:%d]", tm.Name, start, end)
	}
	if dataOffset < 0 || dataOffset > math.MaxInt64-end {
		return errors.Errorf("tensor %q data offsets [%d:%d] overflow int64 with data offset %d", tm.Name, start, end, dataOffset)
	}
	if dataSectionSize >= 0 && end > dataSectionSize {
		return errors.Errorf("tensor %q data offsets [%d:%d] out of range for %d-byte data section", tm.Name, start, end, dataSectionSize)
	}

	spanBytes := end - start
	if spanBytes > maxAllocBytes() {
		return errors.Errorf("tensor %q byte size %d exceeds limit %d", tm.Name, spanBytes, maxAllocBytes())
	}

	if dtype, err := dtypeToGoMLX(tm.Dtype); err == nil {
		expectedBytes, err := checkedByteSizeForDType(dtype, nElements)
		if err != nil {
			return errors.Wrapf(err, "tensor %q shape %v (%s)", tm.Name, tm.Shape, tm.Dtype)
		}
		if spanBytes != expectedBytes {
			return errors.Errorf("tensor %q shape %v (%s) expected %d bytes, but got %d bytes in file",
				tm.Name, tm.Shape, tm.Dtype, expectedBytes, spanBytes)
		}
	} else if nElements > 0 {
		// Even for an unsupported dtype (e.g. FP8), require at least 1 bit per element.
		minBytes := (nElements + 7) / 8
		if minBytes > uint64(spanBytes) {
			return errors.Errorf("tensor %q shape %v (%s) requires at least %d bytes, but data_offsets specify %d bytes",
				tm.Name, tm.Shape, tm.Dtype, minBytes, spanBytes)
		}
	}
	return nil
}

// checkedByteSizeForDType computes the byte size for nElements of dtype with overflow and limit checks.
func checkedByteSizeForDType(dtype dtypes.DType, nElements uint64) (int64, error) {
	var numBytes uint64
	if dtype.IsPacked() {
		bitsPerElem := uint64(dtype.Bits())
		hi, totalBits := bits.Mul64(nElements, bitsPerElem)
		if hi != 0 || totalBits > math.MaxUint64-7 {
			return 0, errors.Errorf("byte size overflows for %d elements of %s", nElements, dtype)
		}
		numBytes = (totalBits + 7) / 8
	} else {
		elemSize := uint64(dtype.Size())
		hi, nb := bits.Mul64(nElements, elemSize)
		if hi != 0 {
			return 0, errors.Errorf("byte size overflows for %d elements of %s", nElements, dtype)
		}
		numBytes = nb
	}
	if numBytes > uint64(maxAllocBytes()) {
		return 0, errors.Errorf("tensor byte size %d exceeds limit %d", numBytes, maxAllocBytes())
	}
	return int64(numBytes), nil
}

// parseHeader reads and parses the header from a safetensors file path or via m.Repo.Open.
func (m *Model) parseHeader(path string) (*Header, int64, error) {
	if m.Repo != nil {
		f, err := m.Repo.Open(path)
		if err == nil {
			defer f.Close()
			return parseHeaderFromReader(f)
		}
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, 0, errors.Wrapf(err, "failed to open file %s", path)
	}
	defer f.Close()
	return parseHeaderFromReader(f)
}

func dtypeToGoMLX(stDtype string) (dtypes.DType, error) {
	dtype, found := dtypes.MapOfNames[strings.ToLower(stDtype)]
	if !found || dtype == dtypes.InvalidDType {
		return dtypes.InvalidDType, errors.Errorf("dtype %q not supported", stDtype)
	}
	return dtype, nil
}
