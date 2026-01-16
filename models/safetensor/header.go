package safetensor

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/pkg/errors"
)

// Header represents the JSON header of a safetensors file.
type Header struct {
	Tensors  map[string]*TensorMetadata // Tensor name -> metadata
	Metadata map[string]interface{}     // Optional __metadata__ field
}

// parseHeader reads and parses the header from a safetensors file.
// Safetensor format:
//
//	[8 bytes: header size as little-endian u64]
//	[header_size bytes: JSON header]
//	[remaining bytes: tensor data]
func (r *Model) parseHeader(path string) (*Header, int64, error) {
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

	header := &Header{
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

func dtypeToGoMLX(stDtype string) (dtypes.DType, error) {
	dtype, found := dtypes.MapOfNames[strings.ToLower(stDtype)]
	if !found {
		return dtypes.InvalidDType, errors.Errorf("dtype %q not supported", stDtype)
	}
	return dtype, nil
}
