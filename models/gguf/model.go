package gguf

import (
	"fmt"
	"path/filepath"
	"sync"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// Model represents a GGUF model, optionally backed by a HuggingFace repo.
type Model struct {
	Repo   *hub.Repo
	File   *File
	reader *Reader
	mu     sync.Mutex
}

// TensorAndName holds a tensor name and its GoMLX tensor data.
type TensorAndName struct {
	Name   string
	Tensor *tensors.Tensor
}

// New creates a Model from a HuggingFace repo, downloading and parsing the GGUF file.
func New(repo *hub.Repo) (*Model, error) {
	m := NewEmpty(repo)
	if err := m.Load(); err != nil {
		return nil, err
	}
	return m, nil
}

// NewFromFile creates a Model directly from a local GGUF file path.
func NewFromFile(path string) (*Model, error) {
	f, err := Open(path)
	if err != nil {
		return nil, err
	}
	return &Model{File: f}, nil
}

// NewEmpty creates an empty Model for manual control. Call Load() to download and parse.
func NewEmpty(repo *hub.Repo) *Model {
	return &Model{Repo: repo}
}

// Load downloads the first .gguf file from the repo and parses it.
func (m *Model) Load() error {
	if m.Repo == nil {
		return fmt.Errorf("gguf: repo is nil")
	}

	// Find the first .gguf file in the repo.
	var ggufFile string
	for filename, err := range m.Repo.IterFileNames() {
		if err != nil {
			return fmt.Errorf("gguf: list repo files: %w", err)
		}
		if filepath.Ext(filename) == ".gguf" {
			ggufFile = filename
			break
		}
	}
	if ggufFile == "" {
		return fmt.Errorf("gguf: no .gguf file found in repository")
	}

	localPath, err := m.Repo.DownloadFile(ggufFile)
	if err != nil {
		return fmt.Errorf("gguf: download %s: %w", ggufFile, err)
	}

	f, err := Open(localPath)
	if err != nil {
		return fmt.Errorf("gguf: parse %s: %w", ggufFile, err)
	}

	m.File = f
	return nil
}

// Close releases resources held by the Model, including any cached reader.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.reader != nil {
		err := m.reader.Close()
		m.reader = nil
		return err
	}
	return nil
}

// getReader returns a cached Reader, creating one if necessary.
func (m *Model) getReader() (*Reader, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.reader == nil {
		r, err := NewReader(m.File)
		if err != nil {
			return nil, err
		}
		m.reader = r
	}
	return m.reader, nil
}

// ListTensorNames returns all tensor names in the model.
func (m *Model) ListTensorNames() []string {
	if m.File == nil {
		return nil
	}
	return m.File.ListTensorNames()
}

// GetKeyValue looks up a metadata key-value pair.
func (m *Model) GetKeyValue(key string) (KeyValue, bool) {
	if m.File == nil {
		return KeyValue{}, false
	}
	return m.File.GetKeyValue(key)
}

// Architecture returns the model architecture string.
func (m *Model) Architecture() string {
	if m.File == nil {
		return ""
	}
	return m.File.Architecture()
}

// GetTensor loads a single tensor by name, dequantizing if needed.
func (m *Model) GetTensor(tensorName string) (*TensorAndName, error) {
	if m.File == nil {
		return nil, fmt.Errorf("gguf: model not loaded, call Load() first")
	}

	reader, err := m.getReader()
	if err != nil {
		return nil, err
	}

	t, err := reader.ReadTensor(tensorName)
	if err != nil {
		return nil, err
	}
	return &TensorAndName{Name: tensorName, Tensor: t}, nil
}

// IterTensors returns an iterator over all tensors as GoMLX tensors.
// Tensors are read sequentially sorted by offset for optimal I/O.
func (m *Model) IterTensors() func(yield func(TensorAndName, error) bool) {
	return func(yield func(TensorAndName, error) bool) {
		if m.File == nil {
			yield(TensorAndName{}, fmt.Errorf("gguf: model not loaded, call Load() first"))
			return
		}

		reader, err := m.getReader()
		if err != nil {
			yield(TensorAndName{}, err)
			return
		}

		// TensorInfos are pre-sorted by offset in Open() for sequential I/O.
		for _, info := range m.File.TensorInfos {
			t, err := reader.ReadTensor(info.Name)
			if err != nil {
				yield(TensorAndName{}, err)
				return
			}
			if !yield(TensorAndName{Name: info.Name, Tensor: t}, nil) {
				return
			}
		}
	}
}

// IterTensorsFromRepo creates a Model from a repo and iterates over all tensors.
func IterTensorsFromRepo(repo *hub.Repo) func(yield func(TensorAndName, error) bool) {
	return func(yield func(TensorAndName, error) bool) {
		m, err := New(repo)
		if err != nil {
			yield(TensorAndName{}, err)
			return
		}
		defer m.Close()
		for tn, err := range m.IterTensors() {
			if err != nil {
				yield(TensorAndName{}, err)
				return
			}
			if !yield(tn, nil) {
				return
			}
		}
	}
}
