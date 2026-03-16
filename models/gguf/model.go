package gguf

import (
	"path/filepath"
	"sync"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/pkg/errors"
)

// Model represents a GGUF model, optionally backed by a HuggingFace repo.
// For multimodal models (e.g., LLaVA), extras holds additional GGUF files
// (such as the vision encoder/projector). File is the primary file used for
// config metadata; tensors are looked up across all files.
type Model struct {
	Repo   *hub.Repo
	File   *File
	extras []extraEntry
	reader *Reader
	mu     sync.Mutex
}

// extraEntry pairs an extra GGUF file with its lazily-initialized reader.
type extraEntry struct {
	file   *File
	reader *Reader
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
	files, err := m.ggufFileNames()
	if err != nil {
		return err
	}
	if len(files) == 0 {
		return errors.Errorf("gguf: no .gguf file found in repository")
	}
	return m.loadFiles(files[:1])
}

// LoadAll downloads all .gguf files from the repo.
// The first file becomes File (primary, used for config metadata);
// additional files are stored as extras and searched for tensors.
// This is needed for multimodal models like LLaVA that split the text
// model and vision encoder into separate GGUF files.
func (m *Model) LoadAll() error {
	files, err := m.ggufFileNames()
	if err != nil {
		return err
	}
	if len(files) == 0 {
		return errors.Errorf("gguf: no .gguf files found in repository")
	}
	return m.loadFiles(files)
}

// LoadFiles downloads specific .gguf files from the repo by name.
// The first file becomes File (primary); additional files become extras.
func (m *Model) LoadFiles(filenames ...string) error {
	if m.Repo == nil {
		return errors.Errorf("gguf: repo is nil")
	}
	if len(filenames) == 0 {
		return errors.Errorf("gguf: no filenames specified")
	}
	return m.loadFiles(filenames)
}

// ListGGUFFiles returns the names of all .gguf files in the repo.
func (m *Model) ListGGUFFiles() ([]string, error) {
	return m.ggufFileNames()
}

// ggufFileNames returns the names of all .gguf files in the repo.
func (m *Model) ggufFileNames() ([]string, error) {
	if m.Repo == nil {
		return nil, errors.Errorf("gguf: repo is nil")
	}
	var files []string
	for filename, err := range m.Repo.IterFileNames() {
		if err != nil {
			return nil, errors.Wrapf(err, "gguf: list repo files")
		}
		if filepath.Ext(filename) == ".gguf" {
			files = append(files, filename)
		}
	}
	return files, nil
}

func (m *Model) loadFiles(filenames []string) error {
	for i, filename := range filenames {
		localPath, err := m.Repo.DownloadFile(filename)
		if err != nil {
			return errors.Wrapf(err, "gguf: download %s", filename)
		}

		f, err := Open(localPath)
		if err != nil {
			return errors.Wrapf(err, "gguf: parse %s", filename)
		}

		if i == 0 {
			m.File = f
		} else {
			m.extras = append(m.extras, extraEntry{file: f})
		}
	}
	return nil
}

// Close releases resources held by the Model, including any cached readers.
func (m *Model) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	var firstErr error
	if m.reader != nil {
		firstErr = m.reader.Close()
		m.reader = nil
	}
	for i := range m.extras {
		if r := m.extras[i].reader; r != nil {
			if err := r.Close(); err != nil && firstErr == nil {
				firstErr = err
			}
			m.extras[i].reader = nil
		}
	}
	return firstErr
}

// getReader returns a cached Reader for the primary file, creating one if necessary.
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

// getExtraReader returns a cached Reader for the extra file at the given index, creating one if necessary.
func (m *Model) getExtraReader(i int) (*Reader, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.extras[i].reader == nil {
		r, err := NewReader(m.extras[i].file)
		if err != nil {
			return nil, err
		}
		m.extras[i].reader = r
	}
	return m.extras[i].reader, nil
}

// findTensor searches all files for the named tensor, returning its info
// and file index (0=primary, 1+=extra). Returns -1 if not found.
func (m *Model) findTensor(name string) (TensorInfo, int) {
	if info, ok := m.File.GetTensorInfo(name); ok {
		return info, 0
	}
	for i := range m.extras {
		if info, ok := m.extras[i].file.GetTensorInfo(name); ok {
			return info, i + 1
		}
	}
	return TensorInfo{}, -1
}

// GetTensorInfo looks up tensor info by name across all files.
func (m *Model) GetTensorInfo(name string) (TensorInfo, bool) {
	info, idx := m.findTensor(name)
	return info, idx >= 0
}

// ListTensorNames returns all tensor names across all files.
func (m *Model) ListTensorNames() []string {
	if m.File == nil {
		return nil
	}
	names := m.File.ListTensorNames()
	for i := range m.extras {
		names = append(names, m.extras[i].file.ListTensorNames()...)
	}
	return names
}

// ExtraFiles returns the parsed File objects for any additional GGUF files
// beyond the primary (e.g., mmproj files for multimodal models).
func (m *Model) ExtraFiles() []*File {
	files := make([]*File, len(m.extras))
	for i := range m.extras {
		files[i] = m.extras[i].file
	}
	return files
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

// ReadTensor loads a single tensor by name, dequantizing if needed.
// Searches the primary file first, then extra files.
func (m *Model) ReadTensor(tensorName string) (*TensorAndName, error) {
	if m.File == nil {
		return nil, errors.Errorf("gguf: model not loaded, call Load() first")
	}

	reader, err := m.readerForTensor(tensorName)
	if err != nil {
		return nil, err
	}

	t, err := reader.ReadTensor(tensorName)
	if err != nil {
		return nil, err
	}
	return &TensorAndName{Name: tensorName, Tensor: t}, nil
}

// ReadTensorBytes loads raw bytes for a tensor without dequantization.
// Searches the primary file first, then extra files.
func (m *Model) ReadTensorBytes(tensorName string) ([]byte, *TensorInfo, error) {
	if m.File == nil {
		return nil, nil, errors.Errorf("gguf: model not loaded, call Load() first")
	}

	reader, err := m.readerForTensor(tensorName)
	if err != nil {
		return nil, nil, err
	}

	return reader.ReadTensorRaw(tensorName)
}

// readerForTensor returns the reader for the file containing the named tensor.
func (m *Model) readerForTensor(name string) (*Reader, error) {
	_, idx := m.findTensor(name)
	if idx < 0 {
		return nil, errors.Errorf("gguf: tensor %q not found", name)
	}
	if idx == 0 {
		return m.getReader()
	}
	return m.getExtraReader(idx - 1)
}

// IterTensors returns an iterator over all tensors across all files as GoMLX tensors.
// Tensors are read sequentially sorted by offset for optimal I/O.
func (m *Model) IterTensors() func(yield func(TensorAndName, error) bool) {
	return func(yield func(TensorAndName, error) bool) {
		if m.File == nil {
			yield(TensorAndName{}, errors.Errorf("gguf: model not loaded, call Load() first"))
			return
		}

		// Iterate primary file.
		reader, err := m.getReader()
		if err != nil {
			yield(TensorAndName{}, err)
			return
		}
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

		// Iterate extra files.
		for i := range m.extras {
			r, err := m.getExtraReader(i)
			if err != nil {
				yield(TensorAndName{}, err)
				return
			}
			for _, info := range m.extras[i].file.TensorInfos {
				t, err := r.ReadTensor(info.Name)
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
