package transformer

import (
	"encoding/json"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/internal/files"
	"github.com/pkg/errors"
)

// CheckpointHFSubDir is the default sub-directory name inside a GoMLX checkpoint directory
// where Hugging Face configuration and tokenizer files are stored.
const CheckpointHFSubDir = "hf"

// isConfigOrTokenizerFile returns whether fileName is a configuration, tokenizer, or metadata file
// that should be preserved when exporting model configs. It deliberately excludes heavy model weights
// (such as .safetensors, .bin, .pt, .gguf, .onnx, etc.).
func isConfigOrTokenizerFile(fileName string) bool {
	lower := strings.ToLower(fileName)
	base := path.Base(lower)

	// All JSON files (config.json, generation_config.json, modules.json, tokenizer.json,
	// tokenizer_config.json, special_tokens_map.json, 1_Pooling/config.json, etc.)
	if strings.HasSuffix(lower, ".json") {
		return true
	}

	// Tokenizer vocabulary, merges, and binary model files
	if base == "vocab.txt" || base == "merges.txt" {
		return true
	}
	if strings.HasSuffix(lower, ".model") {
		return true
	}

	// YAML configuration files if any
	if strings.HasSuffix(lower, ".yaml") || strings.HasSuffix(lower, ".yml") {
		return true
	}

	return false
}

// SaveConfigs exports all configuration and tokenizer files for the model into dir.
//
// It copies all JSON files (e.g. config.json, generation_config.json, modules.json,
// tokenizer_config.json, tokenizer.json, etc.) as well as tokenizer vocabulary and model files
// (e.g. vocab.txt, merges.txt, *.model) directly from the underlying repository.
// It does NOT download or copy large weight files (*.safetensors, *.bin, etc.).
//
// Use SaveConfigs when you want to export configuration and tokenizer files to an arbitrary directory
// (for example, to inspect or distribute the model definition without checkpoints).
//
// If you are using GoMLX training checkpoints, prefer [Model.SaveToCheckpoint], which places
// these files in a dedicated "hf" sub-directory inside your checkpoint directory.
func (m *Model) SaveConfigs(dir string) error {
	resolvedDir, err := files.ReplaceTildeInDir(dir)
	if err != nil {
		resolvedDir = dir
	}
	resolvedDir = filepath.Clean(resolvedDir)

	if err := os.MkdirAll(resolvedDir, 0755); err != nil {
		return errors.Wrapf(err, "failed to create directory %q for model configs", resolvedDir)
	}

	copiedConfig := false
	copiedSTConfig := false
	copiedModules := false
	copiedPooling := false

	if m.Repo != nil {
		for fileName, err := range m.Repo.IterFileNames() {
			if err != nil {
				return errors.Wrapf(err, "failed listing files from repository %q", m.Repo.ID)
			}
			if !isConfigOrTokenizerFile(fileName) {
				continue
			}

			data, err := m.Repo.ReadFile(fileName)
			if err != nil {
				return errors.Wrapf(err, "failed reading %q from repository %q", fileName, m.Repo.ID)
			}

			destPath := filepath.Join(resolvedDir, filepath.FromSlash(fileName))
			if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
				return errors.Wrapf(err, "failed creating parent directory for %q", destPath)
			}
			if err := os.WriteFile(destPath, data, 0644); err != nil {
				return errors.Wrapf(err, "failed writing %q", destPath)
			}

			switch fileName {
			case "config.json":
				copiedConfig = true
			case "config_sentence_transformers.json":
				copiedSTConfig = true
			case "modules.json":
				copiedModules = true
			case "1_Pooling/config.json":
				copiedPooling = true
			}
		}
	}

	// Fallback serialization if files were not copied from repository (or if m.Repo is nil)
	saveJSON := func(relPath string, val any) error {
		destPath := filepath.Join(resolvedDir, filepath.FromSlash(relPath))
		if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
			return errors.Wrapf(err, "failed creating parent directory for %q", destPath)
		}
		data, err := json.MarshalIndent(val, "", "  ")
		if err != nil {
			return errors.Wrapf(err, "failed marshaling %s", relPath)
		}
		return os.WriteFile(destPath, data, 0644)
	}

	if !copiedConfig {
		if err := saveJSON("config.json", &m.Config); err != nil {
			return err
		}
	}
	if !copiedSTConfig && m.SentenceTransformerConfig != nil {
		if err := saveJSON("config_sentence_transformers.json", m.SentenceTransformerConfig); err != nil {
			return err
		}
	}
	if !copiedModules && len(m.Modules) > 0 {
		if err := saveJSON("modules.json", m.Modules); err != nil {
			return err
		}
	}
	if !copiedPooling && m.PoolingConfig != nil {
		if err := saveJSON("1_Pooling/config.json", m.PoolingConfig); err != nil {
			return err
		}
	}

	return nil
}

// SaveToCheckpoint saves the model configuration and tokenizer files into a dedicated "hf"
// sub-directory inside the provided checkpointDir.
//
// This keeps the Hugging Face configuration files cleanly separated from GoMLX checkpoint
// files (such as checkpoint.json and checkpoint-*.bin), preventing any naming conflicts or
// interference with checkpoint rotation.
//
// Models saved this way can be reloaded later with [LoadModelFromCheckpoint].
func (m *Model) SaveToCheckpoint(checkpointDir string) error {
	resolvedDir, err := files.ReplaceTildeInDir(checkpointDir)
	if err != nil {
		resolvedDir = checkpointDir
	}
	hfDir := filepath.Join(resolvedDir, CheckpointHFSubDir)
	return m.SaveConfigs(hfDir)
}

// LoadModelFromCheckpoint loads the model architecture and configuration from a GoMLX checkpoint directory.
//
// It looks for configurations in the "hf" sub-directory (as saved by [Model.SaveToCheckpoint]),
// falling back to checkpointDir itself if "hf" does not exist but "config.json" is present.
//
// Notice that this only loads the model configuration into a [*Model]. It does NOT load the variable
// weights into a [model.Store]. To load the weights, attach GoMLX's checkpoint handler to your store:
//
//	store := model.NewStore()
//	_, err := checkpoint.Build(store).Dir(checkpointDir).Done()
//
// There is no need to call [Model.LoadStore] when restoring from a checkpoint containing all weights.
func LoadModelFromCheckpoint(checkpointDir string) (*Model, error) {
	resolvedDir, err := files.ReplaceTildeInDir(checkpointDir)
	if err != nil {
		resolvedDir = checkpointDir
	}
	resolvedDir = filepath.Clean(resolvedDir)

	hfDir := filepath.Join(resolvedDir, CheckpointHFSubDir)
	if stat, err := os.Stat(hfDir); err == nil && stat.IsDir() {
		return LoadModel(hub.NewLocal(hfDir))
	}

	configPath := filepath.Join(resolvedDir, "config.json")
	if stat, err := os.Stat(configPath); err == nil && !stat.IsDir() {
		return LoadModel(hub.NewLocal(resolvedDir))
	}

	return nil, errors.Errorf("no Hugging Face configuration found in %q (checked %q and %q)", checkpointDir, hfDir, configPath)
}

// LoadModelFromFS loads the model architecture and configuration from an in-memory or embedded
// filesystem (fs.FS), such as Go's embed.FS, a zip archive (via archive/zip), or an in-memory filesystem.
//
// subDir specifies the relative directory inside fsys where the checkpoint or configurations reside.
// Use "" or "." if the checkpoint is at the root of fsys.
//
// It checks if subDir contains an "hf" sub-directory, falling back to subDir itself if "config.json"
// is found at the root of subDir.
//
// Example loading from an embedded zip archive:
//
//	//go:embed model_checkpoint.zip
//	var zipBytes []byte
//
//	zipReader, err := zip.NewReader(bytes.NewReader(zipBytes), int64(len(zipBytes)))
//	m, err := transformer.LoadModelFromFS(zipReader, "")
func LoadModelFromFS(fsys fs.FS, subDir string) (*Model, error) {
	cleanSubDir := strings.TrimPrefix(filepath.ToSlash(subDir), "/")
	if cleanSubDir == "." {
		cleanSubDir = ""
	}

	hfSubDir := path.Join(cleanSubDir, CheckpointHFSubDir)
	if _, err := fs.Stat(fsys, path.Join(hfSubDir, "config.json")); err == nil {
		return LoadModel(hub.NewEmbed(fsys, hfSubDir))
	}

	configPath := "config.json"
	if cleanSubDir != "" {
		configPath = path.Join(cleanSubDir, "config.json")
	}
	if _, err := fs.Stat(fsys, configPath); err == nil {
		return LoadModel(hub.NewEmbed(fsys, cleanSubDir))
	}

	return nil, errors.Errorf("no Hugging Face configuration found in filesystem (checked %q and %q)", path.Join(hfSubDir, "config.json"), configPath)
}
