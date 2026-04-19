package transformer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/safetensors"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/pkg/errors"
)

// Model holds all the configuration loaded about the model.
type Model struct {
	Repo *hub.Repo

	Config                    Config
	SentenceTransformerConfig *SentenceTransformerConfig
	Modules                   []ModuleConfig
	PoolingConfig             *PoolingConfig

	// useCausalMask: The KaLM paper says that the model is trained without a causal mask, but HuggingFace transformer
	// leaves that on by default. We default to off, but we make it configurable via WithCausalMask.
	useCausalMask bool

	// Cached parameter count and bytes for Description
	totalParameters *int64
	totalBytes      *int64

	tokenizer tokenizers.Tokenizer
}

// LoadModel loads the configurations into the Model struct.
// It loads config.json as mandatory, and gracefully handles the optional presence
// of sentence_transformer and pooling configurations.
func LoadModel(repo *hub.Repo) (*Model, error) {
	m := &Model{
		Repo: repo,
	}

	loadFile := func(filename string, v any) (bool, error) {
		path, err := repo.DownloadFile(filename)
		if err != nil {
			// File likely doesn't exist or network error. We treat as missing.
			return false, nil
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return true, fmt.Errorf("failed to read %s: %w", filename, err)
		}
		if err := json.Unmarshal(b, v); err != nil {
			return true, fmt.Errorf("failed to unmarshal %s: %w", filename, err)
		}
		return true, nil
	}

	// config.json is required
	if ok, err := loadFile("config.json", &m.Config); err != nil || !ok {
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("config.json is missing or failed to download from repo")
	}

	// Optional files
	stc := &SentenceTransformerConfig{}
	if ok, _ := loadFile("config_sentence_transformers.json", stc); ok {
		m.SentenceTransformerConfig = stc
	}

	var mods []ModuleConfig
	if ok, _ := loadFile("modules.json", &mods); ok {
		m.Modules = mods
	}

	pc := &PoolingConfig{}
	if ok, _ := loadFile("1_Pooling/config.json", pc); ok {
		m.PoolingConfig = pc
	}

	return m, nil
}

// LoadContext uses models/safetensors to load the variables of the model into a context.
//
// If a backend is provided (not nil), the variables are immediately loaded into the backend
// device #0, saving host memory space or accelerating the loading in some cases.
//
// For distributed execution, better to leave backend and nil, and let the executor decide
// on which devices to place the variables.
func (m *Model) LoadContext(backend compute.Backend, ctx *context.Context) error {
	var totalParams int64
	var totalBytes int64

	isBert := m.Config.ModelType == "bert" || (len(m.Config.Architectures) > 0 && m.Config.Architectures[0] == "BertModel")

	for tensorAndName, err := range safetensors.IterTensorsFromRepo(backend, m.Repo) {
		if err != nil {
			return errors.WithMessagef(err, "failed loading variables of models %q", m.Repo.ID)
		}
		scopePath, varName, ok := mapTensorName(tensorAndName.Name)
		if !ok {
			name := strings.TrimPrefix(tensorAndName.Name, "model.")
			if isBert && (name == "embeddings.position_ids" || strings.HasPrefix(name, "pooler.")) {
				continue
			}
			fmt.Printf("Skipping unmapped tensor: %s\n", tensorAndName.Name)
			continue
		}

		tensorToLoad := tensorAndName.Tensor

		// Track size
		shape := tensorToLoad.Shape()
		totalParams += int64(shape.Size())
		totalBytes += int64(shape.ByteSize())

		scopeCtx := ctx
		for _, scope := range scopePath {
			scopeCtx = scopeCtx.In(scope)
		}

		scopeCtx.VariableWithValue(varName, tensorToLoad)
	}

	m.totalParameters = &totalParams
	m.totalBytes = &totalBytes
	return nil
}

// Description returns a string summarizing the model architecture, size, and loaded configurations.
func (m *Model) Description() string {
	var sb strings.Builder

	sb.WriteString(fmt.Sprintf("Model: %s\n", m.Repo.ID))
	sb.WriteString(fmt.Sprintf("Architecture: %s\n", strings.Join(m.Config.Architectures, ", ")))
	sb.WriteString(fmt.Sprintf("Layers: %d\n", m.Config.NumHiddenLayers))
	sb.WriteString(fmt.Sprintf("Hidden Size: %d\n", m.Config.HiddenSize))
	sb.WriteString(fmt.Sprintf("Attention Heads: %d\n", m.Config.NumAttentionHeads))
	if m.Config.NumKeyValueHeads > 0 {
		sb.WriteString(fmt.Sprintf("KV Heads: %d\n", m.Config.NumKeyValueHeads))
	}

	// Calculate and summarize total parameters/bytes if we haven't already and safetensors index is available
	if m.totalParameters == nil {
		m.estimateSizeFromIndex()
	}
	if m.totalParameters != nil {
		sb.WriteString(fmt.Sprintf("Total Parameters: %.1fM\n", float64(*m.totalParameters)/1e6))
	}
	if m.totalBytes != nil {
		sb.WriteString(fmt.Sprintf("Total Bytes: %.2f GB\n", float64(*m.totalBytes)/(1024*1024*1024)))
	}

	sb.WriteString("Configurations:\n")
	sb.WriteString(fmt.Sprintf(" - config.json: loaded, vocab_size=%d\n", m.Config.VocabSize))
	if m.SentenceTransformerConfig != nil {
		sb.WriteString(" - config_sentence_transformers.json: loaded\n")
	}
	if len(m.Modules) > 0 {
		sb.WriteString(fmt.Sprintf(" - modules.json: loaded (%d modules)\n", len(m.Modules)))
	}
	if m.PoolingConfig != nil {
		sb.WriteString(" - 1_Pooling/config.json: loaded\n")
	}
	return sb.String()
}

func (m *Model) estimateSizeFromIndex() {
	// Try to see if there is an index with total_size metadata
	path, err := m.Repo.DownloadFile("model.safetensors.index.json")
	if err == nil {
		b, err := os.ReadFile(path)
		if err == nil {
			var index struct {
				Metadata struct {
					TotalSize int64 `json:"total_size"`
				} `json:"metadata"`
			}
			if json.Unmarshal(b, &index) == nil && index.Metadata.TotalSize > 0 {
				size := index.Metadata.TotalSize
				m.totalBytes = &size
			}
		}
	}
}

// mapTensorName maps safetensors tensor names to GoMLX context variable names
// HF format (e.g. LLaMA/Gemma): layers.{N}.{component}.weight
// GoMLX format: layer_{N}/{component}/rms_norm/scale, layer_{N}/self_attn/..., etc.
func mapTensorName(safetensorsName string) (scopePath []string, varName string, ok bool) {
	safetensorsName = strings.TrimPrefix(safetensorsName, "model.")

	// BERT specific mappings
	if strings.HasPrefix(safetensorsName, "embeddings.") {
		name := strings.TrimPrefix(safetensorsName, "embeddings.")
		switch name {
		case "word_embeddings.weight":
			return []string{"token_embed"}, "embeddings", true
		case "position_embeddings.weight":
			return []string{"pos_embed"}, "embeddings", true
		case "token_type_embeddings.weight":
			return []string{"token_type_embed"}, "embeddings", true
		case "LayerNorm.weight":
			return []string{"embed_norm", "layer_normalization"}, "gain", true
		case "LayerNorm.bias":
			return []string{"embed_norm", "layer_normalization"}, "offset", true
		}
	}

	if strings.HasPrefix(safetensorsName, "encoder.layer.") {
		parts := strings.Split(safetensorsName, ".")
		// parts: [encoder layer {N} component ...]
		if len(parts) >= 4 {
			var layerNum int
			fmt.Sscanf(parts[2], "%d", &layerNum)
			layerScope := fmt.Sprintf("layer_%d", layerNum)
			component := strings.Join(parts[3:], ".")

			switch component {
			case "attention.self.query.weight":
				return []string{layerScope, "attn", "MultiHeadAttention", "query", "dense"}, "weights", true
			case "attention.self.query.bias":
				return []string{layerScope, "attn", "MultiHeadAttention", "query", "dense"}, "biases", true
			case "attention.self.key.weight":
				return []string{layerScope, "attn", "MultiHeadAttention", "key", "dense"}, "weights", true
			case "attention.self.key.bias":
				return []string{layerScope, "attn", "MultiHeadAttention", "key", "dense"}, "biases", true
			case "attention.self.value.weight":
				return []string{layerScope, "attn", "MultiHeadAttention", "value", "dense"}, "weights", true
			case "attention.self.value.bias":
				return []string{layerScope, "attn", "MultiHeadAttention", "value", "dense"}, "biases", true
			case "attention.output.dense.weight":
				return []string{layerScope, "attn", "MultiHeadAttention", "output", "dense"}, "weights", true
			case "attention.output.dense.bias":
				return []string{layerScope, "attn", "MultiHeadAttention", "output", "dense"}, "biases", true
			case "attention.output.LayerNorm.weight":
				return []string{layerScope, "norm1", "layer_normalization"}, "gain", true
			case "attention.output.LayerNorm.bias":
				return []string{layerScope, "norm1", "layer_normalization"}, "offset", true

			case "intermediate.dense.weight":
				return []string{layerScope, "ff1", "dense"}, "weights", true
			case "intermediate.dense.bias":
				return []string{layerScope, "ff1", "dense"}, "biases", true
			case "output.dense.weight":
				return []string{layerScope, "ff2", "dense"}, "weights", true
			case "output.dense.bias":
				return []string{layerScope, "ff2", "dense"}, "biases", true
			case "output.LayerNorm.weight":
				return []string{layerScope, "norm2", "layer_normalization"}, "gain", true
			case "output.LayerNorm.bias":
				return []string{layerScope, "norm2", "layer_normalization"}, "offset", true
			}
		}
	}

	if safetensorsName == "embed_tokens.weight" {
		return []string{"token_embed"}, "embeddings", true
	}
	if safetensorsName == "norm.weight" {
		return []string{"final_norm", "rms_norm"}, "scale", true
	}

	var layerNum int
	var component, sub1, sub2 string

	// Parse layer-specific components
	if n, err := fmt.Sscanf(safetensorsName, "layers.%d.%s", &layerNum, &component); n == 2 && err == nil {
		layerScope := fmt.Sprintf("layer_%d", layerNum)

		switch component {
		case "input_layernorm.weight":
			return []string{layerScope, "input_norm", "rms_norm"}, "scale", true
		case "post_attention_layernorm.weight":
			return []string{layerScope, "post_attention_norm", "rms_norm"}, "scale", true
		case "pre_feedforward_layernorm.weight":
			return []string{layerScope, "pre_feedforward_norm", "rms_norm"}, "scale", true
		case "post_feedforward_layernorm.weight":
			return []string{layerScope, "post_feedforward_norm", "rms_norm"}, "scale", true
		}
	}

	// Parse attention blocks
	if n, err := fmt.Sscanf(safetensorsName, "layers.%d.self_attn.%s", &layerNum, &sub1); n == 2 && err == nil {
		layerScope := fmt.Sprintf("layer_%d", layerNum)
		switch sub1 {
		case "q_proj.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "query", "dense"}, "weights", true
		case "k_proj.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "key", "dense"}, "weights", true
		case "v_proj.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "value", "dense"}, "weights", true
		case "o_proj.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "output", "dense"}, "weights", true
		case "q_norm.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "query", "rms_norm"}, "scale", true
		case "k_norm.weight":
			return []string{layerScope, "self_attn", "MultiHeadAttention", "key", "rms_norm"}, "scale", true
		}
	}

	// Parse MLP blocks
	if n, err := fmt.Sscanf(safetensorsName, "layers.%d.mlp.%s", &layerNum, &sub2); n == 2 && err == nil {
		layerScope := fmt.Sprintf("layer_%d", layerNum)
		switch sub2 {
		case "gate_proj.weight":
			return []string{layerScope, "mlp", "gate_proj", "dense"}, "weights", true
		case "up_proj.weight":
			return []string{layerScope, "mlp", "up_proj", "dense"}, "weights", true
		case "down_proj.weight":
			return []string{layerScope, "mlp", "down_proj", "dense"}, "weights", true
		}
	}

	return nil, "", false
}

// SetTokenizer sets the tokenizer of the Model.
func (m *Model) SetTokenizer(tok tokenizers.Tokenizer) {
	m.tokenizer = tok
}

// GetTokenizer returns the tokenizer for the model if it exists, or if it hasn't been set yet, attempts to create the default tokenizer for the repo.
func (m *Model) GetTokenizer() (tokenizers.Tokenizer, error) {
	if m.tokenizer != nil {
		return m.tokenizer, nil
	}
	var err error
	m.tokenizer, err = tokenizers.New(m.Repo)
	return m.tokenizer, err
}

// BuildPrompt builds the full sentence prompt, based on a promptName, an index to the
// list of prompts in the SentenceTransformerConfig.
// If the code is not found, it attempts to use the default one. If a default one is
// not defined, it returns the original sentence.
func (m *Model) BuildPrompt(sentence, promptName string) string {
	if m.SentenceTransformerConfig == nil || m.SentenceTransformerConfig.Prompts == nil {
		return sentence
	}
	if promptName == "" {
		promptName = m.SentenceTransformerConfig.DefaultPromptName
	}
	if promptName == "" {
		return sentence
	}
	if taskPrompt, ok := m.SentenceTransformerConfig.Prompts[promptName]; ok {
		return taskPrompt + sentence
	}
	return sentence
}

// BuildQueryPrompt builds the full query prompt, based on a promptName.
// It's exactly like BuildPrompt but if a default prompt doesn't exist it uses
// "Instruct: Given a query, retrieve documents that answer the query \nQuery: " as
// a prompt prefix.
func (m *Model) BuildQueryPrompt(query, promptName string) string {
	if m.SentenceTransformerConfig == nil || m.SentenceTransformerConfig.Prompts == nil {
		return query
	}
	var prompt string
	if promptName == "" {
		promptName = m.SentenceTransformerConfig.DefaultPromptName
	}
	if promptName != "" {
		prompt = m.SentenceTransformerConfig.Prompts[promptName]
	}
	if prompt == "" {
		return query
	}
	return prompt + query
}

// RegisteredPromptTasks returns a list of all task codes for which prompts are registered.
func (m *Model) RegisteredPromptTasks() []string {
	if m.SentenceTransformerConfig == nil || m.SentenceTransformerConfig.Prompts == nil {
		return nil
	}
	return xslices.SortedKeys(m.SentenceTransformerConfig.Prompts)
}

// GetTaskPrompt returns the prompt string for the given task code.
// Returns an empty string if the task code is not found or no prompts are registered.
func (m *Model) GetTaskPrompt(taskCode string) string {
	if m.SentenceTransformerConfig == nil || m.SentenceTransformerConfig.Prompts == nil {
		return ""
	}
	if taskCode == "" {
		taskCode = m.SentenceTransformerConfig.DefaultPromptName
	}
	if taskCode == "" {
		return ""
	}
	if promptStr, ok := m.SentenceTransformerConfig.Prompts[taskCode]; ok {
		return promptStr
	}
	return ""
}
