package transformer

import (
	"encoding/json"
)

// Config represents standard HuggingFace config.json.
type Config struct {
	Architectures         []string       `json:"architectures"`
	HiddenSize            int            `json:"hidden_size"`
	NumHiddenLayers       int            `json:"num_hidden_layers"`
	NumAttentionHeads     int            `json:"num_attention_heads"`
	HeadDim               int            `json:"head_dim"`
	IntermediateSize      int            `json:"intermediate_size"`
	NumKeyValueHeads      int            `json:"num_key_value_heads"`
	RMSNormEps            float64        `json:"rms_norm_eps"`
	RopeTheta             float64        `json:"rope_theta"`
	HiddenActivation      string         `json:"hidden_activation"`
	MaxPositionEmbeddings int            `json:"max_position_embeddings"`
	ModelType             string         `json:"model_type"`
	TorchDtype            string         `json:"torch_dtype"`
	VocabSize             int            `json:"vocab_size"`
	Extra                 map[string]any `json:"-"`
}

func (c *Config) UnmarshalJSON(data []byte) error {
	type wrapper Config
	if err := json.Unmarshal(data, (*wrapper)(c)); err != nil {
		return err
	}
	return json.Unmarshal(data, &c.Extra)
}

// SentenceTransformerConfig represents config_sentence_transformers.json (optional)
type SentenceTransformerConfig struct {
	Extra map[string]any `json:"-"`
}

func (c *SentenceTransformerConfig) UnmarshalJSON(data []byte) error {
	type wrapper SentenceTransformerConfig
	if err := json.Unmarshal(data, (*wrapper)(c)); err != nil {
		return err
	}
	return json.Unmarshal(data, &c.Extra)
}

// ModuleConfig represents an entry in modules.json
type ModuleConfig struct {
	Idx   int            `json:"idx"`
	Name  string         `json:"name"`
	Path  string         `json:"path"`
	Type  string         `json:"type"`
	Extra map[string]any `json:"-"`
}

func (m *ModuleConfig) UnmarshalJSON(data []byte) error {
	type wrapper ModuleConfig
	if err := json.Unmarshal(data, (*wrapper)(m)); err != nil {
		return err
	}
	return json.Unmarshal(data, &m.Extra)
}

// TaskPromptsConfig represents task_prompts.json
type TaskPromptsConfig struct {
	Extra map[string]any `json:"-"`
}

func (c *TaskPromptsConfig) UnmarshalJSON(data []byte) error {
	type wrapper TaskPromptsConfig
	if err := json.Unmarshal(data, (*wrapper)(c)); err != nil {
		return err
	}
	return json.Unmarshal(data, &c.Extra)
}

// PoolingConfig represents 1_Pooling/config.json
type PoolingConfig struct {
	PoolingModeMeanTokens bool           `json:"pooling_mode_mean_tokens"`
	PoolingModeClsToken   bool           `json:"pooling_mode_cls_token"`
	PoolingModeMaxTokens  bool           `json:"pooling_mode_max_tokens"`
	PoolingModeLastToken  bool           `json:"pooling_mode_lasttoken"`
	Extra                 map[string]any `json:"-"`
}

func (c *PoolingConfig) UnmarshalJSON(data []byte) error {
	type wrapper PoolingConfig
	if err := json.Unmarshal(data, (*wrapper)(c)); err != nil {
		return err
	}
	return json.Unmarshal(data, &c.Extra)
}
