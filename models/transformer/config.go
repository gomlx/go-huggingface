package transformer

import (
	"encoding/json"
)

// Config represents standard HuggingFace config.json.
type Config struct {
	Architectures     []string `json:"architectures"`
	HiddenSize        int      `json:"hidden_size"`
	NumHiddenLayers   int      `json:"num_hidden_layers"`
	NumAttentionHeads int      `json:"num_attention_heads"`
	HeadDim           int      `json:"head_dim"`
	IntermediateSize  int      `json:"intermediate_size"`
	NumKeyValueHeads  int      `json:"num_key_value_heads"`
	RMSNormEps        float64  `json:"rms_norm_eps"`

	// RoPE Positional Embedder
	RoPETheta   float64     `json:"rope_theta"`
	RoPEScaling RoPEScaling `json:"rope_scaling"`

	// RoPELocalBaseFreq is the Theta used by sliding attention layers (or so the AI says).
	RoPELocalBaseFreq float64 `json:"rope_local_base_freq"`

	HiddenActivation      string `json:"hidden_activation"`
	MaxPositionEmbeddings int    `json:"max_position_embeddings"`
	ModelType             string `json:"model_type"`
	TorchDtype            string `json:"torch_dtype"`
	VocabSize             int    `json:"vocab_size"`
	PadTokenID            int    `json:"pad_token_id"`

	// SlidingWindow is the size of the sliding window for sliding attention layers.
	SlidingWindow int `json:"sliding_window"`

	// LayerTypes: known values are "full_attention", "sliding_attention"
	LayerTypes []string       `json:"layer_types"`
	Extra      map[string]any `json:"-"`
}

type RoPEScaling struct {
	Factor float64 `json:"factor"`

	// Type: "linear"
	Type string `json:"rope_type"`
}

func (c *Config) UnmarshalJSON(data []byte) error {
	type wrapper Config
	if err := json.Unmarshal(data, (*wrapper)(c)); err != nil {
		return err
	}
	return json.Unmarshal(data, &c.Extra)
}

const DefaultQueryPrompt = "Instruct: Given a query, retrieve documents that answer the query \nQuery: "

// SentenceTransformerConfig represents config_sentence_transformers.json (optional)
type SentenceTransformerConfig struct {
	Extra             map[string]any    `json:"-"`
	Prompts           map[string]string `json:"prompts"`
	DefaultPromptName string            `json:"default_prompt_name"`
	SimilarityFnName  string            `json:"similarity_fn_name"`
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
