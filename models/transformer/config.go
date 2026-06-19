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
	GlobalHeadDim     int      `json:"global_head_dim"`
	IntermediateSize  int      `json:"intermediate_size"`
	NumKeyValueHeads  int      `json:"num_key_value_heads"`
	NumKVSharedLayers int      `json:"num_kv_shared_layers"`
	RMSNormEps            float64  `json:"rms_norm_eps"`
	LayerNormEps          float64  `json:"layer_norm_eps"`
	AttentionLogitCap     float64  `json:"attention_logit_cap"`
	FinalLogitSoftcapping float64  `json:"final_logit_softcapping"`

	HiddenSizePerLayerInput int      `json:"hidden_size_per_layer_input"`
	VocabSizePerLayerInput  int      `json:"vocab_size_per_layer_input"`

	// RoPE Positional Embedder
	RoPETheta      float64               `json:"rope_theta"`
	RoPEScaling    RoPEScaling           `json:"rope_scaling"`
	RoPEParameters map[string]RoPEParams `json:"rope_parameters"`

	// RoPELocalBaseFreq is the Theta used by sliding attention layers (or so the AI says).
	RoPELocalBaseFreq float64 `json:"rope_local_base_freq"`

	HiddenActivation      string `json:"hidden_activation"`
	HiddenAct             string `json:"hidden_act"`
	MaxPositionEmbeddings int    `json:"max_position_embeddings"`
	ModelType             string `json:"model_type"`
	TorchDtype            string `json:"torch_dtype"`
	DType                 string `json:"dtype"`
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

type RoPEParams struct {
	RopeTheta           float64 `json:"rope_theta"`
	RopeType            string  `json:"rope_type"`
	PartialRotaryFactor float64 `json:"partial_rotary_factor"`
}

func (c *Config) UnmarshalJSON(data []byte) error {
	type wrapper Config
	if err := json.Unmarshal(data, (*wrapper)(c)); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &c.Extra); err != nil {
		return err
	}
	if textConfigRaw, ok := c.Extra["text_config"]; ok {
		textConfigBytes, err := json.Marshal(textConfigRaw)
		if err == nil {
			_ = json.Unmarshal(textConfigBytes, (*wrapper)(c))
		}
	}
	return nil
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
