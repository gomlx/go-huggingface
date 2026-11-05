package api

import (
	"encoding/json"
	"os"

	"github.com/pkg/errors"
)

// Config struct to hold HuggingFace's tokenizer_config.json contents.
// There is no formal schema for this file, but these are some common fields that may be of use.
// Specific tokenizer classes are free to implement additional features as they see fit.
//
// The extra field ConfigFile holds the path to the file with the full config.
type Config struct {
	ConfigFile     string
	TokenizerClass string `json:"tokenizer_class"`

	ChatTemplate           string `json:"chat_template"`
	UseDefaultSystemPrompt bool   `json:"use_default_system_prompt"`

	ModelMaxLength float64        `json:"model_max_length"`
	MaxLength      float64        `json:"max_length"`
	SpModelKwargs  map[string]any `json:"sp_model_kwargs"`

	ClsToken  TokensDecoder `json:"cls_token"`
	UnkToken  TokensDecoder `json:"unk_token"`
	SepToken  TokensDecoder `json:"sep_token"`
	MaskToken TokensDecoder `json:"mask_token"`
	BosToken  TokensDecoder `json:"bos_token"`
	EosToken  TokensDecoder `json:"eos_token"`
	PadToken  TokensDecoder `json:"pad_token"`

	AddBosToken             bool                  `json:"add_bos_token"`
	AddEosToken             bool                  `json:"add_eos_token"`
	AddedTokensDecoder      map[int]TokensDecoder `json:"added_tokens_decoder"`
	AdditionalSpecialTokens []string              `json:"additional_special_tokens"`

	DoLowerCase                bool `json:"do_lower_case"`
	CleanUpTokenizationSpaces  bool `json:"clean_up_tokenization_spaces"`
	SpacesBetweenSpecialTokens bool `json:"spaces_between_special_tokens"`

	TokenizeChineseChars bool   `json:"tokenize_chinese_chars"`
	StripAccents         any    `json:"strip_accents"`
	NameOrPath           string `json:"name_or_path"`
	DoBasicTokenize      bool   `json:"do_basic_tokenize"`
	NeverSplit           any    `json:"never_split"`

	Stride             int    `json:"stride"`
	TruncationSide     string `json:"truncation_side"`
	TruncationStrategy string `json:"truncation_strategy"`
}

// ParseConfigFile parses the given file (holding a tokenizer_config.json file) into a Config structure.
func ParseConfigFile(filePath string) (*Config, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read file %q", filePath)
	}
	config, err := ParseConfigContent(content)
	if err != nil {
		return nil, errors.WithMessagef(err, "read from file %q", filePath)
	}
	config.ConfigFile = filePath
	return config, nil
}

// ParseConfigContent parses the given json content (of a tokenizer_config.json file) into a Config structure.
func ParseConfigContent(jsonContent []byte) (*Config, error) {
	config := &Config{}
	err := json.Unmarshal(jsonContent, config)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to parse tokenizer_config json content")
	}
	return config, nil
}

// TokensDecoder holds the properties of a special or added token in tokenizer_config.json
// (such as bos_token, eos_token, or entries in added_tokens_decoder).
// In JSON, it can be represented either as a plain string (e.g., "<s>") or as an
// AddedToken object (e.g., {"content": "<s>", "lstrip": false, ...}).
type TokensDecoder struct {
	Type       string `json:"__type,omitempty"`
	Content    string `json:"content"`
	Lstrip     bool   `json:"lstrip"`
	Normalized bool   `json:"normalized"`
	Rstrip     bool   `json:"rstrip"`
	SingleWord bool   `json:"single_word"`
	Special    bool   `json:"special"`
}

// String returns the token's string content.
func (t TokensDecoder) String() string {
	return t.Content
}

// UnmarshalJSON unmarshals a TokensDecoder from either a JSON string or a JSON object with a "content" field.
func (t *TokensDecoder) UnmarshalJSON(data []byte) error {
	// Try plain string first (also handles JSON null as "").
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		*t = TokensDecoder{Content: s}
		return nil
	}
	// Try object with "content" and optional AddedToken fields.
	type tokensDecoderObj TokensDecoder
	var obj tokensDecoderObj
	if err := json.Unmarshal(data, &obj); err != nil {
		return err
	}
	*t = TokensDecoder(obj)
	return nil
}
