package api

import (
	"encoding/json"
	"github.com/pkg/errors"
	"os"
)

type TokensDecoder struct {
	Content    string `json:"content"`
	Lstrip     bool   `json:"lstrip"`
	Normalized bool   `json:"normalized"`
	Rstrip     bool   `json:"rstrip"`
	SingleWord bool   `json:"single_word"`
	Special    bool   `json:"special"`
}

// FlexToken handles HuggingFace token fields that can be either a plain string
// (e.g., "<s>") or an object with a "content" field (e.g., {"content": "<s>", ...}).
type FlexToken string

func (t *FlexToken) UnmarshalJSON(data []byte) error {
	// Try string first.
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		*t = FlexToken(s)
		return nil
	}
	// Try object with "content" field.
	var obj struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(data, &obj); err != nil {
		return err
	}
	*t = FlexToken(obj.Content)
	return nil
}

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

	ClsToken  FlexToken `json:"cls_token"`
	UnkToken  FlexToken `json:"unk_token"`
	SepToken  FlexToken `json:"sep_token"`
	MaskToken FlexToken `json:"mask_token"`
	BosToken  FlexToken `json:"bos_token"`
	EosToken  FlexToken `json:"eos_token"`
	PadToken  FlexToken `json:"pad_token"`

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
