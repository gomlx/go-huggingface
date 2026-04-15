package hftokenizer

import (
	"encoding/json"
	"regexp"
	"github.com/gomlx/go-huggingface/tokenizers/api"
)

// TokenizerJSON represents the structure of HuggingFace's tokenizer.json file.
type TokenizerJSON struct {
	Version       string          `json:"version"`
	Truncation    json.RawMessage `json:"truncation"`
	Padding       json.RawMessage `json:"padding"`
	AddedTokens   []AddedToken    `json:"added_tokens"`
	Normalizer    *Normalizer     `json:"normalizer"`
	PreTokenizer  *PreTokenizer   `json:"pre_tokenizer"`
	PostProcessor *PostProcessor  `json:"post_processor"`
	Decoder       *Decoder        `json:"decoder"`
	Model         Model           `json:"model"`
}

// AddedToken represents a special token added to the vocabulary.
type AddedToken struct {
	ID         int    `json:"id"`
	Content    string `json:"content"`
	SingleWord bool   `json:"single_word"`
	Lstrip     bool   `json:"lstrip"`
	Rstrip     bool   `json:"rstrip"`
	Normalized bool   `json:"normalized"`
	Special    bool   `json:"special"`
}

// Normalizer represents the normalizer configuration.
type Normalizer struct {
	Type               string       `json:"type"`
	Lowercase          bool         `json:"lowercase"`
	CleanText          bool         `json:"clean_text"`
	HandleChineseChars bool         `json:"handle_chinese_chars"`
	StripAccents       *bool        `json:"strip_accents"`
	Normalizer         *Normalizer  `json:"normalizer"`
	Pattern            *Pattern     `json:"pattern"`
	Normalizers        []Normalizer `json:"normalizers"`
}

// Pattern for regex-based operations.
type Pattern struct {
	Regex  string `json:"Regex,omitempty"`
	String string `json:"String,omitempty"`
}

// PreTokenizer represents the pre-tokenizer configuration.
type PreTokenizer struct {
	Type           string         `json:"type"`
	AddPrefixSpace bool           `json:"add_prefix_space"`
	PreTokenizers  []PreTokenizer `json:"pretokenizers"`
	Pattern        *Pattern       `json:"pattern"`
	Behavior       string         `json:"behavior"`
	Invert         bool           `json:"invert"`
	Replacement    string         `json:"replacement"`
	PrependScheme  string         `json:"prepend_scheme"`
	Split          *bool          `json:"split"`
}

// PostProcessor represents the post-processor configuration.
// Supports TemplateProcessing, BertProcessing, and RobertaProcessing types.
type PostProcessor struct {
	Type          string                          `json:"type"`
	Single        []PostProcItem                  `json:"single"`
	Pair          []PostProcItem                  `json:"pair"`
	SpecialTokens map[string]PostProcSpecialToken `json:"special_tokens"`
	// Sep and Cls are used by BertProcessing and RobertaProcessing.
	// Format in JSON: ["[SEP]", 102] — a [token_string, token_id] tuple.
	Sep json.RawMessage `json:"sep"`
	Cls json.RawMessage `json:"cls"`
}

// PostProcItem is a tagged union item in TemplateProcessing templates.
// Exactly one of SpecialToken or Sequence is non-nil.
type PostProcItem struct {
	SpecialToken *struct {
		ID     string `json:"id"`
		TypeID int    `json:"type_id"`
	} `json:"SpecialToken,omitempty"`
	Sequence *struct {
		ID     string `json:"id"`
		TypeID int    `json:"type_id"`
	} `json:"Sequence,omitempty"`
}

// PostProcSpecialToken defines a special token for TemplateProcessing.
type PostProcSpecialToken struct {
	ID     string   `json:"id"`
	IDs    []int    `json:"ids"`
	Tokens []string `json:"tokens"`
}

// Decoder represents the decoder configuration.
type Decoder struct {
	Type          string         `json:"type"`
	Prefix        string         `json:"prefix"`
	Suffix        string         `json:"suffix"`
	Decoders      []*Decoder     `json:"decoders"`
	Pattern       *Pattern       `json:"pattern"`
	compiled      *regexp.Regexp // Cached compiled pattern, ignored by JSON
	Content       string         `json:"content"`
	Replacement   string         `json:"replacement"`
	PrependScheme string         `json:"prepend_scheme"`
	Split         bool           `json:"split"`
}

// Model represents the tokenizer model (WordPiece, BPE, or Unigram).
type Model struct {
	Type                    string         `json:"type"`
	Vocab                   map[string]int `json:"-"` // Custom unmarshaling handles both map and array formats
	Merges                  []string       `json:"-"` // Custom unmarshaling handles both string and array formats
	UnkToken                string         `json:"unk_token"`
	ContinuingSubwordPrefix string         `json:"continuing_subword_prefix"`
	MaxInputCharsPerWord    int            `json:"max_input_chars_per_word"`
	FuseUnk                 bool           `json:"fuse_unk"`
	ByteFallback            bool           `json:"byte_fallback"`
	Dropout                 *float64       `json:"dropout"`
	EndOfWordSuffix         string         `json:"end_of_word_suffix"`
}

// Tokenizer implements the api.Tokenizer interface for HuggingFace tokenizer.json files.
type Tokenizer struct {
	config     *api.Config
	tokenizer  *TokenizerJSON
	idToToken  map[int]string
	mergeRanks map[string]int // For BPE: maps "token1 token2" to merge priority

	// Special token IDs
	unkID  int
	padID  int
	bosID  int
	eosID  int
	clsID  int
	sepID  int
	maskID int

	// Added tokens lookup (content -> id)
	addedTokens map[string]int

	options api.EncodeOptions

	// addedTokensSorted lists added tokens sorted longest-first for greedy
	// matching when splitting input text. Derived from addedTokens at construction.
	addedTokensSorted []addedTokenEntry
}
