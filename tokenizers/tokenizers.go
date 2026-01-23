// Package tokenizers creates tokenizers from HuggingFace models.
//
// Given a HuggingFace repository (see hub.New to create one), tokenizers will use its "tokenizer_config.json"
// and "tokenizer.json" to instantiate a Tokenizer.
package tokenizers

import (
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/go-huggingface/tokenizers/hftokenizer"
	"github.com/gomlx/go-huggingface/tokenizers/sentencepiece"
	"github.com/pkg/errors"

	// Blank import.
	_ "github.com/gomlx/go-huggingface/tokenizers/sentencepiece"
)

// Tokenizer interface allows one convert test to "tokens" (integer ids) and back.
//
// It also allows mapping of special tokens: tokens with a comman semantic (like padding) but that
// may map to different ids (int) for different tokenizers.
type Tokenizer = api.Tokenizer

// TokenizerWithSpans extends Tokenizer with span tracking capability.
// This is useful for token classification tasks (NER, chunking) where you need
// to map token predictions back to byte positions in the original text.
type TokenizerWithSpans = api.TokenizerWithSpans

// TokenSpan represents the byte span of a token in the original text.
type TokenSpan = api.TokenSpan

// EncodingResult contains tokens with their spans.
type EncodingResult = api.EncodingResult

// SpecialToken is an enum of commonly used special tokens.
type SpecialToken = api.Tokenizer

const (
	TokBeginningOfSentence = api.TokBeginningOfSentence
	TokEndOfSentence       = api.TokEndOfSentence
	TokUnknown             = api.TokUnknown
	TokPad                 = api.TokPad
	TokMask                = api.TokMask
	TokClassification      = api.TokClassification
	TokSpecialTokensCount  = api.TokSpecialTokensCount
)

// New creates a new tokenizer from the given HuggingFace repo (see hub.New).
//
// Currently, it only supports "SentencePiece" encoders, and it attempts to download details from
// the repo files "tokenizer_config.json" and "tokenizer.json".
//
// If it fails to load those files, or create a tokenizer, it returns an error.
func New(repo *hub.Repo) (Tokenizer, error) {
	err := repo.DownloadInfo(false)
	if err != nil {
		return nil, err
	}

	config, err := GetConfig(repo)
	if err != nil {
		return nil, err
	}

	constructor, found := registerOfClasses[config.TokenizerClass]
	if !found {
		return nil, errors.Errorf("unknown tokenizer class %q", config.TokenizerClass)
	}
	return constructor(config, repo)
}

// GetConfig returns the parsed "tokenizer_config.json" Config object for the repo.
func GetConfig(repo *hub.Repo) (*api.Config, error) {
	err := repo.DownloadInfo(false)
	if err != nil {
		return nil, err
	}
	localConfigFile, err := repo.DownloadFile("tokenizer_config.json")
	if err != nil {
		return nil, err
	}
	config, err := api.ParseConfigFile(localConfigFile) // tokenizer_config.json
	if err != nil {
		return nil, err
	}
	return config, nil
}

// Config struct to hold HuggingFace's tokenizer_config.json contents.
// There is no formal schema for this file, but these are some common fields that may be of use.
// Specific tokenizer classes are free to implement additional features as they see fit.
//
// The extra field ConfigFile holds the path to the file with the full config.
type Config = api.Config

// TokenizerConstructor is used by Tokenizer implementations to provide implementations for different
// tokenizer classes.
type TokenizerConstructor func(config *api.Config, repo *hub.Repo) (api.Tokenizer, error)

// RegisterTokenizerClass used by Tokenizer implementations.
func RegisterTokenizerClass(name string, constructor TokenizerConstructor) {
	registerOfClasses[name] = constructor
}

var (
	registerOfClasses = make(map[string]TokenizerConstructor)
)

func init() {
	// Initialize sentencepiece tokenizer classes.
	RegisterTokenizerClass("GemmaTokenizer", sentencepiece.New)

	// Initialize HuggingFace tokenizer classes (WordPiece/BPE based).
	// These use the tokenizer.json format from HuggingFace Tokenizers library.
	for _, className := range []string{
		// BERT-style (WordPiece)
		"BertTokenizer",
		"BertTokenizerFast",
		"DistilBertTokenizer",
		"DistilBertTokenizerFast",
		"DebertaTokenizer",
		"DebertaTokenizerFast",
		"DebertaV2Tokenizer",
		"DebertaV2TokenizerFast",
		"ElectraTokenizer",
		"ElectraTokenizerFast",
		// RoBERTa-style (BPE)
		"RobertaTokenizer",
		"RobertaTokenizerFast",
		"XLMRobertaTokenizer",
		"XLMRobertaTokenizerFast",
		// GPT-2 style (BPE)
		"GPT2Tokenizer",
		"GPT2TokenizerFast",
		// LLaMA-style (BPE, using tokenizer.json)
		"LlamaTokenizer",
		"LlamaTokenizerFast",
		// Other common tokenizers
		"PreTrainedTokenizerFast",
		"AutoTokenizer",
	} {
		RegisterTokenizerClass(className, hftokenizer.New)
	}
}
