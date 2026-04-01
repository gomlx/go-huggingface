// Package api defines the Tokenizer API.
package api

import "errors"

var ErrNotImplemented = errors.New("not implemented")

// Tokenizer interface allows one convert test to "tokens" (integer ids) and back.
//
// It also allows mapping of special tokens: tokens with a common semantic (like padding) but that
// may map to different ids (int) for different tokenizers.
type Tokenizer interface {
	// Encode converts text to token IDs with post-processing (e.g., [CLS]/[SEP]).
	// Equivalent to EncodeWithOptions(text, true).
	Encode(text string) []int

	// EncodeWithAnnotations returns the encoded text along with requested annotations.
	// Use With to select the desired annotation.
	EncodeWithAnnotations(text string) AnnotatedEncoding

	// With applies options to a tokenizer.
	// If an option is unsupported, it returns an ErrNotImplemented derived error.
	//
	// These options may change how the tokenizer behaves, and usually they are set only once.
	// It is up to the caller to coordinate if the tokenizer is being used with alternating
	// options — it’s not a common pattern.
	With(options EncodeOptions) error

	// Decode converts the tokens back to their string representations.
	Decode([]int) string

	// SpecialTokenID returns ID for given special token if registered, or an ErrNotImplemented derived error if not.
	//
	// This allows for a common representation of special tokens across different tokenizers.
	SpecialTokenID(token SpecialToken) (int, error)

	// Normalize returns the normalization used by the tokenizer (e.g.: BERT lower cases the string)
	// If a Tokenizer uses no normalization this simply returns its input.
	Normalize(string) string

	// VocabSize returns the total number of tokens in the vocabulary.
	VocabSize() int

	// Config returns the HuggingFace tokenizer configuration.
	// It is optional, and in case the tokenizer has been instantiated in some other fashion it may return nil.
	Config() *Config
}

// AnnotatedEncoding contains various optional annotations.
//
// The annotations included are controlled by the options selected with Tokenizer.With.
type AnnotatedEncoding struct {
	IDs               []int       // token IDs
	Spans             []TokenSpan // byte spans for each token (use originalText[span.Start:span.End] to extract)
	SpecialTokensMask []int
}

// TokenSpan represents the byte span of a token in the original text.
// Start and End are byte offsets (not rune offsets), suitable for slicing
// Go strings directly: originalText[span.Start:span.End].
// This is useful for token classification tasks (NER, chunking) where you need
// to map token predictions back to positions in the original text.
type TokenSpan struct {
	Start int // start byte position (inclusive)
	End   int // end byte position (exclusive)
}

// EncodeOptions for the tokenizer.
type EncodeOptions struct {

	// AddSpecialTokens option takes a boolean value, and enables post-processing (e.g., [CLS]/[SEP] for BERT).
	AddSpecialTokens bool

	// MaxLen option takes an int value. Set it to a value <= 0 to disable MaxLen.
	// Encoding will be truncated to this length.
	MaxLen int

	// IncludeSpans option takes a boolean, and indicates if EncodeWithAnnotations should include spans.
	IncludeSpans bool

	// IncludeSpecialTokensMask option takes a boolean value, and enables post-processing (e.g., [CLS]/[SEP] for BERT).
	IncludeSpecialTokensMask bool
}

// SpecialToken is an enum of commonly used special tokens.
type SpecialToken int

const (
	TokBeginningOfSentence SpecialToken = iota
	TokEndOfSentence
	TokUnknown
	TokPad
	TokMask
	TokClassification
	TokSpecialTokensCount
)
