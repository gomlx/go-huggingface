// Package api defines the Tokenizer API.
// It's just a hack to break the cyclic dependency, and allow the users to import `tokenizers` and get the
// default implementations.
package api

// TokenSpan represents the byte span of a token in the original text.
// Start and End are byte offsets (not rune offsets), suitable for slicing
// Go strings directly: originalText[span.Start:span.End].
// This is useful for token classification tasks (NER, chunking) where you need
// to map token predictions back to positions in the original text.
type TokenSpan struct {
	Start int // start byte position (inclusive)
	End   int // end byte position (exclusive)
}

// EncodingResult contains tokens with their spans in the original text.
type EncodingResult struct {
	IDs   []int       // token IDs
	Spans []TokenSpan // byte spans for each token (use originalText[span.Start:span.End] to extract)
}

// Tokenizer interface allows one convert test to "tokens" (integer ids) and back.
//
// It also allows mapping of special tokens: tokens with a common semantic (like padding) but that
// may map to different ids (int) for different tokenizers.
type Tokenizer interface {
	Encode(text string) []int
	Decode([]int) string

	// SpecialTokenID returns ID for given special token if registered, or an error if not.
	SpecialTokenID(token SpecialToken) (int, error)
}

// TokenizerWithSpans extends Tokenizer with span tracking capability.
// This is useful for token classification tasks (NER, chunking) where you need
// to map token predictions back to byte positions in the original text.
type TokenizerWithSpans interface {
	Tokenizer
	// EncodeWithSpans returns tokens along with their byte spans in the original text.
	EncodeWithSpans(text string) EncodingResult
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

//go:generate enumer -type=SpecialToken -trimprefix=Tok -transform=snake -values -text -json -yaml api.go
