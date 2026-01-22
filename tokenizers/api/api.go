// Package api defines the Tokenizer API.
// It's just a hack to break the cyclic dependency, and allow the users to import `tokenizers` and get the
// default implementations.
package api

// TokenOffset represents the character span of a token in the original text.
// This is useful for token classification tasks (NER, chunking) where you need
// to map token predictions back to character positions in the original text.
type TokenOffset struct {
	Start int // start character position (inclusive)
	End   int // end character position (exclusive)
}

// EncodingResult contains tokens with their offsets.
type EncodingResult struct {
	IDs     []int         // token IDs
	Offsets []TokenOffset // character offsets for each token
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

// TokenizerWithOffsets extends Tokenizer with offset tracking capability.
// This is useful for token classification tasks (NER, chunking) where you need
// to map token predictions back to character positions in the original text.
type TokenizerWithOffsets interface {
	Tokenizer
	// EncodeWithOffsets returns tokens along with their character offsets in the original text.
	EncodeWithOffsets(text string) EncodingResult
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
