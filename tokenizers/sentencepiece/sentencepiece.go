// Package sentencepiece implements a tokenizers.Tokenizer based on SentencePiece tokenizer.
package sentencepiece

import (
	"strings"

	esentencepiece "github.com/eliben/go-sentencepiece"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/pkg/errors"
)

// New creates a SentencePiece tokenizer based on the "tokenizer.model" file, which must be a
// SentencePiece Model proto (see protos.Model).
//
// It implements a tokenizer.TokenizerConstructor function signature.
func New(config *api.Config, repo *hub.Repo) (api.Tokenizer, error) {
	if !repo.HasFile("tokenizer.model") {
		return nil, errors.Errorf("\"tokenizer.model\" file not found in repo")
	}
	tokenizerFile, err := repo.DownloadFile("tokenizer.model")
	if err != nil {
		return nil, errors.Wrapf(err, "can't download tokenizer.json file")
	}
	proc, err := esentencepiece.NewProcessorFromPath(tokenizerFile)
	if err != nil {
		return nil, errors.Wrapf(err, "can't create sentencepiece tokenizer")
	}
	return &Tokenizer{
		Processor: proc,
		Info:      proc.ModelInfo(),
	}, nil
}

// Tokenizer implements tokenizers.Tokenizer interface based on SentencePiece tokenizer by Google.
type Tokenizer struct {
	*esentencepiece.Processor
	Info *esentencepiece.ModelInfo
}

// Compile time assert that sentencepiece.Tokenizer implements tokenizers.Tokenizer interface.
var _ api.Tokenizer = &Tokenizer{}

// Compile time assert that sentencepiece.Tokenizer implements tokenizers.TokenizerWithSpans interface.
var _ api.TokenizerWithSpans = &Tokenizer{}

// Encode returns the text encoded into a sequence of ids.
// It implements sampler.Vocabulary.
func (p *Tokenizer) Encode(text string) []int {
	tokens := p.Processor.Encode(text)
	return sliceMap(tokens, func(t esentencepiece.Token) int { return t.ID })
}

// EncodeWithSpans returns the text encoded into a sequence of ids along with their byte spans.
// It implements api.TokenizerWithSpans.
func (p *Tokenizer) EncodeWithSpans(text string) api.EncodingResult {
	tokens := p.Processor.Encode(text)
	ids := make([]int, len(tokens))
	spans := make([]api.TokenSpan, len(tokens))

	// Track position in original text by matching token pieces
	pos := 0
	for i, tok := range tokens {
		ids[i] = tok.ID
		piece := tok.Text

		// SentencePiece uses U+2581 (lower one eighth block) as the space replacement
		// We need to handle this when matching back to original text
		matchPiece := piece
		hasLeadingSpace := false
		if len(matchPiece) > 0 && matchPiece[0] == '\xe2' && len(matchPiece) >= 3 &&
			matchPiece[1] == '\x96' && matchPiece[2] == '\x81' {
			// Remove the U+2581 metaspace character for matching
			matchPiece = matchPiece[3:]
			hasLeadingSpace = true
		}

		// Skip any whitespace in the original text before this token
		if hasLeadingSpace {
			for pos < len(text) && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r') {
				pos++
			}
		}

		// Find where this piece starts in the original text
		start := pos

		// Advance position by the length of the actual content
		if matchPiece == "" {
			// Empty piece after removing metaspace (token represents just the space)
			// Check if there was actually a space character
			if hasLeadingSpace && start > 0 {
				start = start - 1
				spans[i] = api.TokenSpan{Start: start, End: pos}
			} else {
				spans[i] = api.TokenSpan{Start: pos, End: pos}
			}
		} else {
			// Find the piece in the text starting from current position
			foundAt := findSubstring(text, matchPiece, pos)
			if foundAt >= 0 {
				start = foundAt
				pos = foundAt + len(matchPiece)
			} else {
				// Fallback: advance by piece length
				pos += len(matchPiece)
				if pos > len(text) {
					pos = len(text)
				}
			}
			spans[i] = api.TokenSpan{Start: start, End: pos}
		}
	}

	return api.EncodingResult{
		IDs:   ids,
		Spans: spans,
	}
}

// findSubstring finds the first occurrence of substr in s starting from position start.
// Returns the byte position of the match, or -1 if not found.
func findSubstring(s, substr string, start int) int {
	if start >= len(s) {
		return -1
	}
	idx := strings.Index(s[start:], substr)
	if idx < 0 {
		return -1
	}
	return start + idx
}

// Decode returns the text from a sequence of ids.
// It implements sampler.Vocabulary.
func (p *Tokenizer) Decode(ids []int) string {
	return p.Processor.Decode(ids)
}

// SpecialTokenID returns the token for the given symbol, or an error if not known.
func (p *Tokenizer) SpecialTokenID(token api.SpecialToken) (int, error) {
	switch token {
	case api.TokUnknown:
		return p.Info.UnknownID, nil
	case api.TokPad:
		return p.Info.PadID, nil
	case api.TokBeginningOfSentence:
		return p.Info.BeginningOfSentenceID, nil
	case api.TokEndOfSentence:
		return p.Info.EndOfSentenceID, nil
	default:
		return 0, errors.Errorf("unknown special token: %s (%d)", token, int(token))
	}
}

// sliceMap executes the given function sequentially for every element on in, and returns a mapped slice.
func sliceMap[In, Out any](in []In, fn func(e In) Out) (out []Out) {
	out = make([]Out, len(in))
	for ii, e := range in {
		out[ii] = fn(e)
	}
	return
}
