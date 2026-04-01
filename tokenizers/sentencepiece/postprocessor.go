package sentencepiece

import (
	"github.com/gomlx/go-huggingface/tokenizers/api"
)

// applyPostProcessor applies the post_processor to add special tokens.
// It injects configured bos/eos tokens from tokenizer_config.json.
func (t *Tokenizer) applyPostProcessor(ids []int, spans []api.TokenSpan) ([]int, []api.TokenSpan, []int) {
	outIDs := ids
	outSpans := spans
	var outSpecial []int

	// Prepare outSpecial mask if it hasn't been instantiated
	if outSpecial == nil && len(outIDs) > 0 {
		outSpecial = make([]int, len(outIDs))
	}

	if t.config != nil {
		if t.config.AddBosToken && t.Info.BeginningOfSentenceID >= 0 {
			if len(outIDs) == 0 || outIDs[0] != t.Info.BeginningOfSentenceID {
				outIDs = append([]int{t.Info.BeginningOfSentenceID}, outIDs...)
				if spans != nil {
					outSpans = append([]api.TokenSpan{{Start: -1, End: -1}}, outSpans...)
				}
				outSpecial = append([]int{1}, outSpecial...)
			}
		}
		if t.config.AddEosToken && t.Info.EndOfSentenceID >= 0 {
			if len(outIDs) == 0 || outIDs[len(outIDs)-1] != t.Info.EndOfSentenceID {
				outIDs = append(outIDs, t.Info.EndOfSentenceID)
				if spans != nil {
					outSpans = append(outSpans, api.TokenSpan{Start: -1, End: -1})
				}
				outSpecial = append(outSpecial, 1)
			}
		}
	}

	return outIDs, outSpans, outSpecial
}
