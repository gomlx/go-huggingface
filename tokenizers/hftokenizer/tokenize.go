package hftokenizer

import (
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"strings"
)

// tokenizeWordWithSpans tokenizes a single word and returns IDs with their offsets.
func (t *Tokenizer) tokenizeWordWithSpans(word wordWithOffset) ([]int, []api.TokenSpan) {
	// First check if word is an added token
	if id, ok := t.addedTokens[word.text]; ok {
		return []int{id}, []api.TokenSpan{{Start: word.start, End: word.end}}
	}

	switch t.tokenizer.Model.Type {
	case "WordPiece":
		return t.wordPieceTokenizeWithSpans(word)
	case "BPE":
		return t.bpeTokenizeWithSpans(word)
	case "Unigram":
		return t.unigramTokenizeWithSpans(word)
	default:
		// Fallback: try to find word in vocab
		if id, ok := t.tokenizer.Model.Vocab[word.text]; ok {
			return []int{id}, []api.TokenSpan{{Start: word.start, End: word.end}}
		}
		if t.unkID >= 0 {
			return []int{t.unkID}, []api.TokenSpan{{Start: word.start, End: word.end}}
		}
		return nil, nil
	}
}

// wordPieceTokenizeWithSpans implements WordPiece tokenization with offset tracking.
func (t *Tokenizer) wordPieceTokenizeWithSpans(word wordWithOffset) ([]int, []api.TokenSpan) {
	text := word.text
	if text == "" {
		return nil, nil
	}

	maxChars := t.tokenizer.Model.MaxInputCharsPerWord
	if maxChars == 0 {
		maxChars = 100
	}
	if len(text) > maxChars {
		if t.unkID >= 0 {
			return []int{t.unkID}, []api.TokenSpan{{Start: word.start, End: word.end}}
		}
		return nil, nil
	}

	prefix := t.tokenizer.Model.ContinuingSubwordPrefix
	if prefix == "" {
		prefix = "##"
	}

	var ids []int
	var offsets []api.TokenSpan
	runes := []rune(text)
	start := 0
	charLen := len(runes)

	for start < charLen {
		end := charLen
		found := false

		for start < end {
			substr := string(runes[start:end])
			if start > 0 {
				substr = prefix + substr
			}

			if id, ok := t.tokenizer.Model.Vocab[substr]; ok {
				ids = append(ids, id)

				// Calculate character offsets for this subword
				// Map from rune position to byte position within the word
				startByte := len(string(runes[:start]))
				endByte := len(string(runes[:end]))

				// Add the word's start offset to get positions in original text
				origStart := word.start + startByte
				origEnd := word.start + endByte

				offsets = append(offsets, api.TokenSpan{Start: origStart, End: origEnd})
				found = true
				break
			}
			end--
		}

		if !found {
			if t.unkID >= 0 {
				return []int{t.unkID}, []api.TokenSpan{{Start: word.start, End: word.end}}
			}
			return nil, nil
		}
		start = end
	}

	return ids, offsets
}

// bpeTokenizeWithSpans implements BPE tokenization with offset tracking.
func (t *Tokenizer) bpeTokenizeWithSpans(word wordWithOffset) ([]int, []api.TokenSpan) {
	text := word.text
	if text == "" {
		return nil, nil
	}

	// Convert word to list of symbols with their character positions (rune indices)
	type symbolWithPos struct {
		text  string
		start int // rune position in word
		end   int // rune position in word
	}

	runes := []rune(text)
	symbols := make([]symbolWithPos, len(runes))
	for i, r := range runes {
		symbols[i] = symbolWithPos{
			text:  string(r),
			start: i,
			end:   i + 1,
		}
	}

	// Add end-of-word suffix if configured
	if t.tokenizer.Model.EndOfWordSuffix != "" && len(symbols) > 0 {
		symbols[len(symbols)-1].text += t.tokenizer.Model.EndOfWordSuffix
	}

	// If word is a single symbol that exists in vocab, return it
	if len(symbols) == 1 {
		if id, ok := t.tokenizer.Model.Vocab[symbols[0].text]; ok {
			return []int{id}, []api.TokenSpan{{Start: word.start, End: word.end}}
		}
	}

	// Apply BPE merges
	for len(symbols) > 1 {
		// Find best pair to merge
		bestPair := ""
		bestRank := -1
		bestIdx := -1

		for i := 0; i < len(symbols)-1; i++ {
			pair := symbols[i].text + " " + symbols[i+1].text
			if rank, ok := t.mergeRanks[pair]; ok {
				if bestRank == -1 || rank < bestRank {
					bestPair = pair
					bestRank = rank
					bestIdx = i
				}
			}
		}

		if bestIdx == -1 {
			break // No more merges possible
		}

		// Apply the merge
		merged := strings.Replace(bestPair, " ", "", 1)
		newSymbols := make([]symbolWithPos, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, symbolWithPos{
			text:  merged,
			start: symbols[bestIdx].start,
			end:   symbols[bestIdx+1].end,
		})
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}

	// Convert symbols to IDs with offsets
	var ids []int
	var offsets []api.TokenSpan

	for _, sym := range symbols {
		if id, ok := t.tokenizer.Model.Vocab[sym.text]; ok {
			ids = append(ids, id)
		} else if t.unkID >= 0 {
			ids = append(ids, t.unkID)
		} else {
			continue
		}

		// Calculate offsets - map from rune position to byte position
		startByte := len(string(runes[:sym.start]))
		endByte := len(string(runes[:sym.end]))

		// Add the word's start offset to get positions in original text
		origStart := word.start + startByte
		origEnd := word.start + endByte

		offsets = append(offsets, api.TokenSpan{Start: origStart, End: origEnd})
	}

	return ids, offsets
}

// unigramTokenizeWithSpans implements Unigram tokenization with offset tracking.
func (t *Tokenizer) unigramTokenizeWithSpans(word wordWithOffset) ([]int, []api.TokenSpan) {
	text := word.text
	if text == "" {
		return nil, nil
	}

	var ids []int
	var offsets []api.TokenSpan
	runes := []rune(text)
	start := 0
	runeLen := len(runes)

	for start < runeLen {
		end := runeLen
		found := false

		for end > start {
			substr := string(runes[start:end])
			if id, ok := t.tokenizer.Model.Vocab[substr]; ok {
				ids = append(ids, id)

				// Calculate offsets - map from rune position to byte position
				startByte := len(string(runes[:start]))
				endByte := len(string(runes[:end]))

				// Add the word's start offset to get positions in original text
				origStart := word.start + startByte
				origEnd := word.start + endByte

				offsets = append(offsets, api.TokenSpan{Start: origStart, End: origEnd})
				found = true
				start = end
				break
			}
			end--
		}

		if !found {
			// Single character fallback
			char := string(runes[start])
			startByte := len(string(runes[:start]))
			endByte := len(string(runes[:start+1]))

			// Add the word's start offset to get positions in original text
			origStart := word.start + startByte
			origEnd := word.start + endByte

			if id, ok := t.tokenizer.Model.Vocab[char]; ok {
				ids = append(ids, id)
			} else if t.unkID >= 0 {
				ids = append(ids, t.unkID)
			}
			offsets = append(offsets, api.TokenSpan{Start: origStart, End: origEnd})
			start++
		}
	}

	return ids, offsets
}
