package hftokenizer

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/gomlx/go-huggingface/tokenizers/api"
)

// spanForSubword returns the original text span [origStart, origEnd] for a subword
// spanning bytes [startByte:endByte] within word.text.
func (word *wordWithOffset) spanForSubword(startByte, endByte int) api.TokenSpan {
	if len(word.offsets) == len(word.text) && endByte <= len(word.offsets) && startByte < endByte {
		origStart := word.offsets[startByte]
		origEnd := word.offsets[endByte-1] + 1
		return api.TokenSpan{Start: origStart, End: origEnd}
	}
	return api.TokenSpan{
		Start: word.start + startByte,
		End:   word.start + endByte,
	}
}

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

				offsets = append(offsets, word.spanForSubword(startByte, endByte))
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

		offsets = append(offsets, word.spanForSubword(startByte, endByte))
	}

	return ids, offsets
}

// unigramTokenizeWithSpans picks the pieces with the highest total score (Viterbi), as HuggingFace's Unigram model does.
func (t *Tokenizer) unigramTokenizeWithSpans(word wordWithOffset) ([]int, []api.TokenSpan) {
	text, model := word.text, &t.tokenizer.Model
	if text == "" {
		return nil, nil
	}
	best := make([]float64, len(text)+1)
	from := make([]int, len(text)+1)
	pieceID := make([]int, len(text)+1)
	for i := range from {
		from[i] = -1
	}
	relax := func(start, end, id int, score float64) {
		if s := best[start] + score; from[end] < 0 || s > best[end] {
			best[end], from[end], pieceID[end] = s, start, id
		}
	}
	for start := 0; start < len(text); {
		_, charLen := utf8.DecodeRuneInString(text[start:])
		hasChar := false
		for end := start + charLen; end <= min(len(text), start+t.unigramMaxPiece); {
			if id, ok := model.Vocab[text[start:end]]; ok {
				score := -1.0 // Fewest pieces win when the model has no scores.
				if id >= 0 && id < len(model.Scores) {
					score = model.Scores[id]
				}
				relax(start, end, id, score)
				hasChar = hasChar || end == start+charLen
			}
			if end == len(text) {
				break
			}
			_, size := utf8.DecodeRuneInString(text[end:])
			end += size
		}
		if !hasChar {
			relax(start, start+charLen, -1, t.unigramUnkScore)
		}
		start += charLen
	}

	var ends []int
	// Trace the best path back from the end, fusing each run of consecutive unknown pieces into one span.
	for end := len(text); end > 0; end = from[end] {
		if pieceID[end] >= 0 || len(ends) == 0 || pieceID[ends[len(ends)-1]] >= 0 {
			ends = append(ends, end)
		}
	}
	var ids []int
	var offsets []api.TokenSpan
	for i := len(ends) - 1; i >= 0; i-- {
		start, end := 0, ends[i]
		if i+1 < len(ends) {
			start = ends[i+1]
		}
		span := word.spanForSubword(start, end)
		if id := pieceID[end]; id >= 0 {
			ids, offsets = append(ids, id), append(offsets, span)
			continue
		}
		var byteIDs []int
		for j := start; model.ByteFallback && j < end; j++ {
			if id, ok := model.Vocab[fmt.Sprintf("<0x%02X>", text[j])]; ok {
				byteIDs = append(byteIDs, id)
			}
		}
		if len(byteIDs) == end-start {
			for _, id := range byteIDs {
				ids, offsets = append(ids, id), append(offsets, span)
			}
		} else if t.unkID >= 0 {
			ids, offsets = append(ids, t.unkID), append(offsets, span)
		}
	}
	return ids, offsets
}
