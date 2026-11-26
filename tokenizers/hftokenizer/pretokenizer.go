package hftokenizer

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

// preTokenizeWithOffsets splits text into words with their byte offsets.
func (t *Tokenizer) preTokenizeWithOffsets(text string, normOffsets []int) []wordWithOffset {
	if t.tokenizer.PreTokenizer == nil {
		// Default: split on whitespace
		return fieldsWithOffsets(text, normOffsets)
	}
	return t.applyPreTokenizerWithOffsets(text, normOffsets, t.tokenizer.PreTokenizer)
}

// preTokenizeWithSpans is an alias for preTokenizeWithOffsets for backward compatibility.
func (t *Tokenizer) preTokenizeWithSpans(text string, normOffsets []int) []wordWithOffset {
	return t.preTokenizeWithOffsets(text, normOffsets)
}

// fieldsWithOffsets splits text on whitespace and returns words with their offsets.
func fieldsWithOffsets(text string, normOffsets []int) []wordWithOffset {
	var words []wordWithOffset
	currentStart := -1

	for i, r := range text {
		if unicode.IsSpace(r) {
			if currentStart != -1 {
				words = append(words, makeWord(text, normOffsets, currentStart, i))
				currentStart = -1
			}
		} else {
			if currentStart == -1 {
				currentStart = i
			}
		}
	}

	if currentStart != -1 {
		words = append(words, makeWord(text, normOffsets, currentStart, len(text)))
	}

	return words
}

// applyPreTokenizerWithOffsets applies pre-tokenization with offset tracking.
func (t *Tokenizer) applyPreTokenizerWithOffsets(text string, normOffsets []int, pt *PreTokenizer) []wordWithOffset {
	switch pt.Type {
	case "BertPreTokenizer":
		return bertPreTokenizeWithOffsets(text, normOffsets)
	case "Whitespace", "WhitespaceSplit":
		return fieldsWithOffsets(text, normOffsets)
	case "ByteLevel":
		if pt.AddPrefixSpace && len(text) > 0 && text[0] != ' ' {
			// Prepend space - adjust offsets
			text = " " + text
			newOffsets := make([]int, len(normOffsets)+1)
			newOffsets[0] = 0 // The added space maps to position 0
			copy(newOffsets[1:], normOffsets)
			normOffsets = newOffsets
		}
		if pt.UseRegex != nil && !*pt.UseRegex {
			return byteLevelEncodeNoSplit(text, normOffsets)
		}
		return byteLevelPreTokenizeWithOffsets(text, normOffsets)
	case "Metaspace":
		// default to true if missing, not false
		split := true
		if pt.Split != nil {
			split = *pt.Split
		}
		return metaspacePreTokenizeWithOffsets(text, normOffsets, pt.AddPrefixSpace, pt.Replacement, pt.PrependScheme, split)
	case "Split":
		return splitPreTokenizeWithOffsets(text, normOffsets, pt)
	case "Sequence":
		result := []wordWithOffset{{text: text, start: 0, end: len(text), offsets: normOffsets}}
		if len(normOffsets) > 0 {
			result[0].end = normOffsets[len(normOffsets)-1] + 1
		}
		for _, child := range pt.PreTokenizers {
			var newResult []wordWithOffset
			childCopy := child
			for _, w := range result {
				// Create sub-offsets for this word
				subOffsets := w.offsets
				if len(subOffsets) != len(w.text) {
					subOffsets = make([]int, len(w.text))
					for i := range subOffsets {
						subOffsets[i] = w.start + i
					}
				}
				subWords := t.applyPreTokenizerWithOffsets(w.text, subOffsets, &childCopy)
				newResult = append(newResult, subWords...)
			}
			result = newResult
		}
		return result
	case "Punctuation":
		return punctuationPreTokenizeWithOffsets(text, normOffsets)
	default:
		return fieldsWithOffsets(text, normOffsets)
	}
}

// applyPreTokenizerWithSpans is an alias for applyPreTokenizerWithOffsets for backward compatibility.
func (t *Tokenizer) applyPreTokenizerWithSpans(text string, normOffsets []int, pt *PreTokenizer) []wordWithOffset {
	return t.applyPreTokenizerWithOffsets(text, normOffsets, pt)
}

// bertPreTokenizeWithOffsets splits on whitespace and punctuation with offset tracking.
func bertPreTokenizeWithOffsets(text string, normOffsets []int) []wordWithOffset {
	var words []wordWithOffset
	currentStart := -1

	runes := []rune(text)
	for i, r := range runes {
		bytePos := len(string(runes[:i]))

		if isWhitespace(r) {
			if currentStart != -1 {
				words = append(words, makeWord(text, normOffsets, currentStart, bytePos))
				currentStart = -1
			}
		} else if isPunctuation(r) {
			if currentStart != -1 {
				words = append(words, makeWord(text, normOffsets, currentStart, bytePos))
				currentStart = -1
			}
			// Add punctuation as its own token
			runeLen := len(string(r))
			words = append(words, makeWord(text, normOffsets, bytePos, bytePos+runeLen))
		} else {
			if currentStart == -1 {
				currentStart = bytePos
			}
		}
	}

	if currentStart != -1 {
		words = append(words, makeWord(text, normOffsets, currentStart, len(text)))
	}

	return words
}

// punctuationPreTokenizeWithOffsets splits on punctuation with offset tracking.
func punctuationPreTokenizeWithOffsets(text string, normOffsets []int) []wordWithOffset {
	var words []wordWithOffset
	currentStart := -1

	runes := []rune(text)
	for i, r := range runes {
		bytePos := len(string(runes[:i]))

		if isPunctuation(r) {
			if currentStart != -1 {
				words = append(words, makeWord(text, normOffsets, currentStart, bytePos))
				currentStart = -1
			}
			// Add punctuation as its own token
			runeLen := len(string(r))
			words = append(words, makeWord(text, normOffsets, bytePos, bytePos+runeLen))
		} else {
			if currentStart == -1 {
				currentStart = bytePos
			}
		}
	}

	if currentStart != -1 {
		words = append(words, makeWord(text, normOffsets, currentStart, len(text)))
	}

	return words
}

// byteLevelEncodeNoSplit maps bytes to GPT-2 byte-level unicode characters without splitting on spaces.
// Used when ByteLevel has use_regex: false (e.g., after a Split pre-tokenizer in Qwen/Llama-3/DeepSeek).
func byteLevelEncodeNoSplit(text string, normOffsets []int) []wordWithOffset {
	if len(text) == 0 {
		return nil
	}
	var current strings.Builder
	currentOffsets := make([]int, 0, len(text))
	for i := 0; i < len(text); i++ {
		current.WriteRune(byteToUnicode[text[i]])
		if i < len(normOffsets) {
			currentOffsets = append(currentOffsets, normOffsets[i])
		}
	}
	origStart := 0
	origEnd := len(text)
	if len(currentOffsets) > 0 {
		origStart = currentOffsets[0]
		origEnd = currentOffsets[len(currentOffsets)-1] + 1
	}
	return []wordWithOffset{{
		text:    current.String(),
		start:   origStart,
		end:     origEnd,
		offsets: currentOffsets,
	}}
}

// byteLevelPreTokenizeWithOffsets handles byte-level BPE pre-tokenization with offsets.
func byteLevelPreTokenizeWithOffsets(text string, normOffsets []int) []wordWithOffset {
	var words []wordWithOffset
	var current strings.Builder
	var currentOffsets []int

	for i, r := range text {
		if r == ' ' {
			if current.Len() > 0 {
				origStart := 0
				origEnd := i
				if len(currentOffsets) > 0 {
					origStart = currentOffsets[0]
					origEnd = currentOffsets[len(currentOffsets)-1] + 1
				}
				words = append(words, wordWithOffset{
					text:    current.String(),
					start:   origStart,
					end:     origEnd,
					offsets: currentOffsets,
				})
				current.Reset()
				currentOffsets = nil
			}
			// Start new token with space
			current.WriteRune(byteToUnicode[' '])
			if i < len(normOffsets) {
				currentOffsets = append(currentOffsets, normOffsets[i])
			}
		} else {
			for _, b := range []byte(string(r)) {
				current.WriteRune(byteToUnicode[b])
				if i < len(normOffsets) {
					currentOffsets = append(currentOffsets, normOffsets[i])
				}
			}
		}
	}

	if current.Len() > 0 {
		origStart := 0
		origEnd := len(text)
		if len(currentOffsets) > 0 {
			origStart = currentOffsets[0]
			origEnd = currentOffsets[len(currentOffsets)-1] + 1
		}
		words = append(words, wordWithOffset{
			text:    current.String(),
			start:   origStart,
			end:     origEnd,
			offsets: currentOffsets,
		})
	}

	return words
}

// metaspacePreTokenizeWithOffsets handles metaspace pre-tokenization with offsets.
func metaspacePreTokenizeWithOffsets(text string, normOffsets []int, addPrefixSpace bool, replacement string, prependScheme string, split bool) []wordWithOffset {
	if replacement == "" {
		replacement = "\u2581"
	}
	if (addPrefixSpace || prependScheme == "always") && len(text) > 0 && text[0] != ' ' {
		text = " " + text
		newOffsets := make([]int, len(normOffsets)+1)
		copy(newOffsets[1:], normOffsets)
		if len(normOffsets) > 0 {
			newOffsets[0] = normOffsets[0]
		}
		normOffsets = newOffsets
	}

	orig := func(i int) int {
		if i < len(normOffsets) {
			return normOffsets[i]
		}
		return i
	}
	var words []wordWithOffset
	var current strings.Builder
	var offsets []int
	flush := func() {
		if current.Len() == 0 {
			return
		}
		words = append(words, wordWithOffset{
			text:    current.String(),
			start:   offsets[0],
			end:     offsets[len(offsets)-1] + 1,
			offsets: offsets,
		})
		current.Reset()
		offsets = nil
	}
	for i, r := range text {
		if r == ' ' {
			if split {
				flush()
			}
			current.WriteString(replacement)
			for range len(replacement) {
				offsets = append(offsets, orig(i))
			}
		} else {
			current.WriteRune(r)
			for j := range utf8.RuneLen(r) {
				offsets = append(offsets, orig(i+j))
			}
		}
	}
	flush()
	return words
}

// findSplitMatches finds all regex match spans in text for a Split pre-tokenizer pattern.
// Go's RE2 engine does not support negative lookahead `(?!...)`, which appears in GPT-2,
// Qwen2/3/3.5, Llama-3, and Tiktoken pre-tokenizer regexes as `\s+(?!\S)`.
// When standard regexp.Compile fails and the pattern contains `\s+(?!\S)`, this function
// rewrites `\s+(?!\S)` into a named capturing group and emulates the negative lookahead
// by trimming the trailing whitespace rune whenever the match is followed by a non-space.
func findSplitMatches(text string, pt *PreTokenizer) ([][]int, bool) {
	if pt.Pattern == nil {
		return nil, false
	}
	if pt.Pattern.Regex == "" {
		var re *regexp.Regexp
		var err error
		if pt.Pattern.String != "" {
			re, err = regexp.Compile(regexp.QuoteMeta(pt.Pattern.String))
		} else {
			re, err = regexp.Compile(`(?s:.)`)
		}
		if err != nil {
			return nil, false
		}
		return re.FindAllStringIndex(text, -1), true
	}

	if re, err := regexp.Compile(pt.Pattern.Regex); err == nil {
		return re.FindAllStringIndex(text, -1), true
	}

	const wsLookahead = `\s+(?!\S)`
	if !strings.Contains(pt.Pattern.Regex, wsLookahead) {
		return nil, false
	}

	rewritten := strings.Replace(pt.Pattern.Regex, wsLookahead, `(?P<ws_not_S>\s+)`, 1)
	re, err := regexp.Compile(rewritten)
	if err != nil {
		return nil, false
	}
	wsGroupIdx := re.SubexpIndex("ws_not_S")

	var matches [][]int
	for pos := 0; pos < len(text); {
		sub := re.FindStringSubmatchIndex(text[pos:])
		if sub == nil {
			break
		}
		mStart := pos + sub[0]
		mEnd := pos + sub[1]
		if wsGroupIdx > 0 && sub[wsGroupIdx*2] >= 0 && mEnd < len(text) {
			// Because `\s+` is greedy and mEnd < len(text), text[mEnd] is `\S`.
			// `\s+(?!\S)` must leave the final whitespace rune for the next match
			// (e.g., "    if" -> "   " + " if") when at least 2 whitespace runes matched.
			_, lastSize := utf8.DecodeLastRuneInString(text[mStart:mEnd])
			if mEnd-lastSize > mStart {
				mEnd -= lastSize
			}
		}
		matches = append(matches, []int{mStart, mEnd})
		if mEnd == pos {
			// Guard against zero-width match infinite loop.
			_, sz := utf8.DecodeRuneInString(text[pos:])
			pos += max(sz, 1)
		} else {
			pos = mEnd
		}
	}
	return matches, true
}

// splitPreTokenizeWithOffsets splits text based on pattern and behavior.
func splitPreTokenizeWithOffsets(text string, normOffsets []int, pt *PreTokenizer) []wordWithOffset {
	if len(text) == 0 {
		return nil
	}

	matches, ok := findSplitMatches(text, pt)
	if !ok {
		// Fallback: return whole text as a single word
		return []wordWithOffset{makeWord(text, normOffsets, 0, len(text))}
	}

	type segment struct {
		start       int
		end         int
		isDelimiter bool
	}

	var segments []segment
	cur := 0
	for _, m := range matches {
		sStart, sEnd := m[0], m[1]
		if sStart > cur {
			segments = append(segments, segment{
				start:       cur,
				end:         sStart,
				isDelimiter: pt.Invert,
			})
		}
		segments = append(segments, segment{
			start:       sStart,
			end:         sEnd,
			isDelimiter: !pt.Invert,
		})
		cur = sEnd
	}
	if cur < len(text) {
		segments = append(segments, segment{
			start:       cur,
			end:         len(text),
			isDelimiter: pt.Invert,
		})
	}

	behavior := strings.ToLower(pt.Behavior)
	var words []wordWithOffset

	switch behavior {
	case "removed":
		for _, seg := range segments {
			if !seg.isDelimiter {
				words = append(words, makeWord(text, normOffsets, seg.start, seg.end))
			}
		}

	case "isolated":
		for _, seg := range segments {
			words = append(words, makeWord(text, normOffsets, seg.start, seg.end))
		}

	case "contiguous":
		for i := 0; i < len(segments); {
			seg := segments[i]
			if !seg.isDelimiter {
				words = append(words, makeWord(text, normOffsets, seg.start, seg.end))
				i++
			} else {
				start := seg.start
				end := seg.end
				j := i + 1
				for j < len(segments) && segments[j].isDelimiter {
					end = segments[j].end
					j++
				}
				words = append(words, makeWord(text, normOffsets, start, end))
				i = j
			}
		}

	case "merged_with_previous":
		for i := 0; i < len(segments); {
			seg := segments[i]
			if !seg.isDelimiter {
				start := seg.start
				end := seg.end
				i++
				if i < len(segments) && segments[i].isDelimiter {
					end = segments[i].end
					i++
				}
				words = append(words, makeWord(text, normOffsets, start, end))
			} else {
				words = append(words, makeWord(text, normOffsets, seg.start, seg.end))
				i++
			}
		}

	case "merged_with_next":
		for i := 0; i < len(segments); {
			seg := segments[i]
			if seg.isDelimiter {
				start := seg.start
				end := seg.end
				i++
				if i < len(segments) && !segments[i].isDelimiter {
					end = segments[i].end
					i++
				}
				words = append(words, makeWord(text, normOffsets, start, end))
			} else {
				words = append(words, makeWord(text, normOffsets, seg.start, seg.end))
				i++
			}
		}

	default:
		// Default to removed behavior if unrecognized
		for _, seg := range segments {
			if !seg.isDelimiter {
				words = append(words, makeWord(text, normOffsets, seg.start, seg.end))
			}
		}
	}

	return words
}

func makeWord(text string, normOffsets []int, start, end int) wordWithOffset {
	origStart := start
	origEnd := end
	if len(normOffsets) > 0 {
		if start < len(normOffsets) {
			origStart = normOffsets[start]
		} else {
			origStart = normOffsets[len(normOffsets)-1] + 1
		}
		if end <= len(normOffsets) && end > 0 {
			origEnd = normOffsets[end-1] + 1
		} else {
			origEnd = normOffsets[len(normOffsets)-1] + 1
		}
	}
	var offsets []int
	if len(normOffsets) >= end && start <= end {
		offsets = normOffsets[start:end]
	}
	return wordWithOffset{
		text:    text[start:end],
		start:   origStart,
		end:     origEnd,
		offsets: offsets,
	}
}
