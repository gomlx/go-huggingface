package hftokenizer

import (
	"strings"
	"unicode"
)

// preTokenizeWithSpans splits text into words with their byte spans.
func (t *Tokenizer) preTokenizeWithSpans(text string, normOffsets []int) []wordWithOffset {
	if t.tokenizer.PreTokenizer == nil {
		// Default: split on whitespace
		return fieldsWithOffsets(text, normOffsets)
	}
	return t.applyPreTokenizerWithSpans(text, normOffsets, t.tokenizer.PreTokenizer)
}

// fieldsWithOffsets splits text on whitespace and returns words with their offsets.
func fieldsWithOffsets(text string, normOffsets []int) []wordWithOffset {
	var words []wordWithOffset
	var current strings.Builder
	currentStart := -1

	for i, r := range text {
		if unicode.IsSpace(r) {
			if current.Len() > 0 {
				end := i
				origStart := 0
				origEnd := len(text)
				if currentStart < len(normOffsets) {
					origStart = normOffsets[currentStart]
				}
				if end <= len(normOffsets) && end > 0 {
					origEnd = normOffsets[end-1] + 1
				}
				words = append(words, wordWithOffset{
					text:  current.String(),
					start: origStart,
					end:   origEnd,
				})
				current.Reset()
				currentStart = -1
			}
		} else {
			if currentStart == -1 {
				currentStart = i
			}
			current.WriteRune(r)
		}
	}

	if current.Len() > 0 {
		origStart := 0
		origEnd := len(text)
		if currentStart < len(normOffsets) {
			origStart = normOffsets[currentStart]
		}
		if len(normOffsets) > 0 {
			origEnd = normOffsets[len(normOffsets)-1] + 1
		}
		words = append(words, wordWithOffset{
			text:  current.String(),
			start: origStart,
			end:   origEnd,
		})
	}

	return words
}

// applyPreTokenizerWithSpans applies pre-tokenization with offset tracking.
func (t *Tokenizer) applyPreTokenizerWithSpans(text string, normOffsets []int, pt *PreTokenizer) []wordWithOffset {
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
		return byteLevelPreTokenizeWithOffsets(text, normOffsets)
	case "Metaspace":
		// default to true if missing, not false
		split := true
		if pt.Split != nil {
			split = *pt.Split
		}
		return metaspacePreTokenizeWithOffsets(text, normOffsets, pt.AddPrefixSpace, pt.Replacement, pt.PrependScheme, split)
	case "Sequence":
		result := []wordWithOffset{{text: text, start: 0, end: len(text)}}
		if len(normOffsets) > 0 {
			result[0].end = normOffsets[len(normOffsets)-1] + 1
		}
		for _, child := range pt.PreTokenizers {
			var newResult []wordWithOffset
			childCopy := child
			for _, w := range result {
				// Create sub-offsets for this word
				subOffsets := make([]int, len(w.text))
				for i := range subOffsets {
					subOffsets[i] = w.start + i
				}
				subWords := t.applyPreTokenizerWithSpans(w.text, subOffsets, &childCopy)
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

// bertPreTokenizeWithOffsets splits on whitespace and punctuation with offset tracking.
func bertPreTokenizeWithOffsets(text string, normOffsets []int) []wordWithOffset {
	var words []wordWithOffset
	var current strings.Builder
	currentStart := -1

	runes := []rune(text)
	for i, r := range runes {
		bytePos := len(string(runes[:i]))

		if isWhitespace(r) {
			if current.Len() > 0 {
				origStart := 0
				origEnd := bytePos
				if currentStart < len(normOffsets) {
					origStart = normOffsets[currentStart]
				}
				if bytePos > 0 && bytePos <= len(normOffsets) {
					origEnd = normOffsets[bytePos-1] + 1
				}
				words = append(words, wordWithOffset{
					text:  current.String(),
					start: origStart,
					end:   origEnd,
				})
				current.Reset()
				currentStart = -1
			}
		} else if isPunctuation(r) {
			if current.Len() > 0 {
				origStart := 0
				origEnd := bytePos
				if currentStart < len(normOffsets) {
					origStart = normOffsets[currentStart]
				}
				if bytePos > 0 && bytePos <= len(normOffsets) {
					origEnd = normOffsets[bytePos-1] + 1
				}
				words = append(words, wordWithOffset{
					text:  current.String(),
					start: origStart,
					end:   origEnd,
				})
				current.Reset()
				currentStart = -1
			}
			// Add punctuation as its own token
			origStart := bytePos
			origEnd := bytePos + len(string(r))
			if bytePos < len(normOffsets) {
				origStart = normOffsets[bytePos]
			}
			endBytePos := bytePos + len(string(r))
			if endBytePos <= len(normOffsets) && endBytePos > 0 {
				origEnd = normOffsets[endBytePos-1] + 1
			}
			words = append(words, wordWithOffset{
				text:  string(r),
				start: origStart,
				end:   origEnd,
			})
		} else {
			if currentStart == -1 {
				currentStart = bytePos
			}
			current.WriteRune(r)
		}
	}

	if current.Len() > 0 {
		origStart := 0
		origEnd := len(text)
		if currentStart < len(normOffsets) {
			origStart = normOffsets[currentStart]
		}
		if len(normOffsets) > 0 {
			origEnd = normOffsets[len(normOffsets)-1] + 1
		}
		words = append(words, wordWithOffset{
			text:  current.String(),
			start: origStart,
			end:   origEnd,
		})
	}

	return words
}

// punctuationPreTokenizeWithOffsets splits on punctuation with offset tracking.
func punctuationPreTokenizeWithOffsets(text string, normOffsets []int) []wordWithOffset {
	var words []wordWithOffset
	var current strings.Builder
	currentStart := -1

	runes := []rune(text)
	for i, r := range runes {
		bytePos := len(string(runes[:i]))

		if isPunctuation(r) {
			if current.Len() > 0 {
				origStart := 0
				origEnd := bytePos
				if currentStart < len(normOffsets) {
					origStart = normOffsets[currentStart]
				}
				if bytePos > 0 && bytePos <= len(normOffsets) {
					origEnd = normOffsets[bytePos-1] + 1
				}
				words = append(words, wordWithOffset{
					text:  current.String(),
					start: origStart,
					end:   origEnd,
				})
				current.Reset()
				currentStart = -1
			}
			// Add punctuation as its own token
			origStart := bytePos
			origEnd := bytePos + len(string(r))
			if bytePos < len(normOffsets) {
				origStart = normOffsets[bytePos]
			}
			endBytePos := bytePos + len(string(r))
			if endBytePos <= len(normOffsets) && endBytePos > 0 {
				origEnd = normOffsets[endBytePos-1] + 1
			}
			words = append(words, wordWithOffset{
				text:  string(r),
				start: origStart,
				end:   origEnd,
			})
		} else {
			if currentStart == -1 {
				currentStart = bytePos
			}
			current.WriteRune(r)
		}
	}

	if current.Len() > 0 {
		origStart := 0
		origEnd := len(text)
		if currentStart < len(normOffsets) {
			origStart = normOffsets[currentStart]
		}
		if len(normOffsets) > 0 {
			origEnd = normOffsets[len(normOffsets)-1] + 1
		}
		words = append(words, wordWithOffset{
			text:  current.String(),
			start: origStart,
			end:   origEnd,
		})
	}

	return words
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
					text:  current.String(),
					start: origStart,
					end:   origEnd,
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
			text:  current.String(),
			start: origStart,
			end:   origEnd,
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
		newOffsets[0] = 0
		copy(newOffsets[1:], normOffsets)
		normOffsets = newOffsets
	}

	// Replace spaces with metaspace character
	var words []wordWithOffset
	var current strings.Builder
	currentStart := -1

	for i, r := range text {
		if r == ' ' {
			if split && current.Len() > 0 {
				origStart := 0
				origEnd := i
				if currentStart < len(normOffsets) && currentStart >= 0 {
					origStart = normOffsets[currentStart]
				}
				if i > 0 && i <= len(normOffsets) {
					origEnd = normOffsets[i-1] + 1
				}
				words = append(words, wordWithOffset{
					text:  current.String(),
					start: origStart,
					end:   origEnd,
				})
				current.Reset()
				currentStart = i
			}
			current.WriteString(replacement)
			if currentStart == -1 {
				currentStart = i
			}
		} else {
			if currentStart == -1 {
				currentStart = i
			}
			current.WriteRune(r)
		}
	}

	if current.Len() > 0 {
		origStart := 0
		origEnd := len(text)
		if currentStart < len(normOffsets) && currentStart >= 0 {
			origStart = normOffsets[currentStart]
		}
		if len(normOffsets) > 0 {
			origEnd = normOffsets[len(normOffsets)-1] + 1
		}
		words = append(words, wordWithOffset{
			text:  current.String(),
			start: origStart,
			end:   origEnd,
		})
	}

	return words
}
