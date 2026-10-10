package hftokenizer

import (
	"regexp"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// NormalizationType represents the type of text normalization.
type NormalizationType int

const (
	NormalizationUnknown NormalizationType = iota
	NormalizationLowercase
	NormalizationBert
	NormalizationNFD
	NormalizationNFC
	NormalizationNFKD
	NormalizationNFKC
	NormalizationStripAccents
	NormalizationSequence
	NormalizationReplace
	NormalizationPrepend
)

// String returns the HuggingFace normalizer type string.
func (t NormalizationType) String() string {
	switch t {
	case NormalizationLowercase:
		return "Lowercase"
	case NormalizationBert:
		return "BertNormalizer"
	case NormalizationNFD:
		return "NFD"
	case NormalizationNFC:
		return "NFC"
	case NormalizationNFKD:
		return "NFKD"
	case NormalizationNFKC:
		return "NFKC"
	case NormalizationStripAccents:
		return "StripAccents"
	case NormalizationSequence:
		return "Sequence"
	case NormalizationReplace:
		return "Replace"
	case NormalizationPrepend:
		return "Prepend"
	default:
		return "Unknown"
	}
}

// ParseNormalizationType converts a type string from tokenizer.json into NormalizationType.
func ParseNormalizationType(s string) NormalizationType {
	switch s {
	case "Lowercase":
		return NormalizationLowercase
	case "BertNormalizer":
		return NormalizationBert
	case "NFD":
		return NormalizationNFD
	case "NFC":
		return NormalizationNFC
	case "NFKD":
		return NormalizationNFKD
	case "NFKC":
		return NormalizationNFKC
	case "StripAccents":
		return NormalizationStripAccents
	case "Sequence":
		return NormalizationSequence
	case "Replace":
		return NormalizationReplace
	case "Prepend":
		return NormalizationPrepend
	default:
		return NormalizationUnknown
	}
}

// NormalizationType returns the parsed NormalizationType for this normalizer.
func (n *Normalizer) NormalizationType() NormalizationType {
	if n == nil {
		return NormalizationUnknown
	}
	return ParseNormalizationType(n.Type)
}

// identityOffsets creates an identity slice of length n where offsets[i] == i.
func identityOffsets(n int) []int {
	offsets := make([]int, n)
	for i := range offsets {
		offsets[i] = i
	}
	return offsets
}

// normalizeWithOffsets applies normalization and returns the normalized text along with
// a slice of byte offsets mapping each byte in the normalized text back to its corresponding
// byte index in the original text (i.e. offsets[normalizedByteIdx] = originalByteIdx).
func (t *Tokenizer) normalizeWithOffsets(text string) (string, []int) {
	if t.tokenizer.Normalizer == nil {
		return text, identityOffsets(len(text))
	}
	return applyNormalizerWithOffsets(text, t.tokenizer.Normalizer)
}

// applyNormalizer applies a normalizer to text without tracking offsets.
func (t *Tokenizer) applyNormalizer(text string, n *Normalizer) string {
	return applyNormalizer(text, n)
}

// applyNormalizerWithOffsets applies a normalizer and returns the normalized text
// along with a per-byte mapping from normalized positions to original text byte positions.
func (t *Tokenizer) applyNormalizerWithOffsets(text string, n *Normalizer) (string, []int) {
	return applyNormalizerWithOffsets(text, n)
}

// applyNormalizer executes the normalizer transformation without offset tracking.
func applyNormalizer(text string, n *Normalizer) string {
	if n == nil {
		return text
	}
	switch n.NormalizationType() {
	case NormalizationLowercase:
		return applyLowercaseNormalizer(text)
	case NormalizationBert:
		return applyBertNormalizer(text, n)
	case NormalizationNFD, NormalizationNFC, NormalizationNFKD, NormalizationNFKC:
		return applyUnicodeNormalizer(text, n)
	case NormalizationStripAccents:
		return applyStripAccentsNormalizer(text)
	case NormalizationSequence:
		return applySequenceNormalizer(text, n)
	case NormalizationReplace:
		return applyReplaceNormalizer(text, n)
	case NormalizationPrepend:
		return applyPrependNormalizer(text, n)
	default:
		return text
	}
}

// applyNormalizerWithOffsets executes the normalizer transformation and returns
// a per-byte mapping from normalized positions to original byte positions.
func applyNormalizerWithOffsets(text string, n *Normalizer) (string, []int) {
	if n == nil {
		return text, identityOffsets(len(text))
	}
	switch n.NormalizationType() {
	case NormalizationLowercase:
		return applyLowercaseNormalizerWithOffsets(text)
	case NormalizationBert:
		return applyBertNormalizerWithOffsets(text, n)
	case NormalizationNFD, NormalizationNFC, NormalizationNFKD, NormalizationNFKC:
		return applyUnicodeNormalizerWithOffsets(text, n)
	case NormalizationStripAccents:
		return applyStripAccentsNormalizerWithOffsets(text)
	case NormalizationSequence:
		return applySequenceNormalizerWithOffsets(text, n)
	case NormalizationReplace:
		return applyReplaceNormalizerWithOffsets(text, n)
	case NormalizationPrepend:
		return applyPrependNormalizerWithOffsets(text, n)
	default:
		// Unknown normalizer - use approximate mapping
		normalized := applyNormalizer(text, n)
		return approximateOffsets(text, normalized)
	}
}

// --- Lowercase Normalizer ---

func applyLowercaseNormalizer(text string) string {
	return strings.ToLower(text)
}

func applyLowercaseNormalizerWithOffsets(text string) (string, []int) {
	// Lowercase preserves character positions (1:1 mapping), but the
	// lowercased form of a single rune can occupy a different number of
	// BYTES than the original (e.g. "É" (2 bytes) -> "é" (2 bytes) is
	// fine, but some runes lowercase to a differently-sized UTF-8
	// encoding). offsets is indexed by byte position in `normalized`
	// (len(normalized) is a byte count), so it must be filled one entry
	// per output BYTE, not per output RUNE — iterating rune-by-rune
	// under-fills it for any multi-byte lowercased character, leaving
	// trailing entries at their zero value instead of a real offset.
	normalized := strings.ToLower(text)
	offsets := make([]int, len(normalized))
	origPos := 0
	normPos := 0
	for _, r := range text {
		lowerStr := strings.ToLower(string(r))
		for range len(lowerStr) {
			if normPos < len(offsets) {
				offsets[normPos] = origPos
				normPos++
			}
		}
		origPos += len(string(r))
	}
	return normalized, offsets
}

// --- Bert Normalizer ---

func applyBertNormalizer(text string, n *Normalizer) string {
	// Clean text, handle Chinese chars, strip accents, lowercase
	result := text
	if n.CleanText {
		result = cleanText(result)
	}
	if n.HandleChineseChars {
		result = tokenizeChineseChars(result)
	}
	if (n.StripAccents != nil && *n.StripAccents) || (n.StripAccents == nil && n.Lowercase) {
		result = removeAccents(norm.NFD.String(result))
	}
	if n.Lowercase {
		result = strings.ToLower(result)
	}
	return result
}

func applyBertNormalizerWithOffsets(text string, n *Normalizer) (string, []int) {
	// Clean text and optionally lowercase
	var result strings.Builder
	var offsets []int
	origPos := 0
	for _, r := range text {
		runeLen := len(string(r))
		if r == 0 || r == 0xFFFD || isControl(r) {
			// Skip this character
			origPos += runeLen
			continue
		}

		if n.HandleChineseChars && isChineseChar(r) {
			result.WriteRune(' ')
			offsets = append(offsets, origPos)
			result.WriteRune(r)
			// r itself may be a multi-byte rune (CJK characters are
			// typically 3 bytes in UTF-8) — append one offset entry per
			// BYTE written, not one entry for the whole rune. See the
			// "else" branch below for the same class of bug and a
			// fuller explanation.
			for range runeLen {
				offsets = append(offsets, origPos)
			}
			result.WriteRune(' ')
			offsets = append(offsets, origPos)
		} else if isWhitespace(r) {
			result.WriteRune(' ')
			offsets = append(offsets, origPos)
		} else {
			// Potential accent stripping and lowercasing
			s := string(r)
			if (n.StripAccents != nil && *n.StripAccents) || (n.StripAccents == nil && n.Lowercase) {
				s = removeAccents(norm.NFD.String(s))
			}
			if n.Lowercase {
				s = strings.ToLower(s)
			}
			// offsets is a per-BYTE map (result.String() is indexed by
			// byte, and downstream code treats offsets[i] as the
			// original-text position of normalized BYTE i). `for range
			// s` iterates s's RUNES, not its bytes, so for any `s` that
			// passes through as (or becomes) a multi-byte UTF-8
			// sequence — every accented Latin character on a cased
			// model that does not strip accents or lowercase, e.g.
			// á/é/í/ó/ú/ñ/ã/ç/ă/â/î/ș/ț/ü/ö/ä — this appended exactly
			// one offset entry while result.WriteString(s) wrote 2+
			// bytes, under-filling offsets by one per such character.
			// The deficit compounds across the string until downstream
			// code indexes past the now-too-short offsets slice
			// (observed live as `slice bounds out of range` panics on
			// Romanian/Spanish/Portuguese/CJK-heavy text). Iterate by
			// byte count instead so offsets always has exactly
			// len(result.String()) entries.
			for range len(s) {
				offsets = append(offsets, origPos)
			}
			result.WriteString(s)
		}
		origPos += runeLen
	}
	return result.String(), offsets
}

// --- Unicode Normalizer (NFD, NFC, NFKC, NFKD) ---

func applyUnicodeNormalizer(text string, n *Normalizer) string {
	switch n.Type {
	case "NFD":
		return norm.NFD.String(text)
	case "NFC":
		return norm.NFC.String(text)
	case "NFKC":
		return norm.NFKC.String(text)
	case "NFKD":
		return norm.NFKD.String(text)
	default:
		return text
	}
}

func applyUnicodeNormalizerWithOffsets(text string, n *Normalizer) (string, []int) {
	// Unicode normalization - approximate mapping
	normalized := applyUnicodeNormalizer(text, n)
	return approximateOffsets(text, normalized)
}

// --- StripAccents Normalizer ---

func applyStripAccentsNormalizer(text string) string {
	// NFD decomposition then remove combining marks (Mn category)
	return removeAccents(norm.NFD.String(text))
}

func applyStripAccentsNormalizerWithOffsets(text string) (string, []int) {
	// NFD then remove combining marks
	nfd := norm.NFD.String(text)
	var result strings.Builder
	var offsets []int
	origPos := 0
	for _, r := range nfd {
		runeLen := len(string(r))
		if !unicode.Is(unicode.Mn, r) {
			result.WriteRune(r)
			offsets = append(offsets, origPos)
		}
		origPos += runeLen
	}
	// Re-map offsets to original text positions
	return result.String(), remapOffsetsFromNFD(text, offsets)
}

// --- Replace Normalizer ---

func applyReplaceNormalizer(text string, n *Normalizer) string {
	if n.Pattern == nil {
		return text
	}
	if n.Pattern.String != "" {
		return strings.ReplaceAll(text, n.Pattern.String, n.Content)
	}
	if n.Pattern.Regex != "" {
		re, err := regexp.Compile(n.Pattern.Regex)
		if err == nil {
			return re.ReplaceAllString(text, n.Content)
		}
	}
	return text
}

func applyReplaceNormalizerWithOffsets(text string, n *Normalizer) (string, []int) {
	if n.Pattern == nil {
		return text, identityOffsets(len(text))
	}
	var re *regexp.Regexp
	var err error
	if n.Pattern.String != "" {
		re, err = regexp.Compile(regexp.QuoteMeta(n.Pattern.String))
	} else if n.Pattern.Regex != "" {
		re, err = regexp.Compile(n.Pattern.Regex)
	}
	if re == nil || err != nil {
		return text, identityOffsets(len(text))
	}

	matches := re.FindAllStringIndex(text, -1)
	if len(matches) == 0 {
		return text, identityOffsets(len(text))
	}

	var result strings.Builder
	var offsets []int
	cur := 0
	for _, m := range matches {
		mStart, mEnd := m[0], m[1]
		if mStart > cur {
			for i := cur; i < mStart; i++ {
				offsets = append(offsets, i)
			}
			result.WriteString(text[cur:mStart])
		}
		// For replaced content, map every byte of replacement to mStart
		for range len(n.Content) {
			offsets = append(offsets, mStart)
		}
		result.WriteString(n.Content)
		cur = mEnd
	}
	if cur < len(text) {
		for i := cur; i < len(text); i++ {
			offsets = append(offsets, i)
		}
		result.WriteString(text[cur:])
	}
	return result.String(), offsets
}

// --- Sequence Normalizer ---

func applySequenceNormalizer(text string, n *Normalizer) string {
	result := text
	for _, child := range n.Normalizers {
		childCopy := child
		result = applyNormalizer(result, &childCopy)
	}
	return result
}

func applySequenceNormalizerWithOffsets(text string, n *Normalizer) (string, []int) {
	result := text
	currentOffsets := identityOffsets(len(text))
	for _, child := range n.Normalizers {
		childCopy := child
		newResult, newOffsets := applyNormalizerWithOffsets(result, &childCopy)
		// Compose the offset mappings
		composedOffsets := make([]int, len(newOffsets))
		for i, off := range newOffsets {
			if off < len(currentOffsets) {
				composedOffsets[i] = currentOffsets[off]
			} else if len(currentOffsets) > 0 {
				composedOffsets[i] = currentOffsets[len(currentOffsets)-1]
			}
		}
		result = newResult
		currentOffsets = composedOffsets
	}
	return result, currentOffsets
}

// --- Prepend Normalizer ---

func applyPrependNormalizer(text string, _ *Normalizer) string {
	// Prepend a string (used by some tokenizers)
	return text
}

func applyPrependNormalizerWithOffsets(text string, _ *Normalizer) (string, []int) {
	return text, identityOffsets(len(text))
}

// --- Offset mapping utilities ---

// approximateOffsets creates an approximate offset mapping when exact tracking is too complex.
// It spreads the original text positions evenly across the normalized text using linear interpolation.
//
// WARNING: This function produces APPROXIMATE offsets that may not accurately reflect the true
// character-to-character mapping between original and normalized text. This is used as a fallback
// for complex normalizers (like certain Unicode normalizations) where exact tracking would require
// significantly more complexity. For token classification tasks (NER, chunking) that require precise
// character offsets, consider using tokenizers with simpler normalizers (e.g., Lowercase, BertNormalizer)
// that support exact offset tracking.
func approximateOffsets(original, normalized string) (string, []int) {
	if len(normalized) == 0 {
		return normalized, nil
	}
	if len(original) == 0 {
		return normalized, make([]int, len(normalized))
	}

	offsets := make([]int, len(normalized))
	ratio := float64(len(original)) / float64(len(normalized))

	for i := range offsets {
		offsets[i] = int(float64(i) * ratio)
		if offsets[i] >= len(original) {
			offsets[i] = len(original) - 1
		}
	}
	return normalized, offsets
}

// remapOffsetsFromNFD maps offsets from NFD-normalized text back to original text positions.
func remapOffsetsFromNFD(original string, nfdOffsets []int) []int {
	// This is an approximation - maps NFD positions to original positions
	nfd := norm.NFD.String(original)
	if len(nfd) == len(original) {
		return nfdOffsets // No change in length, direct mapping
	}

	// Build mapping from NFD position to original position
	nfdToOrig := make([]int, len(nfd))
	origPos := 0
	nfdPos := 0
	for _, r := range original {
		nfdRunes := []rune(norm.NFD.String(string(r)))
		for range nfdRunes {
			if nfdPos < len(nfdToOrig) {
				nfdToOrig[nfdPos] = origPos
				nfdPos++
			}
		}
		origPos += len(string(r))
	}

	// Remap the offsets
	result := make([]int, len(nfdOffsets))
	for i, off := range nfdOffsets {
		if off < len(nfdToOrig) {
			result[i] = nfdToOrig[off]
		} else if len(nfdToOrig) > 0 {
			result[i] = nfdToOrig[len(nfdToOrig)-1]
		}
	}
	return result
}

// --- Character classification and text cleaning helpers ---

func isChineseChar(r rune) bool {
	// CJK Unified Ideographs: 4E00-9FFF
	// CJK Unified Ideographs Extension A: 3400-4DBF
	// CJK Unified Ideographs Extension B: 20000-2A6DF
	// ...
	if (r >= 0x4E00 && r <= 0x9FFF) ||
		(r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) ||
		(r >= 0x2A700 && r <= 0x2B73F) ||
		(r >= 0x2B740 && r <= 0x2B81F) ||
		(r >= 0x2B820 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) ||
		(r >= 0x2F800 && r <= 0x2FA1F) {
		return true
	}
	return false
}

func tokenizeChineseChars(text string) string {
	var result strings.Builder
	for _, r := range text {
		if isChineseChar(r) {
			result.WriteRune(' ')
			result.WriteRune(r)
			result.WriteRune(' ')
		} else {
			result.WriteRune(r)
		}
	}
	return result.String()
}

func cleanText(text string) string {
	var result strings.Builder
	for _, r := range text {
		if r == 0 || r == 0xFFFD || isControl(r) {
			continue
		}
		if isChineseChar(r) {
			result.WriteRune(' ')
			result.WriteRune(r)
			result.WriteRune(' ')
		} else if isWhitespace(r) {
			result.WriteRune(' ')
		} else {
			result.WriteRune(r)
		}
	}
	return result.String()
}

func isWhitespace(r rune) bool {
	if r == ' ' || r == '\t' || r == '\n' || r == '\r' {
		return true
	}
	return unicode.Is(unicode.Zs, r)
}

func isControl(r rune) bool {
	if r == '\t' || r == '\n' || r == '\r' {
		return false
	}
	return unicode.IsControl(r)
}

func removeAccents(text string) string {
	// Simplified accent removal
	var result strings.Builder
	for _, r := range text {
		if !unicode.Is(unicode.Mn, r) { // Mn = Mark, Nonspacing
			result.WriteRune(r)
		}
	}
	return result.String()
}
