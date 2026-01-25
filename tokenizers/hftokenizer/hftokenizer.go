// Package hftokenizer implements a tokenizer for HuggingFace's tokenizer.json format.
// This format is used by the HuggingFace Tokenizers library (the "fast" tokenizers)
// and supports WordPiece (BERT), BPE (GPT-2, RoBERTa), and Unigram models.
package hftokenizer

import (
	"encoding/json"
	"os"
	"sort"
	"strings"
	"unicode"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/pkg/errors"
	"golang.org/x/text/unicode/norm"
)

// TokenizerJSON represents the structure of HuggingFace's tokenizer.json file.
type TokenizerJSON struct {
	Version       string          `json:"version"`
	Truncation    json.RawMessage `json:"truncation"`
	Padding       json.RawMessage `json:"padding"`
	AddedTokens   []AddedToken    `json:"added_tokens"`
	Normalizer    *Normalizer     `json:"normalizer"`
	PreTokenizer  *PreTokenizer   `json:"pre_tokenizer"`
	PostProcessor *PostProcessor  `json:"post_processor"`
	Decoder       *Decoder        `json:"decoder"`
	Model         Model           `json:"model"`
}

// AddedToken represents a special token added to the vocabulary.
type AddedToken struct {
	ID         int    `json:"id"`
	Content    string `json:"content"`
	SingleWord bool   `json:"single_word"`
	Lstrip     bool   `json:"lstrip"`
	Rstrip     bool   `json:"rstrip"`
	Normalized bool   `json:"normalized"`
	Special    bool   `json:"special"`
}

// Normalizer represents the normalizer configuration.
type Normalizer struct {
	Type       string       `json:"type"`
	Lowercase  bool         `json:"lowercase"`
	Normalizer *Normalizer  `json:"normalizer"`
	Pattern    *Pattern     `json:"pattern"`
	Normalizers []Normalizer `json:"normalizers"`
}

// Pattern for regex-based operations.
type Pattern struct {
	Regex  string `json:"Regex,omitempty"`
	String string `json:"String,omitempty"`
}

// PreTokenizer represents the pre-tokenizer configuration.
type PreTokenizer struct {
	Type           string         `json:"type"`
	AddPrefixSpace bool           `json:"add_prefix_space"`
	PreTokenizers  []PreTokenizer `json:"pretokenizers"`
	Pattern        *Pattern       `json:"pattern"`
	Behavior       string         `json:"behavior"`
	Invert         bool           `json:"invert"`
}

// PostProcessor represents the post-processor configuration.
type PostProcessor struct {
	Type          string         `json:"type"`
	Single        []PostProcItem `json:"single"`
	Pair          []PostProcItem `json:"pair"`
	SpecialTokens map[string]PostProcSpecialToken `json:"special_tokens"`
}

// PostProcItem is an item in post-processing.
type PostProcItem struct {
	ID           string `json:"id,omitempty"`
	TypeID       int    `json:"type_id"`
	SpecialToken *struct {
		ID     string `json:"id"`
		TypeID int    `json:"type_id"`
	} `json:"SpecialToken,omitempty"`
	Sequence *struct {
		ID     string `json:"id"`
		TypeID int    `json:"type_id"`
	} `json:"Sequence,omitempty"`
}

// PostProcSpecialToken defines a special token for post-processing.
type PostProcSpecialToken struct {
	ID     string   `json:"id"`
	IDs    []int    `json:"ids"`
	Tokens []string `json:"tokens"`
}

// Decoder represents the decoder configuration.
type Decoder struct {
	Type       string    `json:"type"`
	Prefix     string    `json:"prefix"`
	Suffix     string    `json:"suffix"`
	Decoders   []Decoder `json:"decoders"`
	Pattern    *Pattern  `json:"pattern"`
	Content    string    `json:"content"`
}

// Model represents the tokenizer model (WordPiece, BPE, or Unigram).
type Model struct {
	Type                    string         `json:"type"`
	Vocab                   map[string]int `json:"-"` // Custom unmarshaling handles both map and array formats
	Merges                  []string       `json:"merges"`
	UnkToken                string         `json:"unk_token"`
	ContinuingSubwordPrefix string         `json:"continuing_subword_prefix"`
	MaxInputCharsPerWord    int            `json:"max_input_chars_per_word"`
	FuseUnk                 bool           `json:"fuse_unk"`
	ByteFallback            bool           `json:"byte_fallback"`
	Dropout                 *float64       `json:"dropout"`
	EndOfWordSuffix         string         `json:"end_of_word_suffix"`
}

// UnmarshalJSON implements custom unmarshaling to handle both vocab formats:
// 1. Object format: {"token": id, ...} (WordPiece, BPE)
// 2. Array format: [["token", score], ...] (Unigram) - ID is the array index
func (m *Model) UnmarshalJSON(data []byte) error {
	// Use an alias to avoid infinite recursion
	type ModelAlias Model
	type ModelWithRawVocab struct {
		ModelAlias
		Vocab json.RawMessage `json:"vocab"`
	}

	var raw ModelWithRawVocab
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	// Copy all fields except Vocab
	*m = Model(raw.ModelAlias)

	// Now handle Vocab which can be either map or array
	if len(raw.Vocab) == 0 {
		m.Vocab = make(map[string]int)
		return nil
	}

	// Try to unmarshal as map first (most common format)
	var vocabMap map[string]int
	if err := json.Unmarshal(raw.Vocab, &vocabMap); err == nil {
		m.Vocab = vocabMap
		return nil
	}

	// Try array format: [["token", score], ...] (Unigram models)
	// For Unigram, the second element is a score (log probability), not an ID.
	// The token's ID is its index/position in the array.
	var vocabArray [][]interface{}
	if err := json.Unmarshal(raw.Vocab, &vocabArray); err == nil {
		m.Vocab = make(map[string]int, len(vocabArray))
		for idx, pair := range vocabArray {
			if len(pair) >= 1 {
				token, ok := pair[0].(string)
				if ok {
					// Use array index as the token ID
					m.Vocab[token] = idx
				}
			}
		}
		return nil
	}

	// If neither format works, return empty vocab
	m.Vocab = make(map[string]int)
	return nil
}

// Tokenizer implements the api.Tokenizer interface for HuggingFace tokenizer.json files.
type Tokenizer struct {
	config     *api.Config
	tokenizer  *TokenizerJSON
	idToToken  map[int]string
	mergeRanks map[string]int // For BPE: maps "token1 token2" to merge priority

	// Special token IDs
	unkID  int
	padID  int
	bosID  int
	eosID  int
	clsID  int
	sepID  int
	maskID int

	// Added tokens lookup (content -> id)
	addedTokens map[string]int
}

// Compile time assert that Tokenizer implements api.Tokenizer interface.
var _ api.Tokenizer = &Tokenizer{}

// Compile time assert that Tokenizer implements api.TokenizerWithSpans interface.
var _ api.TokenizerWithSpans = &Tokenizer{}

// New creates a HuggingFace tokenizer from the tokenizer.json file.
// It implements a tokenizer.TokenizerConstructor function signature.
func New(config *api.Config, repo *hub.Repo) (api.Tokenizer, error) {
	if !repo.HasFile("tokenizer.json") {
		return nil, errors.Errorf("\"tokenizer.json\" file not found in repo")
	}
	tokenizerFile, err := repo.DownloadFile("tokenizer.json")
	if err != nil {
		return nil, errors.Wrapf(err, "can't download tokenizer.json file")
	}
	return NewFromFile(config, tokenizerFile)
}

// NewFromFile creates a HuggingFace tokenizer from a local tokenizer.json file path.
func NewFromFile(config *api.Config, filePath string) (*Tokenizer, error) {
	content, err := os.ReadFile(filePath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read tokenizer.json file %q", filePath)
	}
	return NewFromContent(config, content)
}

// NewFromContent creates a HuggingFace tokenizer from tokenizer.json content.
func NewFromContent(config *api.Config, content []byte) (*Tokenizer, error) {
	var tj TokenizerJSON
	if err := json.Unmarshal(content, &tj); err != nil {
		return nil, errors.Wrapf(err, "failed to parse tokenizer.json")
	}

	t := &Tokenizer{
		config:      config,
		tokenizer:   &tj,
		idToToken:   make(map[int]string),
		addedTokens: make(map[string]int),
		unkID:       -1,
		padID:       -1,
		bosID:       -1,
		eosID:       -1,
		clsID:       -1,
		sepID:       -1,
		maskID:      -1,
	}

	// Build reverse vocab (id -> token)
	for token, id := range tj.Model.Vocab {
		t.idToToken[id] = token
	}

	// Build added tokens map
	for _, at := range tj.AddedTokens {
		t.addedTokens[at.Content] = at.ID
		t.idToToken[at.ID] = at.Content
	}

	// Build merge ranks for BPE
	if tj.Model.Type == "BPE" {
		t.mergeRanks = make(map[string]int)
		for i, merge := range tj.Model.Merges {
			t.mergeRanks[merge] = i
		}
	}

	// Resolve special token IDs
	t.resolveSpecialTokens()

	return t, nil
}

// resolveSpecialTokens maps special tokens from config to their IDs.
func (t *Tokenizer) resolveSpecialTokens() {
	// First check the model's unk_token
	if t.tokenizer.Model.UnkToken != "" {
		if id, ok := t.tokenizer.Model.Vocab[t.tokenizer.Model.UnkToken]; ok {
			t.unkID = id
		}
	}

	// Then check added tokens for special tokens
	for _, at := range t.tokenizer.AddedTokens {
		if !at.Special {
			continue
		}
		content := at.Content
		switch {
		case content == "[UNK]" || content == "<unk>":
			t.unkID = at.ID
		case content == "[PAD]" || content == "<pad>":
			t.padID = at.ID
		case content == "[CLS]" || content == "<s>":
			t.clsID = at.ID
		case content == "[SEP]" || content == "</s>":
			t.sepID = at.ID
		case content == "[MASK]" || content == "<mask>":
			t.maskID = at.ID
		}
		// Also check for BOS/EOS
		if t.config != nil {
			if content == t.config.BosToken {
				t.bosID = at.ID
			}
			if content == t.config.EosToken {
				t.eosID = at.ID
			}
		}
	}

	// Fall back to config special tokens if available
	if t.config != nil {
		if t.unkID == -1 && t.config.UnkToken != "" {
			if id, ok := t.tokenizer.Model.Vocab[t.config.UnkToken]; ok {
				t.unkID = id
			}
		}
		if t.padID == -1 && t.config.PadToken != "" {
			if id, ok := t.tokenizer.Model.Vocab[t.config.PadToken]; ok {
				t.padID = id
			}
		}
		if t.clsID == -1 && t.config.ClsToken != "" {
			if id, ok := t.tokenizer.Model.Vocab[t.config.ClsToken]; ok {
				t.clsID = id
			}
		}
		if t.sepID == -1 && t.config.SepToken != "" {
			if id, ok := t.tokenizer.Model.Vocab[t.config.SepToken]; ok {
				t.sepID = id
			}
		}
		if t.maskID == -1 && t.config.MaskToken != "" {
			if id, ok := t.tokenizer.Model.Vocab[t.config.MaskToken]; ok {
				t.maskID = id
			}
		}
		if t.bosID == -1 && t.config.BosToken != "" {
			if id, ok := t.tokenizer.Model.Vocab[t.config.BosToken]; ok {
				t.bosID = id
			}
		}
		if t.eosID == -1 && t.config.EosToken != "" {
			if id, ok := t.tokenizer.Model.Vocab[t.config.EosToken]; ok {
				t.eosID = id
			}
		}
	}
}

// Encode converts text to a sequence of token IDs.
//
// Note: This delegates to EncodeWithSpans for simplicity. The overhead of computing
// spans is minimal (just tracking byte positions during tokenization), so we avoid
// maintaining duplicate code paths. If profiling shows this to be a bottleneck for
// your use case, please open an issue.
func (t *Tokenizer) Encode(text string) []int {
	result := t.EncodeWithSpans(text)
	return result.IDs
}

// wordWithOffset holds a word/token string along with its character offset in the original text.
type wordWithOffset struct {
	text  string
	start int // start position in original text (inclusive)
	end   int // end position in original text (exclusive)
}

// EncodeWithSpans converts text to a sequence of token IDs along with their byte spans.
func (t *Tokenizer) EncodeWithSpans(text string) api.EncodingResult {
	// Apply normalization with span tracking
	normalized, normSpans := t.normalizeWithSpans(text)

	// Apply pre-tokenization with span tracking
	words := t.preTokenizeWithSpans(normalized, normSpans)

	// Tokenize each word according to the model type
	var ids []int
	var spans []api.TokenSpan

	for _, word := range words {
		wordIDs, wordSpans := t.tokenizeWordWithSpans(word)
		ids = append(ids, wordIDs...)
		spans = append(spans, wordSpans...)
	}

	return api.EncodingResult{
		IDs:   ids,
		Spans: spans,
	}
}

// normalizeWithSpans applies normalization and returns the normalized text along with
// a mapping from normalized byte positions to original byte positions.
// The returned slice maps normalized position -> original position.
func (t *Tokenizer) normalizeWithSpans(text string) (string, []int) {
	if t.tokenizer.Normalizer == nil {
		// No normalization - create identity mapping
		offsets := make([]int, len(text))
		for i := range text {
			offsets[i] = i
		}
		return text, offsets
	}
	return t.applyNormalizerWithSpans(text, t.tokenizer.Normalizer)
}

// applyNormalizerWithSpans applies a normalizer and tracks byte positions.
func (t *Tokenizer) applyNormalizerWithSpans(text string, n *Normalizer) (string, []int) {
	// For most normalizers, we need to track how characters map through the transformation.
	// This is complex because normalizers can:
	// 1. Remove characters (accents, control chars)
	// 2. Replace characters (lowercase)
	// 3. Expand characters (NFD decomposition)
	// 4. Contract characters (NFC composition)
	//
	// For simplicity, we handle the common cases and fall back to approximate mapping for complex cases.

	switch n.Type {
	case "Lowercase":
		// Lowercase preserves character positions (1:1 mapping)
		normalized := strings.ToLower(text)
		offsets := make([]int, len(normalized))
		origPos := 0
		normPos := 0
		for _, r := range text {
			lowerRunes := []rune(strings.ToLower(string(r)))
			for range lowerRunes {
				if normPos < len(offsets) {
					offsets[normPos] = origPos
					normPos++
				}
			}
			origPos += len(string(r))
		}
		return normalized, offsets

	case "BertNormalizer":
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
			if isWhitespace(r) {
				result.WriteRune(' ')
				offsets = append(offsets, origPos)
			} else if n.Lowercase {
				lower := strings.ToLower(string(r))
				for range lower {
					offsets = append(offsets, origPos)
				}
				result.WriteString(lower)
			} else {
				result.WriteRune(r)
				offsets = append(offsets, origPos)
			}
			origPos += runeLen
		}
		return result.String(), offsets

	case "NFD", "NFC", "NFKC", "NFKD":
		// Unicode normalization - approximate mapping
		normalized := t.applyNormalizer(text, n)
		return approximateOffsets(text, normalized)

	case "StripAccents":
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

	case "Sequence":
		result := text
		currentOffsets := make([]int, len(text))
		for i := range text {
			currentOffsets[i] = i
		}
		for _, child := range n.Normalizers {
			childCopy := child
			newResult, newOffsets := t.applyNormalizerWithSpans(result, &childCopy)
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

	default:
		// Unknown normalizer - use approximate mapping
		normalized := t.applyNormalizer(text, n)
		return approximateOffsets(text, normalized)
	}
}

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
		return metaspacePreTokenizeWithOffsets(text, normOffsets, pt.AddPrefixSpace)
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
func metaspacePreTokenizeWithOffsets(text string, normOffsets []int, addPrefixSpace bool) []wordWithOffset {
	if addPrefixSpace && len(text) > 0 && text[0] != ' ' {
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
			if current.Len() > 0 {
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
			}
			current.WriteRune('\u2581')
			currentStart = i
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

func (t *Tokenizer) applyNormalizer(text string, n *Normalizer) string {
	switch n.Type {
	case "Lowercase":
		return strings.ToLower(text)
	case "NFD":
		return norm.NFD.String(text)
	case "NFC":
		return norm.NFC.String(text)
	case "NFKC":
		return norm.NFKC.String(text)
	case "NFKD":
		return norm.NFKD.String(text)
	case "StripAccents":
		// NFD decomposition then remove combining marks (Mn category)
		return removeAccents(norm.NFD.String(text))
	case "BertNormalizer":
		// Clean text, handle Chinese chars, strip accents, lowercase
		result := cleanText(text)
		if n.Lowercase {
			result = strings.ToLower(result)
		}
		return result
	case "Sequence":
		result := text
		for _, child := range n.Normalizers {
			childCopy := child
			result = t.applyNormalizer(result, &childCopy)
		}
		return result
	case "Replace":
		// Handle replace patterns if needed
		return text
	case "Prepend":
		// Prepend a string (used by some tokenizers)
		return text
	default:
		return text
	}
}

// Decode converts a sequence of token IDs back to text.
func (t *Tokenizer) Decode(ids []int) string {
	var tokens []string
	for _, id := range ids {
		if token, ok := t.idToToken[id]; ok {
			tokens = append(tokens, token)
		}
	}

	// Apply decoder
	result := t.applyDecoder(tokens)
	return result
}

// applyDecoder applies the decoder to convert tokens back to text.
func (t *Tokenizer) applyDecoder(tokens []string) string {
	if t.tokenizer.Decoder == nil {
		// Default: handle WordPiece-style decoding
		return t.defaultDecode(tokens)
	}

	switch t.tokenizer.Decoder.Type {
	case "WordPiece":
		return t.wordPieceDecode(tokens)
	case "ByteLevel":
		return t.byteLevelDecode(tokens)
	case "Metaspace":
		return t.metaspaceDecode(tokens)
	case "BPEDecoder":
		return t.bpeDecode(tokens)
	case "Sequence":
		result := tokens
		for _, dec := range t.tokenizer.Decoder.Decoders {
			decCopy := dec
			result = t.applyDecoderStep(result, &decCopy)
		}
		return strings.Join(result, "")
	default:
		return t.defaultDecode(tokens)
	}
}

func (t *Tokenizer) applyDecoderStep(tokens []string, d *Decoder) []string {
	switch d.Type {
	case "Replace":
		// Replace pattern in tokens
		var result []string
		for _, tok := range tokens {
			result = append(result, tok)
		}
		return result
	case "Strip":
		// Strip characters
		return tokens
	case "ByteFallback":
		// Handle byte fallback decoding
		return tokens
	default:
		return tokens
	}
}

func (t *Tokenizer) defaultDecode(tokens []string) string {
	prefix := t.tokenizer.Model.ContinuingSubwordPrefix
	if prefix == "" {
		prefix = "##"
	}

	var result strings.Builder
	for i, token := range tokens {
		if strings.HasPrefix(token, prefix) {
			result.WriteString(strings.TrimPrefix(token, prefix))
		} else {
			if i > 0 {
				result.WriteString(" ")
			}
			result.WriteString(token)
		}
	}
	return result.String()
}

func (t *Tokenizer) wordPieceDecode(tokens []string) string {
	prefix := t.tokenizer.Decoder.Prefix
	if prefix == "" {
		prefix = "##"
	}

	var result strings.Builder
	for i, token := range tokens {
		if strings.HasPrefix(token, prefix) {
			result.WriteString(strings.TrimPrefix(token, prefix))
		} else {
			if i > 0 {
				result.WriteString(" ")
			}
			result.WriteString(token)
		}
	}
	return result.String()
}

func (t *Tokenizer) byteLevelDecode(tokens []string) string {
	// Join tokens and decode byte-level representation
	text := strings.Join(tokens, "")
	// The byte-level encoding uses special unicode characters
	// We need to map them back to bytes
	return byteLevelDecode(text)
}

func (t *Tokenizer) metaspaceDecode(tokens []string) string {
	var result strings.Builder
	for _, token := range tokens {
		// Metaspace replaces leading space with special char
		decoded := strings.ReplaceAll(token, "\u2581", " ")
		result.WriteString(decoded)
	}
	return strings.TrimLeft(result.String(), " ")
}

func (t *Tokenizer) bpeDecode(tokens []string) string {
	suffix := t.tokenizer.Model.EndOfWordSuffix

	var result strings.Builder
	for i, token := range tokens {
		if suffix != "" && strings.HasSuffix(token, suffix) {
			result.WriteString(strings.TrimSuffix(token, suffix))
			if i < len(tokens)-1 {
				result.WriteString(" ")
			}
		} else {
			result.WriteString(token)
		}
	}
	return result.String()
}

// SpecialTokenID returns the ID for a given special token.
func (t *Tokenizer) SpecialTokenID(token api.SpecialToken) (int, error) {
	switch token {
	case api.TokUnknown:
		if t.unkID >= 0 {
			return t.unkID, nil
		}
	case api.TokPad:
		if t.padID >= 0 {
			return t.padID, nil
		}
	case api.TokBeginningOfSentence:
		if t.bosID >= 0 {
			return t.bosID, nil
		}
		// Fall back to CLS for BERT-style models
		if t.clsID >= 0 {
			return t.clsID, nil
		}
	case api.TokEndOfSentence:
		if t.eosID >= 0 {
			return t.eosID, nil
		}
		// Fall back to SEP for BERT-style models
		if t.sepID >= 0 {
			return t.sepID, nil
		}
	case api.TokMask:
		if t.maskID >= 0 {
			return t.maskID, nil
		}
	case api.TokClassification:
		if t.clsID >= 0 {
			return t.clsID, nil
		}
	}
	return 0, errors.Errorf("special token %s not found", token)
}

// VocabSize returns the size of the vocabulary.
func (t *Tokenizer) VocabSize() int {
	return len(t.tokenizer.Model.Vocab) + len(t.tokenizer.AddedTokens)
}

// GetVocab returns the full vocabulary mapping.
func (t *Tokenizer) GetVocab() map[string]int {
	vocab := make(map[string]int)
	for k, v := range t.tokenizer.Model.Vocab {
		vocab[k] = v
	}
	for _, at := range t.tokenizer.AddedTokens {
		vocab[at.Content] = at.ID
	}
	return vocab
}

// Helper functions

func cleanText(text string) string {
	var result strings.Builder
	for _, r := range text {
		if r == 0 || r == 0xFFFD || isControl(r) {
			continue
		}
		if isWhitespace(r) {
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

func isPunctuation(r rune) bool {
	// ASCII punctuation
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) ||
		(r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.IsPunct(r)
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

// Byte-level BPE encoding/decoding
// GPT-2 uses a specific byte-to-unicode mapping
var byteToUnicode map[byte]rune
var unicodeToByte map[rune]byte

func init() {
	byteToUnicode = make(map[byte]rune)
	unicodeToByte = make(map[rune]byte)

	// Build the byte-to-unicode mapping used by GPT-2
	n := 0
	for b := 0; b < 256; b++ {
		if (b >= '!' && b <= '~') || (b >= '\xa1' && b <= '\xac') || (b >= '\xae' && b <= '\xff') {
			byteToUnicode[byte(b)] = rune(b)
			unicodeToByte[rune(b)] = byte(b)
		} else {
			byteToUnicode[byte(b)] = rune(256 + n)
			unicodeToByte[rune(256+n)] = byte(b)
			n++
		}
	}
}

func byteLevelDecode(text string) string {
	var result []byte
	for _, r := range text {
		if b, ok := unicodeToByte[r]; ok {
			result = append(result, b)
		} else {
			// Fallback for characters not in the mapping
			result = append(result, []byte(string(r))...)
		}
	}
	return string(result)
}

// GetTokenizerType returns the model type (WordPiece, BPE, Unigram).
func (t *Tokenizer) GetTokenizerType() string {
	return t.tokenizer.Model.Type
}

// TokenToID converts a token string to its ID.
func (t *Tokenizer) TokenToID(token string) (int, bool) {
	if id, ok := t.addedTokens[token]; ok {
		return id, true
	}
	id, ok := t.tokenizer.Model.Vocab[token]
	return id, ok
}

// IDToToken converts a token ID to its string.
func (t *Tokenizer) IDToToken(id int) (string, bool) {
	token, ok := t.idToToken[id]
	return token, ok
}

// AddedTokensList returns the list of added tokens sorted by ID.
func (t *Tokenizer) AddedTokensList() []AddedToken {
	result := make([]AddedToken, len(t.tokenizer.AddedTokens))
	copy(result, t.tokenizer.AddedTokens)
	sort.Slice(result, func(i, j int) bool {
		return result[i].ID < result[j].ID
	})
	return result
}
