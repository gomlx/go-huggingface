// Package hftokenizer implements a tokenizer for HuggingFace's tokenizer.json format.
// This format is used by the HuggingFace Tokenizers library (the "fast" tokenizers)
// and supports WordPiece (BERT), BPE (GPT-2, RoBERTa), and Unigram models.
//
// It is written in pure Go.
package hftokenizer

import (
	"encoding/json"
	"os"
	"sort"
	"unicode"
	"unicode/utf8"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/pkg/errors"
)

// UnmarshalJSON implements custom unmarshaling to handle both vocab and merges formats:
// Vocab formats:
//  1. Object format: {"token": id, ...} (WordPiece, BPE)
//  2. Array format: [["token", score], ...] (Unigram) - ID is the array index
//
// Merges formats:
//  1. Array of strings: ["token1 token2", ...] (standard BPE)
//  2. Array of arrays: [["token1", "token2"], ...] (some models like embeddinggemma)
func (m *Model) UnmarshalJSON(data []byte) error {
	// Use an alias to avoid infinite recursion
	type ModelAlias Model
	type ModelWithRawFields struct {
		ModelAlias
		Vocab  json.RawMessage `json:"vocab"`
		Merges json.RawMessage `json:"merges"`
	}

	var raw ModelWithRawFields
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	// Copy all fields except Vocab and Merges
	*m = Model(raw.ModelAlias)
	m.Merges = nil // Clear since we'll handle it separately

	// Handle Vocab which can be either map or array
	if len(raw.Vocab) > 0 {
		// Try to unmarshal as map first (most common format)
		var vocabMap map[string]int
		if err := json.Unmarshal(raw.Vocab, &vocabMap); err == nil {
			m.Vocab = vocabMap
		} else {
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
			} else {
				m.Vocab = make(map[string]int)
			}
		}
	} else {
		m.Vocab = make(map[string]int)
	}

	// Handle Merges which can be array of strings or array of arrays
	if len(raw.Merges) > 0 {
		// Try array of strings first (standard format): ["a b", "c d", ...]
		var mergesStrings []string
		if err := json.Unmarshal(raw.Merges, &mergesStrings); err == nil {
			m.Merges = mergesStrings
		} else {
			// Try array of arrays: [["a", "b"], ["c", "d"], ...]
			var mergesArrays [][]string
			if err := json.Unmarshal(raw.Merges, &mergesArrays); err == nil {
				m.Merges = make([]string, len(mergesArrays))
				for i, pair := range mergesArrays {
					if len(pair) == 2 {
						// Join the pair with a space to match standard format
						m.Merges[i] = pair[0] + " " + pair[1]
					}
				}
			}
		}
	}

	return nil
}

// Compile time assert that Tokenizer implements api.Tokenizer interface.
var _ api.Tokenizer = &Tokenizer{}

// New creates a HuggingFace tokenizer from the tokenizer.json file.
// It implements a tokenizer.TokenizerConstructor function signature.
func New(config *api.Config, repo *hub.Repo) (api.Tokenizer, error) {
	if !repo.HasFile("tokenizer.json") {
		return nil, errors.Errorf("\"tokenizer.json\" file not found in repo")
	}
	content, err := repo.ReadFile("tokenizer.json")
	if err != nil {
		return nil, errors.Wrapf(err, "can't read tokenizer.json file")
	}
	return NewFromContent(config, content)
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

	err := compileDecoderRegex(tj.Decoder)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to compile decoder regex")
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
		options: api.EncodeOptions{
			AddSpecialTokens: true,
		},
	}

	// Build reverse vocab (id -> token)
	for token, id := range tj.Model.Vocab {
		t.idToToken[id] = token
	}

	// Build added tokens map and sorted list for splitting
	for _, at := range tj.AddedTokens {
		t.addedTokens[at.Content] = at.ID
		t.idToToken[at.ID] = at.Content
		t.addedTokensSorted = append(t.addedTokensSorted, addedTokenEntry{content: at.Content, id: at.ID})
	}
	// Sort longest-first for greedy matching
	sort.Slice(t.addedTokensSorted, func(i, j int) bool {
		return len(t.addedTokensSorted[i].content) > len(t.addedTokensSorted[j].content)
	})

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

// Normalize the text using the tokenizer normalization rules (e.g.: BERT lower cases the string).
func (t *Tokenizer) Normalize(text string) string {
	if t.tokenizer.Normalizer == nil {
		return text
	}
	return t.applyNormalizer(text, t.tokenizer.Normalizer)
}

// With applies options to a tokenizer.
func (t *Tokenizer) With(options api.EncodeOptions) error {
	t.options = options
	return nil
}

func (t *Tokenizer) Encode(text string) []int {
	result := t.encodeCore(text)
	if t.options.AddSpecialTokens {
		result.IDs, result.Spans, _ = t.applyPostProcessor(result.IDs, result.Spans)
	}
	return result.IDs
}

// EncodeWithAnnotations returns the encoded text along with requested annotations.
func (t *Tokenizer) EncodeWithAnnotations(text string) api.AnnotatedEncoding {
	result := t.encodeCore(text)
	var specialTokensMask []int
	if t.options.AddSpecialTokens {
		result.IDs, result.Spans, specialTokensMask = t.applyPostProcessor(result.IDs, result.Spans)
	}
	if !t.options.IncludeSpans {
		result.Spans = nil
	}
	if t.options.IncludeSpecialTokensMask {
		result.SpecialTokensMask = specialTokensMask
	}
	return result
}

// wordWithOffset holds a word/token string along with its character offset in the original text.
type wordWithOffset struct {
	text    string
	start   int   // start position in original text (inclusive)
	end     int   // end position in original text (exclusive)
	offsets []int // byte mapping for each byte of text into original text (len == len(text))
}

// encodeCore runs the core tokenization pipeline (split added tokens → normalize →
// pre-tokenize → tokenize) without post-processing.
func (t *Tokenizer) encodeCore(text string) api.AnnotatedEncoding {
	segments := t.splitOnAddedTokens(text)

	var ids []int
	var spans []api.TokenSpan

	for _, seg := range segments {
		if seg.isAddedToken {
			ids = append(ids, seg.tokenID)
			spans = append(spans, api.TokenSpan{Start: seg.start, End: seg.end})
			continue
		}

		segText := text[seg.start:seg.end]

		normalized, normOffsets := t.normalizeWithOffsets(segText)
		for i := range normOffsets {
			normOffsets[i] += seg.start
		}

		words := t.preTokenizeWithOffsets(normalized, normOffsets)

		for _, word := range words {
			wordIDs, wordSpans := t.tokenizeWordWithSpans(word)
			ids = append(ids, wordIDs...)
			spans = append(spans, wordSpans...)
		}
	}

	return api.AnnotatedEncoding{
		IDs:   ids,
		Spans: spans,
	}
}

// parseTokenIDTuple parses a JSON [string, int] tuple (e.g., ["[CLS]", 101])
// used by BertProcessing and RobertaProcessing.
func parseTokenIDTuple(raw json.RawMessage) (int, bool) {
	if len(raw) == 0 {
		return 0, false
	}
	var tuple [2]json.RawMessage
	if err := json.Unmarshal(raw, &tuple); err != nil {
		return 0, false
	}
	var id int
	if err := json.Unmarshal(tuple[1], &id); err != nil {
		return 0, false
	}
	return id, true
}

// addedTokenEntry pairs a token string with its ID for efficient matching.
type addedTokenEntry struct {
	content string
	id      int
}

// textSegment represents a piece of input text, either an added token or regular text.
type textSegment struct {
	start        int  // byte offset in original text
	end          int  // byte offset in original text
	isAddedToken bool // true if this segment matches an added token
	tokenID      int  // only valid if isAddedToken is true
}

// splitOnAddedTokens splits text into segments of added tokens and regular text.
// Added tokens are matched greedily (longest first).
func (t *Tokenizer) splitOnAddedTokens(text string) []textSegment {
	if len(text) == 0 {
		return nil
	}
	if len(t.addedTokensSorted) == 0 {
		return []textSegment{{start: 0, end: len(text)}}
	}

	var segments []textSegment
	regularStart := 0 // start of current regular text run
	pos := 0

	for pos < len(text) {
		matched := false
		for _, entry := range t.addedTokensSorted {
			if pos+len(entry.content) <= len(text) && text[pos:pos+len(entry.content)] == entry.content {
				// Flush any preceding regular text
				if regularStart < pos {
					segments = append(segments, textSegment{start: regularStart, end: pos})
				}
				segments = append(segments, textSegment{
					start:        pos,
					end:          pos + len(entry.content),
					isAddedToken: true,
					tokenID:      entry.id,
				})
				pos += len(entry.content)
				regularStart = pos
				matched = true
				break
			}
		}
		if !matched {
			_, size := utf8.DecodeRuneInString(text[pos:])
			pos += size
		}
	}

	// Flush any trailing regular text
	if regularStart < len(text) {
		segments = append(segments, textSegment{start: regularStart, end: len(text)})
	}

	return segments
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

// Config returns the HuggingFace tokenizer configuration.
func (t *Tokenizer) Config() *api.Config {
	return t.config
}

// Helper functions

func isPunctuation(r rune) bool {
	// ASCII punctuation
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) ||
		(r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.IsPunct(r)
}

// Byte-level BPE encoding/decoding
// GPT-2 uses a specific byte-to-unicode mapping
var (
	byteToUnicode map[byte]rune
	unicodeToByte map[rune]byte
)

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
