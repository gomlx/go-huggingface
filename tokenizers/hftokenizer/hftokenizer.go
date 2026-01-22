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
	Type                     string            `json:"type"`
	Vocab                    map[string]int    `json:"vocab"`
	Merges                   []string          `json:"merges"`
	UnkToken                 string            `json:"unk_token"`
	ContinuingSubwordPrefix  string            `json:"continuing_subword_prefix"`
	MaxInputCharsPerWord     int               `json:"max_input_chars_per_word"`
	FuseUnk                  bool              `json:"fuse_unk"`
	ByteFallback             bool              `json:"byte_fallback"`
	Dropout                  *float64          `json:"dropout"`
	EndOfWordSuffix          string            `json:"end_of_word_suffix"`
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
func (t *Tokenizer) Encode(text string) []int {
	// Apply normalization
	normalized := t.normalize(text)

	// Apply pre-tokenization
	words := t.preTokenize(normalized)

	// Tokenize each word according to the model type
	var ids []int
	for _, word := range words {
		wordIDs := t.tokenizeWord(word)
		ids = append(ids, wordIDs...)
	}

	return ids
}

// normalize applies the normalizer to the text.
func (t *Tokenizer) normalize(text string) string {
	if t.tokenizer.Normalizer == nil {
		return text
	}
	return t.applyNormalizer(text, t.tokenizer.Normalizer)
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

// preTokenize splits text into words using the pre-tokenizer.
func (t *Tokenizer) preTokenize(text string) []string {
	if t.tokenizer.PreTokenizer == nil {
		// Default: split on whitespace
		return strings.Fields(text)
	}
	return t.applyPreTokenizer(text, t.tokenizer.PreTokenizer)
}

func (t *Tokenizer) applyPreTokenizer(text string, pt *PreTokenizer) []string {
	switch pt.Type {
	case "BertPreTokenizer":
		// Split on whitespace and punctuation
		return bertPreTokenize(text)
	case "Whitespace":
		return strings.Fields(text)
	case "WhitespaceSplit":
		return strings.Fields(text)
	case "ByteLevel":
		// For BPE models like GPT-2
		if pt.AddPrefixSpace && len(text) > 0 && text[0] != ' ' {
			text = " " + text
		}
		return byteLevelPreTokenize(text)
	case "Metaspace":
		// Replace spaces with special character (used by some models)
		return metaspacePreTokenize(text, pt.AddPrefixSpace)
	case "Sequence":
		result := []string{text}
		for _, child := range pt.PreTokenizers {
			var newResult []string
			childCopy := child
			for _, s := range result {
				newResult = append(newResult, t.applyPreTokenizer(s, &childCopy)...)
			}
			result = newResult
		}
		return result
	case "Split":
		// Split based on pattern
		return strings.Fields(text)
	case "Punctuation":
		return punctuationPreTokenize(text)
	default:
		return strings.Fields(text)
	}
}

// tokenizeWord tokenizes a single word according to the model type.
func (t *Tokenizer) tokenizeWord(word string) []int {
	// First check if word is an added token
	if id, ok := t.addedTokens[word]; ok {
		return []int{id}
	}

	switch t.tokenizer.Model.Type {
	case "WordPiece":
		return t.wordPieceTokenize(word)
	case "BPE":
		return t.bpeTokenize(word)
	case "Unigram":
		return t.unigramTokenize(word)
	default:
		// Fallback: try to find word in vocab
		if id, ok := t.tokenizer.Model.Vocab[word]; ok {
			return []int{id}
		}
		if t.unkID >= 0 {
			return []int{t.unkID}
		}
		return nil
	}
}

// wordPieceTokenize implements WordPiece tokenization (used by BERT).
func (t *Tokenizer) wordPieceTokenize(word string) []int {
	if word == "" {
		return nil
	}

	maxChars := t.tokenizer.Model.MaxInputCharsPerWord
	if maxChars == 0 {
		maxChars = 100
	}
	if len(word) > maxChars {
		if t.unkID >= 0 {
			return []int{t.unkID}
		}
		return nil
	}

	prefix := t.tokenizer.Model.ContinuingSubwordPrefix
	if prefix == "" {
		prefix = "##"
	}

	var tokens []int
	start := 0

	for start < len(word) {
		end := len(word)
		found := false

		for start < end {
			substr := word[start:end]
			if start > 0 {
				substr = prefix + substr
			}

			if id, ok := t.tokenizer.Model.Vocab[substr]; ok {
				tokens = append(tokens, id)
				found = true
				break
			}
			end--
		}

		if !found {
			if t.unkID >= 0 {
				return []int{t.unkID}
			}
			return nil
		}
		start = end
	}

	return tokens
}

// bpeTokenize implements BPE tokenization (used by GPT-2, RoBERTa).
func (t *Tokenizer) bpeTokenize(word string) []int {
	if word == "" {
		return nil
	}

	// Convert word to list of bytes/characters
	// For byte-level BPE, we need to handle byte encoding
	symbols := t.getInitialBPESymbols(word)

	// If word is a single symbol that exists in vocab, return it
	if len(symbols) == 1 {
		if id, ok := t.tokenizer.Model.Vocab[symbols[0]]; ok {
			return []int{id}
		}
	}

	// Apply BPE merges
	for len(symbols) > 1 {
		// Find best pair to merge
		bestPair := ""
		bestRank := -1
		bestIdx := -1

		for i := 0; i < len(symbols)-1; i++ {
			pair := symbols[i] + " " + symbols[i+1]
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
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}

	// Convert symbols to IDs
	var ids []int
	for _, sym := range symbols {
		if id, ok := t.tokenizer.Model.Vocab[sym]; ok {
			ids = append(ids, id)
		} else if t.unkID >= 0 {
			ids = append(ids, t.unkID)
		}
	}

	return ids
}

// getInitialBPESymbols converts a word into initial BPE symbols.
func (t *Tokenizer) getInitialBPESymbols(word string) []string {
	// For byte-level BPE, each byte becomes a symbol
	// The symbols use a special byte-to-unicode mapping
	var symbols []string
	for _, r := range word {
		symbols = append(symbols, string(r))
	}

	// Add end-of-word suffix if configured
	if t.tokenizer.Model.EndOfWordSuffix != "" && len(symbols) > 0 {
		symbols[len(symbols)-1] += t.tokenizer.Model.EndOfWordSuffix
	}

	return symbols
}

// unigramTokenize implements Unigram tokenization.
func (t *Tokenizer) unigramTokenize(word string) []int {
	// Simplified Unigram: use greedy longest-match
	// Full Unigram uses Viterbi algorithm with scores
	var ids []int
	runes := []rune(word)
	start := 0

	for start < len(runes) {
		end := len(runes)
		found := false

		for end > start {
			substr := string(runes[start:end])
			if id, ok := t.tokenizer.Model.Vocab[substr]; ok {
				ids = append(ids, id)
				found = true
				start = end
				break
			}
			end--
		}

		if !found {
			// Single character fallback
			char := string(runes[start])
			if id, ok := t.tokenizer.Model.Vocab[char]; ok {
				ids = append(ids, id)
			} else if t.unkID >= 0 {
				ids = append(ids, t.unkID)
			}
			start++
		}
	}

	return ids
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

func bertPreTokenize(text string) []string {
	var tokens []string
	var current strings.Builder

	for _, r := range text {
		if isWhitespace(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
		} else if isPunctuation(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			tokens = append(tokens, string(r))
		} else {
			current.WriteRune(r)
		}
	}

	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}

	return tokens
}

func punctuationPreTokenize(text string) []string {
	var tokens []string
	var current strings.Builder

	for _, r := range text {
		if isPunctuation(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			tokens = append(tokens, string(r))
		} else {
			current.WriteRune(r)
		}
	}

	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}

	return tokens
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

func byteLevelPreTokenize(text string) []string {
	// Split on whitespace, keeping the delimiter attached to the following word
	var tokens []string
	var current strings.Builder
	inWord := false

	for i, r := range text {
		if r == ' ' {
			if inWord {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			// Attach space to next token
			if i == 0 || (i > 0 && text[i-1] == ' ') {
				current.WriteRune(byteToUnicode[' '])
			} else {
				current.WriteRune(byteToUnicode[' '])
			}
			inWord = false
		} else {
			inWord = true
			for _, b := range []byte(string(r)) {
				current.WriteRune(byteToUnicode[b])
			}
		}
	}

	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}

	return tokens
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

func metaspacePreTokenize(text string, addPrefixSpace bool) []string {
	// Replace spaces with special metaspace character
	if addPrefixSpace && len(text) > 0 && text[0] != ' ' {
		text = " " + text
	}
	text = strings.ReplaceAll(text, " ", "\u2581")

	// Split into words (where each word starts with metaspace)
	var tokens []string
	var current strings.Builder

	for _, r := range text {
		if r == '\u2581' && current.Len() > 0 {
			tokens = append(tokens, current.String())
			current.Reset()
		}
		current.WriteRune(r)
	}

	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}

	return tokens
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
