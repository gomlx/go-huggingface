package hftokenizer

import (
	"testing"

	"github.com/gomlx/go-huggingface/tokenizers/api"
)

// Test tokenizer.json content for a WordPiece model (BERT-style)
var testWordPieceTokenizerJSON = []byte(`{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "[PAD]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 100, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 101, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 102, "content": "[SEP]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 103, "content": "[MASK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": {
    "type": "BertNormalizer",
    "lowercase": true
  },
  "pre_tokenizer": {
    "type": "BertPreTokenizer"
  },
  "post_processor": null,
  "decoder": {
    "type": "WordPiece",
    "prefix": "##"
  },
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[PAD]": 0,
      "hello": 1,
      "world": 2,
      "test": 3,
      "##ing": 4,
      "##ed": 5,
      "[UNK]": 100,
      "[CLS]": 101,
      "[SEP]": 102,
      "[MASK]": 103,
      "the": 104,
      "a": 105,
      "is": 106,
      "this": 107
    }
  }
}`)

// Test tokenizer.json content for a BPE model (GPT-2-style)
var testBPETokenizerJSON = []byte(`{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<|padding|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "ByteLevel",
    "add_prefix_space": false
  },
  "post_processor": null,
  "decoder": {
    "type": "ByteLevel"
  },
  "model": {
    "type": "BPE",
    "unk_token": null,
    "vocab": {
      "hello": 2,
      "world": 3,
      "hel": 4,
      "lo": 5,
      "wor": 6,
      "ld": 7,
      "test": 8,
      " ": 9,
      "Ġhello": 10,
      "Ġworld": 11,
      "Ġtest": 12
    },
    "merges": [
      "h e",
      "l o",
      "w o",
      "r l",
      "he l",
      "hel lo",
      "wo r",
      "wor ld"
    ]
  }
}`)

func TestNewFromContent_WordPiece(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	if tok.GetTokenizerType() != "WordPiece" {
		t.Errorf("expected type WordPiece, got %s", tok.GetTokenizerType())
	}
}

func TestNewFromContent_BPE(t *testing.T) {
	tok, err := NewFromContent(nil, testBPETokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	if tok.GetTokenizerType() != "BPE" {
		t.Errorf("expected type BPE, got %s", tok.GetTokenizerType())
	}
}

// Simple BPE tokenizer for testing merge logic (uses whitespace pre-tokenizer)
// Merges are applied in rank order (lower index = higher priority)
// "hello" merges: h+e->he, l+l->ll, he+ll->hell, hell+o->hello
// "world" merges: w+o->wo, r+l->rl, wo+rl->worl, worl+d->world
var testSimpleBPETokenizerJSON = []byte(`{
  "version": "1.0",
  "added_tokens": [
    {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Whitespace"
  },
  "decoder": {
    "type": "BPEDecoder"
  },
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {
      "<unk>": 0,
      "h": 1,
      "e": 2,
      "l": 3,
      "o": 4,
      "w": 5,
      "r": 6,
      "d": 7,
      "he": 8,
      "ll": 9,
      "rl": 10,
      "hell": 11,
      "hello": 12,
      "wo": 13,
      "worl": 14,
      "world": 15
    },
    "merges": [
      "h e",
      "l l",
      "r l",
      "he ll",
      "hell o",
      "w o",
      "wo rl",
      "worl d"
    ]
  }
}`)

// Test tokenizer.json content for a Unigram model (PEGASUS-style)
// Note: Unigram vocab is an array of [token, score] pairs where:
// - score is the log probability (float)
// - ID is the array index (not the score!)
var testUnigramTokenizerJSON = []byte(`{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Metaspace",
    "add_prefix_space": true
  },
  "post_processor": null,
  "decoder": {
    "type": "Metaspace"
  },
  "model": {
    "type": "Unigram",
    "unk_token": "<unk>",
    "vocab": [
      ["<pad>", 0.0],
      ["</s>", 0.0],
      ["<unk>", 0.0],
      ["▁hello", -5.5],
      ["▁world", -5.8],
      ["▁test", -6.2],
      ["▁", -2.1],
      ["hello", -7.5],
      ["world", -7.8],
      ["test", -8.2],
      ["ing", -4.5],
      ["▁the", -3.2]
    ]
  }
}`)

func TestNewFromContent_Unigram(t *testing.T) {
	tok, err := NewFromContent(nil, testUnigramTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	if tok.GetTokenizerType() != "Unigram" {
		t.Errorf("expected type Unigram, got %s", tok.GetTokenizerType())
	}
}

// TestUnigram_VocabParsing verifies that Unigram vocab arrays are parsed correctly.
// The second element is a score (log probability), NOT the ID.
// The ID should be the array index.
func TestUnigram_VocabParsing(t *testing.T) {
	tok, err := NewFromContent(nil, testUnigramTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	// Verify that tokens have the correct IDs (array index, not score)
	tests := []struct {
		token  string
		wantID int
	}{
		{"<pad>", 0},  // index 0, score 0.0
		{"</s>", 1},   // index 1, score 0.0
		{"<unk>", 2},  // index 2, score 0.0
		{"▁hello", 3}, // index 3, score -5.5
		{"▁world", 4}, // index 4, score -5.8
		{"▁test", 5},  // index 5, score -6.2
		{"▁", 6},      // index 6, score -2.1
		{"hello", 7},  // index 7, score -7.5
		{"world", 8},  // index 8, score -7.8
		{"test", 9},   // index 9, score -8.2
		{"ing", 10},   // index 10, score -4.5
		{"▁the", 11},  // index 11, score -3.2
	}

	for _, tt := range tests {
		t.Run(tt.token, func(t *testing.T) {
			gotID, ok := tok.TokenToID(tt.token)
			if !ok {
				t.Errorf("TokenToID(%q): token not found", tt.token)
				return
			}
			if gotID != tt.wantID {
				t.Errorf("TokenToID(%q) = %d, want %d", tt.token, gotID, tt.wantID)
			}
			// Also verify reverse lookup
			gotToken, ok := tok.IDToToken(tt.wantID)
			if !ok {
				t.Errorf("IDToToken(%d): ID not found", tt.wantID)
				return
			}
			if gotToken != tt.token {
				t.Errorf("IDToToken(%d) = %q, want %q", tt.wantID, gotToken, tt.token)
			}
		})
	}
}

func TestUnigram_Encode(t *testing.T) {
	tok, err := NewFromContent(nil, testUnigramTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	tests := []struct {
		name  string
		input string
		want  []int
	}{
		{
			name:  "single word hello",
			input: "hello",
			want:  []int{3}, // "▁hello" (metaspace adds prefix)
		},
		{
			name:  "single word world",
			input: "world",
			want:  []int{4}, // "▁world"
		},
		{
			name:  "two words",
			input: "hello world",
			want:  []int{3, 4}, // "▁hello" + "▁world"
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tok.Encode(tt.input)
			if !intSliceEqual(got, tt.want) {
				t.Errorf("Encode(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestUnigram_Decode(t *testing.T) {
	tok, err := NewFromContent(nil, testUnigramTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	tests := []struct {
		name  string
		input []int
		want  string
	}{
		{
			name:  "single token hello",
			input: []int{3}, // "▁hello"
			want:  "hello",
		},
		{
			name:  "single token world",
			input: []int{4}, // "▁world"
			want:  "world",
		},
		{
			name:  "multiple tokens",
			input: []int{3, 4}, // "▁hello" + "▁world"
			want:  "hello world",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tok.Decode(tt.input)
			if got != tt.want {
				t.Errorf("Decode(%v) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestBPE_Encode(t *testing.T) {
	tok, err := NewFromContent(nil, testSimpleBPETokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	tests := []struct {
		name  string
		input string
		want  []int
	}{
		{
			name:  "single word hello",
			input: "hello",
			want:  []int{12}, // "hello" after merges: h+e->he, l+l->ll, he+ll->hell, hell+o->hello
		},
		{
			name:  "single word world",
			input: "world",
			want:  []int{15}, // "world" after merges: w+o->wo, r+l->rl, wo+rl->worl, worl+d->world
		},
		{
			name:  "two words",
			input: "hello world",
			want:  []int{12, 15}, // both words merge fully
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tok.Encode(tt.input)
			if !intSliceEqual(got, tt.want) {
				t.Errorf("Encode(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestBPE_Decode(t *testing.T) {
	tok, err := NewFromContent(nil, testSimpleBPETokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	tests := []struct {
		name  string
		input []int
		want  string
	}{
		{
			name:  "single token hello",
			input: []int{12},
			want:  "hello",
		},
		{
			name:  "single token world",
			input: []int{15},
			want:  "world",
		},
		{
			name:  "multiple tokens",
			input: []int{12, 15},
			want:  "helloworld", // BPE decoder joins without spaces
		},
		{
			name:  "subword tokens",
			input: []int{8, 9, 4}, // "he" + "ll" + "o"
			want:  "hello",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tok.Decode(tt.input)
			if got != tt.want {
				t.Errorf("Decode(%v) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestBPE_PartialMerge(t *testing.T) {
	// Test that partial merges work correctly
	tok, err := NewFromContent(nil, testSimpleBPETokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	// "helloworld" without space - should be two separate words from pre-tokenizer
	// but as one word, BPE should handle it
	ids := tok.Encode("helloworld")
	decoded := tok.Decode(ids)

	// Encode then decode should give us back the original
	if decoded != "helloworld" {
		t.Errorf("round-trip failed: got %q, want %q", decoded, "helloworld")
	}
}

// Test tokenizer.json with BPE merges in array format (like embeddinggemma)
var testArrayMergesBPETokenizerJSON = []byte(`{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "BPE",
    "vocab": {
      "h": 0, "e": 1, "l": 2, "o": 3, "w": 4, "r": 5, "d": 6,
      "he": 7, "ll": 8, "rl": 9, "hell": 10, "hello": 11,
      "wo": 12, "worl": 13, "world": 14
    },
    "merges": [
      ["h", "e"],
      ["l", "l"],
      ["r", "l"],
      ["he", "ll"],
      ["hell", "o"],
      ["w", "o"],
      ["wo", "rl"],
      ["worl", "d"]
    ]
  }
}`)

func TestBPE_ArrayFormatMerges(t *testing.T) {
	// Test that array-format merges (like embeddinggemma uses) are parsed correctly
	tok, err := NewFromContent(nil, testArrayMergesBPETokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	if tok.GetTokenizerType() != "BPE" {
		t.Errorf("expected type BPE, got %s", tok.GetTokenizerType())
	}

	// Test encoding works
	tests := []struct {
		name  string
		input string
		want  []int
	}{
		{
			name:  "hello",
			input: "hello",
			want:  []int{11}, // fully merged
		},
		{
			name:  "world",
			input: "world",
			want:  []int{14}, // fully merged
		},
		{
			name:  "hello world",
			input: "hello world",
			want:  []int{11, 14}, // two separate tokens
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tok.Encode(tt.input)
			if len(got) != len(tt.want) {
				t.Errorf("Encode(%q) got %v, want %v", tt.input, got, tt.want)
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("Encode(%q) got %v, want %v", tt.input, got, tt.want)
					return
				}
			}
		})
	}

	// Test decode works
	decoded := tok.Decode([]int{11, 14})
	if decoded != "hello world" {
		t.Errorf("Decode([11, 14]) = %q, want %q", decoded, "hello world")
	}
}

func TestWordPiece_Encode(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	tests := []struct {
		name  string
		input string
		want  []int
	}{
		{
			name:  "single word in vocab",
			input: "hello",
			want:  []int{1},
		},
		{
			name:  "multiple words",
			input: "hello world",
			want:  []int{1, 2},
		},
		{
			name:  "word with subword",
			input: "testing",
			want:  []int{3, 4}, // test + ##ing
		},
		{
			name:  "the",
			input: "the",
			want:  []int{104},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tok.Encode(tt.input)
			if !intSliceEqual(got, tt.want) {
				t.Errorf("Encode(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestWordPiece_Decode(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	tests := []struct {
		name  string
		input []int
		want  string
	}{
		{
			name:  "single word",
			input: []int{1},
			want:  "hello",
		},
		{
			name:  "multiple words",
			input: []int{1, 2},
			want:  "hello world",
		},
		{
			name:  "word with subword",
			input: []int{3, 4},
			want:  "testing",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tok.Decode(tt.input)
			if got != tt.want {
				t.Errorf("Decode(%v) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestWordPiece_SpecialTokenID(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	tests := []struct {
		name    string
		token   api.SpecialToken
		want    int
		wantErr bool
	}{
		{
			name:    "unknown token",
			token:   api.TokUnknown,
			want:    100,
			wantErr: false,
		},
		{
			name:    "pad token",
			token:   api.TokPad,
			want:    0,
			wantErr: false,
		},
		{
			name:    "mask token",
			token:   api.TokMask,
			want:    103,
			wantErr: false,
		},
		{
			name:    "cls/bos token",
			token:   api.TokBeginningOfSentence,
			want:    101, // Falls back to CLS
			wantErr: false,
		},
		{
			name:    "sep/eos token",
			token:   api.TokEndOfSentence,
			want:    102, // Falls back to SEP
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tok.SpecialTokenID(tt.token)
			if (err != nil) != tt.wantErr {
				t.Errorf("SpecialTokenID(%v) error = %v, wantErr %v", tt.token, err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("SpecialTokenID(%v) = %v, want %v", tt.token, got, tt.want)
			}
		})
	}
}

func TestWordPiece_VocabSize(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	// Vocab has 13 entries, plus 5 added tokens
	// But some added tokens overlap with vocab, so we get unique count
	size := tok.VocabSize()
	if size < 13 {
		t.Errorf("VocabSize() = %d, want >= 13", size)
	}
}

func TestTokenToID_IDToToken(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	// Test TokenToID
	id, ok := tok.TokenToID("hello")
	if !ok {
		t.Error("TokenToID(hello) not found")
	}
	if id != 1 {
		t.Errorf("TokenToID(hello) = %d, want 1", id)
	}

	// Test IDToToken
	token, ok := tok.IDToToken(1)
	if !ok {
		t.Error("IDToToken(1) not found")
	}
	if token != "hello" {
		t.Errorf("IDToToken(1) = %q, want hello", token)
	}

	// Test added token
	id, ok = tok.TokenToID("[CLS]")
	if !ok {
		t.Error("TokenToID([CLS]) not found")
	}
	if id != 101 {
		t.Errorf("TokenToID([CLS]) = %d, want 101", id)
	}
}

func TestGetVocab(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	vocab := tok.GetVocab()

	// Check some expected entries
	if vocab["hello"] != 1 {
		t.Errorf("vocab[hello] = %d, want 1", vocab["hello"])
	}
	if vocab["[CLS]"] != 101 {
		t.Errorf("vocab[[CLS]] = %d, want 101", vocab["[CLS]"])
	}
}

func TestAddedTokensList(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	added := tok.AddedTokensList()
	if len(added) != 5 {
		t.Errorf("AddedTokensList() length = %d, want 5", len(added))
	}

	// Should be sorted by ID
	for i := 1; i < len(added); i++ {
		if added[i-1].ID > added[i].ID {
			t.Error("AddedTokensList() not sorted by ID")
			break
		}
	}
}

func TestCleanText(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"hello world", "hello world"},
		{"hello\tworld", "hello world"},
		{"hello\nworld", "hello world"},
		{"hello\x00world", "helloworld"}, // null char removed
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := cleanText(tt.input)
			if got != tt.want {
				t.Errorf("cleanText(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

func TestIsPunctuation(t *testing.T) {
	tests := []struct {
		r    rune
		want bool
	}{
		{'.', true},
		{',', true},
		{'!', true},
		{'?', true},
		{';', true},
		{':', true},
		{'"', true},
		{'\'', true},
		{'a', false},
		{'1', false},
		{' ', false},
	}

	for _, tt := range tests {
		t.Run(string(tt.r), func(t *testing.T) {
			got := isPunctuation(tt.r)
			if got != tt.want {
				t.Errorf("isPunctuation(%q) = %v, want %v", tt.r, got, tt.want)
			}
		})
	}
}

func TestInvalidJSON(t *testing.T) {
	_, err := NewFromContent(nil, []byte("not valid json"))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestEmptyVocab(t *testing.T) {
	emptyVocabJSON := []byte(`{
		"model": {
			"type": "WordPiece",
			"vocab": {},
			"unk_token": "[UNK]"
		}
	}`)

	tok, err := NewFromContent(nil, emptyVocabJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	// Encoding unknown text should return empty (no unk token defined)
	ids := tok.Encode("hello")
	if len(ids) != 0 {
		t.Errorf("Encode() with empty vocab = %v, want empty", ids)
	}
}

func TestUnicodeNormalization(t *testing.T) {
	// Test tokenizer with NFD normalizer
	nfdTokenizerJSON := []byte(`{
		"normalizer": {"type": "NFD"},
		"pre_tokenizer": {"type": "Whitespace"},
		"model": {
			"type": "WordPiece",
			"vocab": {"cafe": 1, "e": 2, "\u0301": 3},
			"unk_token": ""
		}
	}`)

	tok, err := NewFromContent(nil, nfdTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	// "café" in NFC form (single é character) should be normalized to NFD (e + combining accent)
	// The combining acute accent is U+0301
	cafeNFC := "caf\u00e9"  // café with precomposed é
	cafeNFD := "cafe\u0301" // café with e + combining acute accent

	// After NFD normalization, both should produce the same result
	ids1 := tok.Encode(cafeNFC)
	ids2 := tok.Encode(cafeNFD)

	if !intSliceEqual(ids1, ids2) {
		t.Errorf("NFD normalization failed: Encode(%q) = %v, Encode(%q) = %v", cafeNFC, ids1, cafeNFD, ids2)
	}
}

func TestNFKCNormalization(t *testing.T) {
	// Test NFKC normalization (used by some models)
	nfkcTokenizerJSON := []byte(`{
		"normalizer": {"type": "NFKC"},
		"pre_tokenizer": {"type": "Whitespace"},
		"model": {
			"type": "WordPiece",
			"vocab": {"fi": 1},
			"unk_token": ""
		}
	}`)

	tok, err := NewFromContent(nil, nfkcTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	// The fi ligature (U+FB01) should be normalized to "fi" by NFKC
	fiLigature := "\ufb01" // ﬁ ligature

	ids := tok.Encode(fiLigature)
	// Should find "fi" in vocab after NFKC normalization
	if len(ids) != 1 || ids[0] != 1 {
		t.Errorf("NFKC normalization failed: Encode(%q) = %v, want [1]", fiLigature, ids)
	}
}

// Tests for EncodeWithAnnotations

func TestWordPiece_EncodeWithAnnotations(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}
	tok.options.IncludeSpans = true

	tests := []struct {
		name      string
		input     string
		wantIDs   []int
		wantSpans []api.TokenSpan
	}{
		{
			name:      "single word",
			input:     "hello",
			wantIDs:   []int{1},
			wantSpans: []api.TokenSpan{{Start: 0, End: 5}},
		},
		{
			name:      "two words",
			input:     "hello world",
			wantIDs:   []int{1, 2},
			wantSpans: []api.TokenSpan{{Start: 0, End: 5}, {Start: 6, End: 11}},
		},
		{
			name:      "word with subword",
			input:     "testing",
			wantIDs:   []int{3, 4}, // test + ##ing
			wantSpans: []api.TokenSpan{{Start: 0, End: 4}, {Start: 4, End: 7}},
		},
		{
			name:      "sentence",
			input:     "this is a test",
			wantIDs:   []int{107, 106, 105, 3},
			wantSpans: []api.TokenSpan{{Start: 0, End: 4}, {Start: 5, End: 7}, {Start: 8, End: 9}, {Start: 10, End: 14}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tok.EncodeWithAnnotations(tt.input)
			if !intSliceEqual(result.IDs, tt.wantIDs) {
				t.Errorf("EncodeWithSpans(%q).IDs = %v, want %v", tt.input, result.IDs, tt.wantIDs)
			}
			if !spansEqual(result.Spans, tt.wantSpans) {
				t.Errorf("EncodeWithSpans(%q).Spans = %v, want %v", tt.input, result.Spans, tt.wantSpans)
			}
			// Verify offsets point to correct text
			for i, off := range result.Spans {
				if off.Start >= 0 && off.End <= len(tt.input) {
					substr := tt.input[off.Start:off.End]
					t.Logf("Token %d: ID=%d, offset=[%d,%d], text=%q", i, result.IDs[i], off.Start, off.End, substr)
				}
			}
		})
	}
}

func TestBPE_EncodeWithAnnotations(t *testing.T) {
	tok, err := NewFromContent(nil, testSimpleBPETokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}
	tok.options.IncludeSpans = true

	tests := []struct {
		name      string
		input     string
		wantIDs   []int
		wantSpans []api.TokenSpan
	}{
		{
			name:      "single word hello",
			input:     "hello",
			wantIDs:   []int{12},
			wantSpans: []api.TokenSpan{{Start: 0, End: 5}},
		},
		{
			name:      "single word world",
			input:     "world",
			wantIDs:   []int{15},
			wantSpans: []api.TokenSpan{{Start: 0, End: 5}},
		},
		{
			name:      "two words",
			input:     "hello world",
			wantIDs:   []int{12, 15},
			wantSpans: []api.TokenSpan{{Start: 0, End: 5}, {Start: 6, End: 11}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tok.EncodeWithAnnotations(tt.input)
			if !intSliceEqual(result.IDs, tt.wantIDs) {
				t.Errorf("EncodeWithSpans(%q).IDs = %v, want %v", tt.input, result.IDs, tt.wantIDs)
			}
			if !spansEqual(result.Spans, tt.wantSpans) {
				t.Errorf("EncodeWithSpans(%q).Spans = %v, want %v", tt.input, result.Spans, tt.wantSpans)
			}
			// Verify offsets point to correct text
			for i, off := range result.Spans {
				if off.Start >= 0 && off.End <= len(tt.input) {
					substr := tt.input[off.Start:off.End]
					t.Logf("Token %d: ID=%d, offset=[%d,%d], text=%q", i, result.IDs[i], off.Start, off.End, substr)
				}
			}
		})
	}
}

func TestEncodeWithAnnotations_Unicode(t *testing.T) {
	// Test with a simple tokenizer that handles unicode
	unicodeTokenizerJSON := []byte(`{
		"normalizer": null,
		"pre_tokenizer": {"type": "Whitespace"},
		"model": {
			"type": "WordPiece",
			"vocab": {
				"hello": 1,
				"世界": 2,
				"日本": 3,
				"café": 4,
				"test": 5
			},
			"unk_token": "",
			"continuing_subword_prefix": "##"
		}
	}`)

	tok, err := NewFromContent(nil, unicodeTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}
	tok.options.IncludeSpans = true

	tests := []struct {
		name      string
		input     string
		wantIDs   []int
		checkSpan bool
	}{
		{
			name:      "mixed ascii and unicode",
			input:     "hello 世界",
			wantIDs:   []int{1, 2},
			checkSpan: true,
		},
		{
			name:      "unicode only",
			input:     "日本",
			wantIDs:   []int{3},
			checkSpan: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tok.EncodeWithAnnotations(tt.input)
			if !intSliceEqual(result.IDs, tt.wantIDs) {
				t.Errorf("EncodeWithSpans(%q).IDs = %v, want %v", tt.input, result.IDs, tt.wantIDs)
			}
			if tt.checkSpan {
				// Verify offsets are valid
				if len(result.Spans) != len(result.IDs) {
					t.Errorf("len(Spans)=%d != len(IDs)=%d", len(result.Spans), len(result.IDs))
				}
				for i, off := range result.Spans {
					if off.Start < 0 || off.End > len(tt.input) || off.Start > off.End {
						t.Errorf("Invalid offset at %d: [%d, %d] for input length %d", i, off.Start, off.End, len(tt.input))
					} else {
						substr := tt.input[off.Start:off.End]
						t.Logf("Token %d: ID=%d, offset=[%d,%d], text=%q", i, result.IDs[i], off.Start, off.End, substr)
					}
				}
			}
		})
	}
}

func TestEncodeWithAnnotations_Punctuation(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}
	tok.options.IncludeSpans = true

	// Test that punctuation gets its own offset
	input := "hello, world!"
	result := tok.EncodeWithAnnotations(input)

	// Verify we get some offsets
	if len(result.Spans) == 0 {
		t.Fatal("Expected some offsets")
	}

	// Check that offsets are valid and non-overlapping
	for i, off := range result.Spans {
		if off.Start < 0 || off.End > len(input) {
			t.Errorf("Span %d out of bounds: [%d, %d]", i, off.Start, off.End)
		}
		if off.Start > off.End {
			t.Errorf("Invalid offset %d: start > end: [%d, %d]", i, off.Start, off.End)
		}
		// Log for debugging
		if off.Start < off.End {
			t.Logf("Token %d: ID=%d, offset=[%d,%d], text=%q", i, result.IDs[i], off.Start, off.End, input[off.Start:off.End])
		}
	}
}

func TestEncodeWithAnnotations_EmptyString(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}
	tok.options.IncludeSpans = true

	result := tok.EncodeWithAnnotations("")
	if len(result.IDs) != 0 {
		t.Errorf("Expected empty IDs for empty input, got %v", result.IDs)
	}
	if len(result.Spans) != 0 {
		t.Errorf("Expected empty offsets for empty input, got %v", result.Spans)
	}
}

func TestEncodeWithAnnotations_MatchesEncode(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	inputs := []string{
		"hello",
		"hello world",
		"testing",
		"this is a test",
		"hello, world!",
	}

	for _, input := range inputs {
		t.Run(input, func(t *testing.T) {
			ids := tok.Encode(input)
			result := tok.EncodeWithAnnotations(input)
			if !intSliceEqual(ids, result.IDs) {
				t.Errorf("Encode(%q) = %v, EncodeWithAnnotations(%q).IDs = %v", input, ids, input, result.IDs)
			}
		})
	}
}

func spansEqual(a, b []api.TokenSpan) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].Start != b[i].Start || a[i].End != b[i].End {
			return false
		}
	}
	return true
}

// Helper functions

func intSliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Test that special/added tokens in the input are recognized as single tokens
// rather than being split by pre-tokenization (e.g., <bos> → "<", "bos", ">").
func TestEncodeSpecialTokens(t *testing.T) {
	// Tokenizer with special tokens similar to Gemma/LLaMA style
	specialTokenizerJSON := []byte(`{
		"version": "1.0",
		"added_tokens": [
			{"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
			{"id": 1, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
			{"id": 2, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
			{"id": 3, "content": "<start_of_turn>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
			{"id": 4, "content": "<end_of_turn>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
		],
		"normalizer": null,
		"pre_tokenizer": {"type": "Whitespace"},
		"decoder": null,
		"model": {
			"type": "WordPiece",
			"unk_token": "",
			"continuing_subword_prefix": "##",
			"vocab": {
				"<pad>": 0,
				"<bos>": 1,
				"<eos>": 2,
				"<start_of_turn>": 3,
				"<end_of_turn>": 4,
				"hello": 10,
				"world": 11,
				"user": 12,
				"model": 13
			}
		}
	}`)

	tok, err := NewFromContent(nil, specialTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	tests := []struct {
		name    string
		input   string
		wantIDs []int
	}{
		{
			name:    "bos token alone",
			input:   "<bos>",
			wantIDs: []int{1},
		},
		{
			name:    "bos then text",
			input:   "<bos>hello",
			wantIDs: []int{1, 10},
		},
		{
			name:    "chat template style",
			input:   "<bos><start_of_turn>user\nhello world<end_of_turn>\n<start_of_turn>model\n",
			wantIDs: []int{1, 3, 12, 10, 11, 4, 3, 13},
		},
		{
			name:    "multiple special tokens",
			input:   "<bos><eos>",
			wantIDs: []int{1, 2},
		},
		{
			name:    "special token with surrounding text",
			input:   "hello<eos>world",
			wantIDs: []int{10, 2, 11},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tok.Encode(tt.input)
			if !intSliceEqual(got, tt.wantIDs) {
				t.Errorf("Encode(%q) = %v, want %v", tt.input, got, tt.wantIDs)
			}
		})
	}
}

// Test that multi-byte Unicode characters adjacent to special tokens don't
// cause mid-rune matching issues.
func TestEncodeSpecialTokens_Unicode(t *testing.T) {
	specialTokenizerJSON := []byte(`{
		"version": "1.0",
		"added_tokens": [
			{"id": 1, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
		],
		"normalizer": null,
		"pre_tokenizer": {"type": "Whitespace"},
		"decoder": null,
		"model": {
			"type": "WordPiece",
			"unk_token": "",
			"continuing_subword_prefix": "##",
			"vocab": {
				"<eos>": 1,
				"日本": 10,
				"世界": 11
			}
		}
	}`)

	tok, err := NewFromContent(nil, specialTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	tests := []struct {
		name    string
		input   string
		wantIDs []int
	}{
		{
			name:    "unicode before special token",
			input:   "日本<eos>世界",
			wantIDs: []int{10, 1, 11},
		},
		{
			name:    "special token between unicode",
			input:   "世界<eos>日本",
			wantIDs: []int{11, 1, 10},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tok.Encode(tt.input)
			if !intSliceEqual(got, tt.wantIDs) {
				t.Errorf("Encode(%q) = %v, want %v", tt.input, got, tt.wantIDs)
			}
		})
	}
}

// Test that TemplateProcessing post-processor adds [CLS] and [SEP] tokens.
func TestPostProcessor_TemplateProcessing(t *testing.T) {
	// WordPiece tokenizer with BERT-style TemplateProcessing post-processor
	bertTokenizerJSON := []byte(`{
		"version": "1.0",
		"added_tokens": [
			{"id": 0, "content": "[PAD]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
			{"id": 100, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
			{"id": 101, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
			{"id": 102, "content": "[SEP]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
		],
		"normalizer": {"type": "BertNormalizer", "lowercase": true},
		"pre_tokenizer": {"type": "BertPreTokenizer"},
		"post_processor": {
			"type": "TemplateProcessing",
			"single": [
				{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
				{"Sequence": {"id": "A", "type_id": 0}},
				{"SpecialToken": {"id": "[SEP]", "type_id": 0}}
			],
			"pair": [
				{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
				{"Sequence": {"id": "A", "type_id": 0}},
				{"SpecialToken": {"id": "[SEP]", "type_id": 0}},
				{"Sequence": {"id": "B", "type_id": 1}},
				{"SpecialToken": {"id": "[SEP]", "type_id": 1}}
			],
			"special_tokens": {
				"[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
				"[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
			}
		},
		"decoder": {"type": "WordPiece", "prefix": "##"},
		"model": {
			"type": "WordPiece",
			"unk_token": "[UNK]",
			"continuing_subword_prefix": "##",
			"max_input_chars_per_word": 100,
			"vocab": {
				"[PAD]": 0, "hello": 1, "world": 2, "test": 3,
				"[UNK]": 100, "[CLS]": 101, "[SEP]": 102
			}
		}
	}`)

	tok, err := NewFromContent(nil, bertTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	tests := []struct {
		name    string
		input   string
		wantIDs []int
	}{
		{
			name:    "single word gets CLS and SEP",
			input:   "hello",
			wantIDs: []int{101, 1, 102}, // [CLS] hello [SEP]
		},
		{
			name:    "two words get CLS and SEP",
			input:   "hello world",
			wantIDs: []int{101, 1, 2, 102}, // [CLS] hello world [SEP]
		},
		{
			name:    "empty input gets only CLS and SEP",
			input:   "",
			wantIDs: []int{101, 102}, // [CLS] [SEP]
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tok.Encode(tt.input)
			if !intSliceEqual(got, tt.wantIDs) {
				t.Errorf("Encode(%q) = %v, want %v", tt.input, got, tt.wantIDs)
			}
		})
	}
}

// Test that post-processor span tracking works correctly.
func TestPostProcessor_Spans(t *testing.T) {
	bertTokenizerJSON := []byte(`{
		"version": "1.0",
		"added_tokens": [
			{"id": 101, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
			{"id": 102, "content": "[SEP]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
		],
		"normalizer": {"type": "BertNormalizer", "lowercase": true},
		"pre_tokenizer": {"type": "BertPreTokenizer"},
		"post_processor": {
			"type": "TemplateProcessing",
			"single": [
				{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
				{"Sequence": {"id": "A", "type_id": 0}},
				{"SpecialToken": {"id": "[SEP]", "type_id": 0}}
			],
			"special_tokens": {
				"[CLS]": {"id": "[CLS]", "ids": [101], "tokens": ["[CLS]"]},
				"[SEP]": {"id": "[SEP]", "ids": [102], "tokens": ["[SEP]"]}
			}
		},
		"decoder": {"type": "WordPiece", "prefix": "##"},
		"model": {
			"type": "WordPiece",
			"unk_token": "",
			"continuing_subword_prefix": "##",
			"vocab": {"hello": 1, "world": 2, "[CLS]": 101, "[SEP]": 102}
		}
	}`)

	tok, err := NewFromContent(nil, bertTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}
	tok.options.IncludeSpans = true

	result := tok.EncodeWithAnnotations("hello world")

	// Expect: [CLS] hello world [SEP]
	wantIDs := []int{101, 1, 2, 102}
	if !intSliceEqual(result.IDs, wantIDs) {
		t.Fatalf("IDs = %v, want %v", result.IDs, wantIDs)
	}

	// [CLS] and [SEP] spans should be {-1, -1} (synthetic tokens)
	if result.Spans[0].Start != -1 || result.Spans[0].End != -1 {
		t.Errorf("[CLS] span = %v, want {-1, -1}", result.Spans[0])
	}
	if result.Spans[3].Start != -1 || result.Spans[3].End != -1 {
		t.Errorf("[SEP] span = %v, want {-1, -1}", result.Spans[3])
	}

	// "hello" and "world" should have valid spans
	if result.Spans[1].Start != 0 || result.Spans[1].End != 5 {
		t.Errorf("hello span = %v, want {0, 5}", result.Spans[1])
	}
	if result.Spans[2].Start != 6 || result.Spans[2].End != 11 {
		t.Errorf("world span = %v, want {6, 11}", result.Spans[2])
	}
}

// Test that null post_processor doesn't add anything (existing behavior preserved).
func TestPostProcessor_Null(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	// With null post_processor, no [CLS]/[SEP] should be added
	got := tok.Encode("hello world")
	want := []int{1, 2}
	if !intSliceEqual(got, want) {
		t.Errorf("Encode with null post_processor = %v, want %v", got, want)
	}
}

// Test BertProcessing post-processor (used by bert-base-uncased, etc.)
func TestPostProcessor_BertProcessing(t *testing.T) {
	bertTokenizerJSON := []byte(`{
		"version": "1.0",
		"added_tokens": [
			{"id": 101, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
			{"id": 102, "content": "[SEP]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
		],
		"normalizer": {"type": "BertNormalizer", "lowercase": true},
		"pre_tokenizer": {"type": "BertPreTokenizer"},
		"post_processor": {
			"type": "BertProcessing",
			"sep": ["[SEP]", 102],
			"cls": ["[CLS]", 101]
		},
		"decoder": {"type": "WordPiece", "prefix": "##"},
		"model": {
			"type": "WordPiece",
			"unk_token": "",
			"continuing_subword_prefix": "##",
			"vocab": {"hello": 1, "world": 2, "[CLS]": 101, "[SEP]": 102}
		}
	}`)

	tok, err := NewFromContent(nil, bertTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	got := tok.Encode("hello world")
	want := []int{101, 1, 2, 102} // [CLS] hello world [SEP]
	if !intSliceEqual(got, want) {
		t.Errorf("Encode = %v, want %v", got, want)
	}
}

// Test RobertaProcessing post-processor.
func TestPostProcessor_RobertaProcessing(t *testing.T) {
	robertaTokenizerJSON := []byte(`{
		"version": "1.0",
		"added_tokens": [
			{"id": 0, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
			{"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
		],
		"normalizer": null,
		"pre_tokenizer": {"type": "Whitespace"},
		"post_processor": {
			"type": "RobertaProcessing",
			"sep": ["</s>", 2],
			"cls": ["<s>", 0]
		},
		"decoder": null,
		"model": {
			"type": "WordPiece",
			"unk_token": "",
			"continuing_subword_prefix": "##",
			"vocab": {"hello": 1, "world": 3, "<s>": 0, "</s>": 2}
		}
	}`)

	tok, err := NewFromContent(nil, robertaTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	got := tok.Encode("hello world")
	want := []int{0, 1, 3, 2} // <s> hello world </s>
	if !intSliceEqual(got, want) {
		t.Errorf("Encode = %v, want %v", got, want)
	}
}

// Test EncodeWithOptions(text, false) skips post-processing.
func TestEncodeWithOptions(t *testing.T) {
	bertTokenizerJSON := []byte(`{
		"version": "1.0",
		"added_tokens": [
			{"id": 101, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
			{"id": 102, "content": "[SEP]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
		],
		"normalizer": {"type": "BertNormalizer", "lowercase": true},
		"pre_tokenizer": {"type": "BertPreTokenizer"},
		"post_processor": {
			"type": "BertProcessing",
			"sep": ["[SEP]", 102],
			"cls": ["[CLS]", 101]
		},
		"decoder": {"type": "WordPiece", "prefix": "##"},
		"model": {
			"type": "WordPiece",
			"unk_token": "",
			"continuing_subword_prefix": "##",
			"vocab": {"hello": 1, "world": 2, "[CLS]": 101, "[SEP]": 102}
		}
	}`)

	tok, err := NewFromContent(nil, bertTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	// Encode includes [CLS]/[SEP]
	full := tok.Encode("hello world")
	wantFull := []int{101, 1, 2, 102}
	if !intSliceEqual(full, wantFull) {
		t.Errorf("Encode = %v, want %v", full, wantFull)
	}

	// Encode AddSpecialTokens: false skips post-processing
	err = tok.With(api.EncodeOptions{AddSpecialTokens: false})
	if err != nil {
		t.Fatalf("With failed: %v", err)
	}

	raw := tok.Encode("hello world")
	wantRaw := []int{1, 2}
	if !intSliceEqual(raw, wantRaw) {
		t.Errorf("Encode = %v, want %v", raw, wantRaw)
	}
}

// Benchmarks for offset tracking overhead

func BenchmarkEncode(b *testing.B) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		b.Fatalf("NewFromContent failed: %v", err)
	}

	inputs := []string{
		"hello world",
		"this is a test",
		"testing tokenization",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, input := range inputs {
			_ = tok.Encode(input)
		}
	}
}

func BenchmarkEncodeWithAnnotations(b *testing.B) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		b.Fatalf("NewFromContent failed: %v", err)
	}

	inputs := []string{
		"hello world",
		"this is a test",
		"testing tokenization",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, input := range inputs {
			_ = tok.EncodeWithAnnotations(input)
		}
	}
}

func TestEncodeWithAnnotations_AllOutputs(t *testing.T) {
	bertTokenizerJSON := []byte(`{
		"version": "1.0",
		"added_tokens": [
			{"id": 101, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
			{"id": 102, "content": "[SEP]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
		],
		"normalizer": {"type": "BertNormalizer", "lowercase": true},
		"pre_tokenizer": {"type": "BertPreTokenizer"},
		"post_processor": {
			"type": "BertProcessing",
			"sep": ["[SEP]", 102],
			"cls": ["[CLS]", 101]
		},
		"model": {
			"type": "WordPiece",
			"vocab": {"hello": 1, "world": 2, "[CLS]": 101, "[SEP]": 102}
		}
	}`)

	tok, err := NewFromContent(nil, bertTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	options := api.EncodeOptions{
		AddSpecialTokens:         true,
		IncludeSpans:             true,
		IncludeSpecialTokensMask: true,
	}
	err = tok.With(options)
	if err != nil {
		t.Fatalf("With failed: %v", err)
	}

	result := tok.EncodeWithAnnotations("hello world")

	// IDs: [CLS] hello world [SEP]
	wantIDs := []int{101, 1, 2, 102}
	if !intSliceEqual(result.IDs, wantIDs) {
		t.Errorf("IDs = %v, want %v", result.IDs, wantIDs)
	}

	// SpecialTokensMask: 1 for [CLS] and [SEP], 0 for others
	wantSpecialMask := []int{1, 0, 0, 1}
	if len(result.SpecialTokensMask) != len(wantIDs) {
		t.Errorf("len(SpecialTokensMask) = %d, want %d", len(result.SpecialTokensMask), len(wantIDs))
	} else {
		for i, mask := range result.SpecialTokensMask {
			if mask != wantSpecialMask[i] {
				t.Errorf("SpecialTokensMask[%d] = %d, want %d", i, mask, wantSpecialMask[i])
			}
		}
	}
}

func stringSliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func BenchmarkEncode_LongText(b *testing.B) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		b.Fatalf("NewFromContent failed: %v", err)
	}

	// Generate a longer input
	input := "this is a test hello world testing "
	for len(input) < 1000 {
		input += input
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.Encode(input)
	}
}

func BenchmarkEncodeWithAnnotations_LongText(b *testing.B) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		b.Fatalf("NewFromContent failed: %v", err)
	}

	// Generate a longer input
	input := "this is a test hello world testing "
	for len(input) < 1000 {
		input += input
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tok.EncodeWithAnnotations(input)
	}
}
