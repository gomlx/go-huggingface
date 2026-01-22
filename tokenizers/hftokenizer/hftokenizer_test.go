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

func TestBertPreTokenize(t *testing.T) {
	tests := []struct {
		input string
		want  []string
	}{
		{
			input: "Hello, world!",
			want:  []string{"Hello", ",", "world", "!"},
		},
		{
			input: "It's a test.",
			want:  []string{"It", "'", "s", "a", "test", "."},
		},
		{
			input: "simple text",
			want:  []string{"simple", "text"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := bertPreTokenize(tt.input)
			if !strSliceEqual(got, tt.want) {
				t.Errorf("bertPreTokenize(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
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

func strSliceEqual(a, b []string) bool {
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
