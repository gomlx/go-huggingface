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

// Tests for EncodeWithOffsets

func TestWordPiece_EncodeWithOffsets(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	tests := []struct {
		name        string
		input       string
		wantIDs     []int
		wantOffsets []api.TokenOffset
	}{
		{
			name:        "single word",
			input:       "hello",
			wantIDs:     []int{1},
			wantOffsets: []api.TokenOffset{{Start: 0, End: 5}},
		},
		{
			name:        "two words",
			input:       "hello world",
			wantIDs:     []int{1, 2},
			wantOffsets: []api.TokenOffset{{Start: 0, End: 5}, {Start: 6, End: 11}},
		},
		{
			name:        "word with subword",
			input:       "testing",
			wantIDs:     []int{3, 4}, // test + ##ing
			wantOffsets: []api.TokenOffset{{Start: 0, End: 4}, {Start: 4, End: 7}},
		},
		{
			name:        "sentence",
			input:       "this is a test",
			wantIDs:     []int{107, 106, 105, 3},
			wantOffsets: []api.TokenOffset{{Start: 0, End: 4}, {Start: 5, End: 7}, {Start: 8, End: 9}, {Start: 10, End: 14}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tok.EncodeWithOffsets(tt.input)
			if !intSliceEqual(result.IDs, tt.wantIDs) {
				t.Errorf("EncodeWithOffsets(%q).IDs = %v, want %v", tt.input, result.IDs, tt.wantIDs)
			}
			if !offsetsEqual(result.Offsets, tt.wantOffsets) {
				t.Errorf("EncodeWithOffsets(%q).Offsets = %v, want %v", tt.input, result.Offsets, tt.wantOffsets)
			}
			// Verify offsets point to correct text
			for i, off := range result.Offsets {
				if off.Start >= 0 && off.End <= len(tt.input) {
					substr := tt.input[off.Start:off.End]
					t.Logf("Token %d: ID=%d, offset=[%d,%d], text=%q", i, result.IDs[i], off.Start, off.End, substr)
				}
			}
		})
	}
}

func TestBPE_EncodeWithOffsets(t *testing.T) {
	tok, err := NewFromContent(nil, testSimpleBPETokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	tests := []struct {
		name        string
		input       string
		wantIDs     []int
		wantOffsets []api.TokenOffset
	}{
		{
			name:        "single word hello",
			input:       "hello",
			wantIDs:     []int{12},
			wantOffsets: []api.TokenOffset{{Start: 0, End: 5}},
		},
		{
			name:        "single word world",
			input:       "world",
			wantIDs:     []int{15},
			wantOffsets: []api.TokenOffset{{Start: 0, End: 5}},
		},
		{
			name:        "two words",
			input:       "hello world",
			wantIDs:     []int{12, 15},
			wantOffsets: []api.TokenOffset{{Start: 0, End: 5}, {Start: 6, End: 11}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tok.EncodeWithOffsets(tt.input)
			if !intSliceEqual(result.IDs, tt.wantIDs) {
				t.Errorf("EncodeWithOffsets(%q).IDs = %v, want %v", tt.input, result.IDs, tt.wantIDs)
			}
			if !offsetsEqual(result.Offsets, tt.wantOffsets) {
				t.Errorf("EncodeWithOffsets(%q).Offsets = %v, want %v", tt.input, result.Offsets, tt.wantOffsets)
			}
			// Verify offsets point to correct text
			for i, off := range result.Offsets {
				if off.Start >= 0 && off.End <= len(tt.input) {
					substr := tt.input[off.Start:off.End]
					t.Logf("Token %d: ID=%d, offset=[%d,%d], text=%q", i, result.IDs[i], off.Start, off.End, substr)
				}
			}
		})
	}
}

func TestEncodeWithOffsets_Unicode(t *testing.T) {
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

	tests := []struct {
		name        string
		input       string
		wantIDs     []int
		checkOffset bool
	}{
		{
			name:        "mixed ascii and unicode",
			input:       "hello 世界",
			wantIDs:     []int{1, 2},
			checkOffset: true,
		},
		{
			name:        "unicode only",
			input:       "日本",
			wantIDs:     []int{3},
			checkOffset: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tok.EncodeWithOffsets(tt.input)
			if !intSliceEqual(result.IDs, tt.wantIDs) {
				t.Errorf("EncodeWithOffsets(%q).IDs = %v, want %v", tt.input, result.IDs, tt.wantIDs)
			}
			if tt.checkOffset {
				// Verify offsets are valid
				if len(result.Offsets) != len(result.IDs) {
					t.Errorf("len(Offsets)=%d != len(IDs)=%d", len(result.Offsets), len(result.IDs))
				}
				for i, off := range result.Offsets {
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

func TestEncodeWithOffsets_Punctuation(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	// Test that punctuation gets its own offset
	input := "hello, world!"
	result := tok.EncodeWithOffsets(input)

	// Verify we get some offsets
	if len(result.Offsets) == 0 {
		t.Fatal("Expected some offsets")
	}

	// Check that offsets are valid and non-overlapping
	for i, off := range result.Offsets {
		if off.Start < 0 || off.End > len(input) {
			t.Errorf("Offset %d out of bounds: [%d, %d]", i, off.Start, off.End)
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

func TestEncodeWithOffsets_EmptyString(t *testing.T) {
	tok, err := NewFromContent(nil, testWordPieceTokenizerJSON)
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	result := tok.EncodeWithOffsets("")
	if len(result.IDs) != 0 {
		t.Errorf("Expected empty IDs for empty input, got %v", result.IDs)
	}
	if len(result.Offsets) != 0 {
		t.Errorf("Expected empty offsets for empty input, got %v", result.Offsets)
	}
}

func TestEncodeWithOffsets_MatchesEncode(t *testing.T) {
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
			result := tok.EncodeWithOffsets(input)
			if !intSliceEqual(ids, result.IDs) {
				t.Errorf("Encode(%q) = %v, EncodeWithOffsets(%q).IDs = %v", input, ids, input, result.IDs)
			}
		})
	}
}

func offsetsEqual(a, b []api.TokenOffset) bool {
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

func BenchmarkEncodeWithOffsets(b *testing.B) {
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
			_ = tok.EncodeWithOffsets(input)
		}
	}
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

func BenchmarkEncodeWithOffsets_LongText(b *testing.B) {
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
		_ = tok.EncodeWithOffsets(input)
	}
}
