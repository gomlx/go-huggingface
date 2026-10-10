package hftokenizer

import (
	"testing"
)

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
