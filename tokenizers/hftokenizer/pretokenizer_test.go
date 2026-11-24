package hftokenizer

import (
	"testing"

	"github.com/gomlx/go-huggingface/tokenizers/api"
)

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

func TestSplitPreTokenizerEmptyRegexIsolatesCharacters(t *testing.T) {
	tok, err := NewFromContent(nil, []byte(`{
		"pre_tokenizer": {
			"type": "Split",
			"pattern": {"Regex": ""},
			"behavior": "Isolated"
		},
		"model": {"type": "WordPiece", "vocab": {}}
	}`))
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}

	input := "aé\nb"
	words := tok.preTokenizeWithSpans(input, []int{0, 1, 2, 3, 4})
	want := []wordWithOffset{
		{text: "a", start: 0, end: 1},
		{text: "é", start: 1, end: 3},
		{text: "\n", start: 3, end: 4},
		{text: "b", start: 4, end: 5},
	}
	if len(words) != len(want) {
		t.Fatalf("preTokenizeWithSpans() returned %d words, want %d: %+v", len(words), len(want), words)
	}
	for i := range want {
		if words[i].text != want[i].text || words[i].start != want[i].start || words[i].end != want[i].end {
			t.Errorf("preTokenizeWithSpans() word %d = %+v, want %+v", i, words[i], want[i])
		}
	}
}

func TestMetaspacePreTokenizeSpans(t *testing.T) {
	tests := []struct {
		text  string
		start int
		split bool
		want  []api.TokenSpan
	}{
		{"hello world", 0, true, []api.TokenSpan{{Start: 0, End: 5}, {Start: 5, End: 11}}},
		{"éé", 0, true, []api.TokenSpan{{Start: 0, End: 4}}},
		{"a  b", 0, true, []api.TokenSpan{{Start: 0, End: 1}, {Start: 1, End: 2}, {Start: 2, End: 4}}},
		{" x", 0, true, []api.TokenSpan{{Start: 0, End: 2}}},
		{"hello world", 0, false, []api.TokenSpan{{Start: 0, End: 11}}},
		{"hello", 5, true, []api.TokenSpan{{Start: 5, End: 10}}},
	}
	for _, tt := range tests {
		offsets := identityOffsets(len(tt.text))
		for i := range offsets {
			offsets[i] += tt.start
		}
		words := metaspacePreTokenizeWithOffsets(tt.text, offsets, true, "", "always", tt.split)
		if len(words) != len(tt.want) {
			t.Fatalf("%q: got %d words, want %d", tt.text, len(words), len(tt.want))
		}
		for i, w := range words {
			if len(w.offsets) != len(w.text) {
				t.Errorf("%q word %q: %d offsets for %d bytes", tt.text, w.text, len(w.offsets), len(w.text))
			}
			if got := w.spanForSubword(0, len(w.text)); got != tt.want[i] {
				t.Errorf("%q word %q: span %v, want %v", tt.text, w.text, got, tt.want[i])
			}
		}
	}
	w := metaspacePreTokenizeWithOffsets("hello world", identityOffsets(11), true, "", "always", true)[1]
	if got := w.spanForSubword(0, 3); got != (api.TokenSpan{Start: 5, End: 6}) {
		t.Errorf("subword ▁: span %v, want {5 6}", got)
	}
	if got := w.spanForSubword(3, len(w.text)); got != (api.TokenSpan{Start: 6, End: 11}) {
		t.Errorf("subword world: span %v, want {6 11}", got)
	}
}
