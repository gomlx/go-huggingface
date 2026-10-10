package hftokenizer

import (
	"testing"
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
