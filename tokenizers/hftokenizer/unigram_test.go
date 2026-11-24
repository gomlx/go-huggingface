package hftokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"
	"testing"

	"github.com/gomlx/go-huggingface/hub"
)

func TestUnigram_ViterbiUnknownAndByteFallback(t *testing.T) {
	for _, byteFallback := range []bool{false, true} {
		tok, err := NewFromContent(nil, fmt.Appendf(nil, `{"version": "1.0",
  "added_tokens": [{"id": 0, "content": "<unk>", "special": true}],
  "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
  "model": {"type": "Unigram", "unk_id": 0, "byte_fallback": %t, "vocab": [["<unk>", 0.0], ["▁", -2.0], ["▁ab", -10.0],
    ["▁a", -1.0], ["b", -1.0], ["c", -3.0], ["<0xC3>", -5.0], ["<0xA9>", -5.0]]}}`, byteFallback))
		if err != nil {
			t.Fatal(err)
		}
		tok.options.IncludeSpans = true
		// Expected ids come from HuggingFace tokenizers on the same tokenizer.json.
		want := map[string][]int{"ab": {3, 4}, "abc": {3, 4, 5}, "axyb": {3, 0, 4}, "é": {1, 0}, "éé": {1, 0}}
		if byteFallback {
			want["é"], want["éé"] = []int{1, 6, 7}, []int{1, 6, 7, 6, 7}
		}
		for text, ids := range want {
			if got := tok.Encode(text); !slices.Equal(got, ids) {
				t.Errorf("byte_fallback=%t Encode(%q) = %v, want %v", byteFallback, text, got, ids)
			}
		}
		spans := tok.EncodeWithAnnotations("éé").Spans
		for _, span := range spans[1:] {
			if span != spans[1] || span.End-span.Start != len("éé") {
				t.Errorf("byte_fallback=%t spans(%q) = %v, want one shared span over both runes", byteFallback, "éé", spans)
			}
		}
	}
}

func TestUnigram_ModelUnkID(t *testing.T) {
	for _, added := range []string{``, `{"id": 0, "content": "<unk>", "special": true}`} {
		tok, err := NewFromContent(nil, fmt.Appendf(nil, `{"version": "1.0", "added_tokens": [%s],
  "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
  "model": {"type": "Unigram", "unk_id": 3, "vocab": [["<unk>", 0.0], ["▁", -2.0], ["a", -1.0], ["<UNK>", 0.0]]}}`, added))
		if err != nil {
			t.Fatal(err)
		}
		if got := tok.Encode("axa"); !slices.Equal(got, []int{1, 2, 3, 2}) {
			t.Errorf("added_tokens=%q Encode(\"axa\") = %v, want [1 2 3 2]", added, got)
		}
	}
}

func TestNewFromGGUF_Unigram(t *testing.T) {
	params := GGUFTokenizerParams{
		Model:          "llama",
		Tokens:         []string{"<unk>", "▁", "▁ab", "▁a", "b"},
		UnknownTokenID: 0, BosTokenID: -1, EosTokenID: -1, SepTokenID: -1, PadTokenID: -1, ClsTokenID: -1, MaskTokenID: -1,
	}
	for _, c := range []struct {
		scores []float32
		want   []int
	}{{[]float32{0, -2, -10, -1, -1}, []int{3, 4}}, {nil, []int{2}}} {
		params.Scores = c.scores
		tok, err := NewFromGGUF(nil, params)
		if err != nil {
			t.Fatal(err)
		}
		if got := tok.Encode("ab"); !slices.Equal(got, c.want) {
			t.Errorf("scores=%v Encode(\"ab\") = %v, want %v", c.scores, got, c.want)
		}
	}
}

func TestUnigram_HuggingFaceParity(t *testing.T) {
	if testing.Short() {
		t.Skip("downloads tokenizer.json")
	}
	data, err := os.ReadFile("testdata/unigram_parity.json")
	if err != nil {
		t.Fatal(err)
	}
	var parity struct {
		Repo  string `json:"repo"`
		Cases []struct {
			Text string `json:"text"`
			IDs  []int  `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &parity); err != nil {
		t.Fatal(err)
	}
	tok, err := New(nil, hub.New(parity.Repo))
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range parity.Cases {
		if got := tok.Encode(c.Text); !slices.Equal(got, c.IDs) {
			t.Errorf("Encode(%q)\n got %v\nwant %v", c.Text, got, c.IDs)
		}
	}
}
