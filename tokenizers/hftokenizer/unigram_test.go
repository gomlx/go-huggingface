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
		// Expected ids come from HuggingFace tokenizers on the same tokenizer.json.
		want := map[string][]int{"ab": {3, 4}, "abc": {3, 4, 5}, "axyb": {3, 0, 4}, "é": {1, 0}}
		if byteFallback {
			want["é"] = []int{1, 6, 7}
		}
		for text, ids := range want {
			if got := tok.Encode(text); !slices.Equal(got, ids) {
				t.Errorf("byte_fallback=%t Encode(%q) = %v, want %v", byteFallback, text, got, ids)
			}
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
	path, err := hub.New(parity.Repo).DownloadFile("tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	tok, err := NewFromFile(nil, path)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range parity.Cases {
		if got := tok.Encode(c.Text); !slices.Equal(got, c.IDs) {
			t.Errorf("Encode(%q)\n got %v\nwant %v", c.Text, got, c.IDs)
		}
	}
}
