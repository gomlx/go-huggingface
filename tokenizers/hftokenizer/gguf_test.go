package hftokenizer

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewFromGGUF_Bert(t *testing.T) {
	params := GGUFTokenizerParams{
		Model: "bert",
		Tokens: []string{
			"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
			"hello", "world", "embed", "##ding", "##s", ",", ".", "!",
		},
		TokenTypes: []int32{
			3, 3, 3, 3, 3,
			1, 1, 1, 1, 1, 1, 1, 1,
		},
		BosTokenID:     2,
		EosTokenID:     3,
		UnknownTokenID: 1,
		SepTokenID:     3,
		PadTokenID:     0,
		ClsTokenID:     2,
		MaskTokenID:    4,
	}

	tok, err := NewFromGGUF(nil, params)
	require.NoError(t, err)
	assert.Equal(t, 13, tok.VocabSize())

	ids := tok.Encode("Hello, embeddings!")
	// [CLS]=2, "hello"=5, ","=10, "embed"=7, "##ding"=8, "##s"=9, "!"=12, [SEP]=3
	assert.Equal(t, []int{2, 5, 10, 7, 8, 9, 12, 3}, ids)

	clsID, err := tok.SpecialTokenID(api.TokClassification)
	require.NoError(t, err)
	assert.Equal(t, 2, clsID)

	padID, err := tok.SpecialTokenID(api.TokPad)
	require.NoError(t, err)
	assert.Equal(t, 0, padID)

	unkID, err := tok.SpecialTokenID(api.TokUnknown)
	require.NoError(t, err)
	assert.Equal(t, 1, unkID)
}

func TestNewFromGGUF_BertPhantomSpace(t *testing.T) {
	// llama.cpp convert_hf_to_gguf.py exports BERT WordPiece vocabularies with "\u2581" on
	// word-initial normal tokens and omits "##" on continuation subwords.
	params := GGUFTokenizerParams{
		Model: "bert",
		Tokens: []string{
			"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
			"\u2581what", "\u2581is", "\u2581embed", "ding", "s", "\u2581?",
		},
		TokenTypes: []int32{
			3, 2, 3, 3, 3,
			1, 1, 1, 1, 1, 1,
		},
		BosTokenID:     2,
		EosTokenID:     3,
		UnknownTokenID: 1,
		SepTokenID:     3,
		PadTokenID:     0,
		ClsTokenID:     2,
		MaskTokenID:    4,
	}

	tok, err := NewFromGGUF(nil, params)
	require.NoError(t, err)
	ids := tok.Encode("What is embeddings?")
	assert.Equal(t, []int{2, 5, 6, 7, 8, 9, 10, 3}, ids)
	assert.Equal(t, "what is embeddings ?", tok.Decode([]int{5, 6, 7, 8, 9, 10}))
}

func TestNewFromGGUF_MatchesMiniLMTokenizerJSON(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Skip("no home dir")
	}
	matches, _ := filepath.Glob(filepath.Join(
		home,
		".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*/tokenizer.json",
	))
	if len(matches) == 0 {
		t.Skip("cached sentence-transformers/all-MiniLM-L6-v2 tokenizer.json not present")
	}

	refTok, err := NewFromFile(nil, matches[0])
	require.NoError(t, err)

	// Reconstruct GGUF params from the reference vocabulary in ID order.
	vocab := refTok.GetVocab()
	tokens := make([]string, len(vocab))
	tokenTypes := make([]int32, len(vocab))
	for tokStr, id := range vocab {
		tokens[id] = tokStr
		tokenTypes[id] = 1
	}
	for _, specialID := range []int{0, 100, 101, 102, 103} {
		tokenTypes[specialID] = 3
	}

	ggufTok, err := NewFromGGUF(nil, GGUFTokenizerParams{
		Model:          "bert",
		Tokens:         tokens,
		TokenTypes:     tokenTypes,
		BosTokenID:     101,
		EosTokenID:     102,
		UnknownTokenID: 100,
		SepTokenID:     102,
		PadTokenID:     0,
		ClsTokenID:     101,
		MaskTokenID:    103,
	})
	require.NoError(t, err)
	assert.Equal(t, refTok.VocabSize(), ggufTok.VocabSize())

	sentences := []string{
		"That is a happy person",
		"Today is a sunny day!",
		" Café résumé — 2026 quantization (Q4_K_M & Q5_0) test.",
		"中文测试 mixed with English subwords: transformerization!",
	}
	for _, s := range sentences {
		assert.Equal(t, refTok.Encode(s), ggufTok.Encode(s), "mismatch on %q", s)
	}
}
