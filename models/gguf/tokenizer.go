package gguf

import (
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/go-huggingface/tokenizers/hftokenizer"
	"github.com/pkg/errors"
)

// GGUF tokenizer metadata keys from the GGUF specification.
const (
	KeyTokenizerModel          = "tokenizer.ggml.model"
	KeyTokenizerPre            = "tokenizer.ggml.pre"
	KeyTokenizerTokens         = "tokenizer.ggml.tokens"
	KeyTokenizerScores         = "tokenizer.ggml.scores"
	KeyTokenizerTokenType      = "tokenizer.ggml.token_type"
	KeyTokenizerMerges         = "tokenizer.ggml.merges"
	KeyTokenizerBosID          = "tokenizer.ggml.bos_token_id"
	KeyTokenizerEosID          = "tokenizer.ggml.eos_token_id"
	KeyTokenizerUnknownID      = "tokenizer.ggml.unknown_token_id"
	KeyTokenizerSeperatorID    = "tokenizer.ggml.seperator_token_id" // historical GGUF spelling
	KeyTokenizerSeparatorID    = "tokenizer.ggml.separator_token_id"
	KeyTokenizerPaddingID      = "tokenizer.ggml.padding_token_id"
	KeyTokenizerClsID          = "tokenizer.ggml.cls_token_id"
	KeyTokenizerMaskID         = "tokenizer.ggml.mask_token_id"
	KeyTokenizerAddBos         = "tokenizer.ggml.add_bos_token"
	KeyTokenizerAddEos         = "tokenizer.ggml.add_eos_token"
	KeyTokenizerAddSpacePrefix = "tokenizer.ggml.add_space_prefix"
	KeyTokenizerDoLowerCase    = "tokenizer.ggml.do_lower_case"
)

// TokenizerParams extracts embedded GGUF "tokenizer.ggml.*" metadata into [hftokenizer.GGUFTokenizerParams].
// Returns false if the GGUF file does not contain an embedded vocabulary ("tokenizer.ggml.tokens").
func (f *File) TokenizerParams() (hftokenizer.GGUFTokenizerParams, bool) {
	if f == nil {
		return hftokenizer.GGUFTokenizerParams{}, false
	}
	tokensKV, ok := f.GetKeyValue(KeyTokenizerTokens)
	if !ok {
		return hftokenizer.GGUFTokenizerParams{}, false
	}
	tokens := tokensKV.Strings()
	if len(tokens) == 0 {
		return hftokenizer.GGUFTokenizerParams{}, false
	}

	optTokenID := func(keys ...string) int {
		for _, k := range keys {
			if kv, found := f.GetKeyValue(k); found {
				return int(kv.Int64())
			}
		}
		return -1
	}
	optBool := func(key string) *bool {
		if kv, found := f.GetKeyValue(key); found {
			if b, isBool := kv.Raw().(bool); isBool {
				return &b
			}
		}
		return nil
	}

	params := hftokenizer.GGUFTokenizerParams{
		Tokens:         tokens,
		BosTokenID:     optTokenID(KeyTokenizerBosID),
		EosTokenID:     optTokenID(KeyTokenizerEosID),
		UnknownTokenID: optTokenID(KeyTokenizerUnknownID),
		SepTokenID:     optTokenID(KeyTokenizerSeperatorID, KeyTokenizerSeparatorID),
		PadTokenID:     optTokenID(KeyTokenizerPaddingID),
		ClsTokenID:     optTokenID(KeyTokenizerClsID),
		MaskTokenID:    optTokenID(KeyTokenizerMaskID),
		AddBosToken:    optBool(KeyTokenizerAddBos),
		AddEosToken:    optBool(KeyTokenizerAddEos),
		AddSpacePrefix: optBool(KeyTokenizerAddSpacePrefix),
		DoLowerCase:    optBool(KeyTokenizerDoLowerCase),
	}

	if kv, found := f.GetKeyValue(KeyTokenizerModel); found {
		params.Model = kv.String()
	}
	if kv, found := f.GetKeyValue(KeyTokenizerPre); found {
		params.Pre = kv.String()
	}
	if kv, found := f.GetKeyValue(KeyTokenizerScores); found {
		if f32s, isF32 := kv.Raw().([]float32); isF32 {
			params.Scores = f32s
		} else if f64s := kv.Float64s(); len(f64s) > 0 {
			scores := make([]float32, len(f64s))
			for i, v := range f64s {
				scores[i] = float32(v)
			}
			params.Scores = scores
		}
	}
	if kv, found := f.GetKeyValue(KeyTokenizerTokenType); found {
		if i32s, isI32 := kv.Raw().([]int32); isI32 {
			params.TokenTypes = i32s
		} else if i64s := kv.Int64s(); len(i64s) > 0 {
			tts := make([]int32, len(i64s))
			for i, v := range i64s {
				tts[i] = int32(v)
			}
			params.TokenTypes = tts
		}
	}
	if kv, found := f.GetKeyValue(KeyTokenizerMerges); found {
		params.Merges = kv.Strings()
	}

	return params, true
}

// LoadTokenizer constructs a pure-Go [hftokenizer.Tokenizer] from the GGUF file's embedded
// "tokenizer.ggml.*" metadata.
func (f *File) LoadTokenizer(config *api.Config) (*hftokenizer.Tokenizer, error) {
	params, ok := f.TokenizerParams()
	if !ok {
		return nil, errors.New("gguf: file does not contain embedded tokenizer.ggml.tokens metadata")
	}
	return hftokenizer.NewFromGGUF(config, params)
}
