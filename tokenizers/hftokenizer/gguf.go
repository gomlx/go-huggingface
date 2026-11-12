package hftokenizer

import (
	"strings"

	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/pkg/errors"
)

// GGUF token_type constants from the GGML / llama.cpp specification.
const (
	ggufTokenTypeNormal      int32 = 1
	ggufTokenTypeUnknown     int32 = 2
	ggufTokenTypeControl     int32 = 3
	ggufTokenTypeUserDefined int32 = 4
	ggufTokenTypeByte        int32 = 6
)

// GGUFTokenizerParams holds the tokenizer vocabulary, scores, merges, and special-token
// metadata extracted from a GGUF file's "tokenizer.ggml.*" key-value entries.
//
// Defining this pure-Go struct in hftokenizer preserves the modularity boundary so
// tokenizers never imports gomlx or compute.
type GGUFTokenizerParams struct {
	// Model is "tokenizer.ggml.model": e.g. "bert" (WordPiece), "gpt2" (ByteLevel BPE),
	// or "llama" (SentencePiece Unigram).
	Model string

	// Pre is "tokenizer.ggml.pre" (optional pre-tokenizer identifier).
	Pre string

	// Tokens is "tokenizer.ggml.tokens", indexed by token ID.
	Tokens []string

	// Scores is "tokenizer.ggml.scores" (optional, used by Unigram models).
	Scores []float32

	// TokenTypes is "tokenizer.ggml.token_type" (optional; 1=normal, 2=unknown, 3=control, 4=user_defined, 6=byte).
	TokenTypes []int32

	// Merges is "tokenizer.ggml.merges" (optional, used by BPE models).
	Merges []string

	// Special token IDs (-1 when not present in GGUF metadata).
	BosTokenID     int
	EosTokenID     int
	UnknownTokenID int
	SepTokenID     int // populated from "tokenizer.ggml.seperator_token_id" or "tokenizer.ggml.separator_token_id"
	PadTokenID     int
	ClsTokenID     int
	MaskTokenID    int

	// Optional boolean flags when explicitly present in GGUF metadata.
	AddBosToken    *bool
	AddEosToken    *bool
	AddSpacePrefix *bool
	DoLowerCase    *bool
}

// NewFromGGUF constructs a pure-Go HuggingFace-compatible Tokenizer from GGUF
// "tokenizer.ggml.*" metadata without requiring tokenizer.json or tokenizer_config.json.
func NewFromGGUF(config *api.Config, params GGUFTokenizerParams) (*Tokenizer, error) {
	if len(params.Tokens) == 0 {
		return nil, errors.New("hftokenizer.NewFromGGUF: empty vocabulary (tokenizer.ggml.tokens is missing)")
	}

	modelKind := strings.ToLower(strings.TrimSpace(params.Model))
	tokens := params.Tokens
	if modelKind == "bert" || modelKind == "wp" {
		tokens = normalizeBertGGUFTokens(params.Tokens, params.TokenTypes)
	}

	vocab := make(map[string]int, len(tokens))
	var addedTokens []AddedToken
	hasByteTokens := false

	for id, tok := range tokens {
		if tok != "" {
			if _, exists := vocab[tok]; !exists {
				vocab[tok] = id
			}
		}
		if id < len(params.TokenTypes) {
			tt := params.TokenTypes[id]
			switch tt {
			case ggufTokenTypeControl, ggufTokenTypeUnknown:
				if tok != "" {
					addedTokens = append(addedTokens, AddedToken{
						ID:      id,
						Content: tok,
						Special: true,
					})
				}
			case ggufTokenTypeUserDefined:
				if tok != "" {
					addedTokens = append(addedTokens, AddedToken{
						ID:      id,
						Content: tok,
						Special: false,
					})
				}
			case ggufTokenTypeByte:
				hasByteTokens = true
			}
		}
	}

	tokenAt := func(id int) string {
		if id >= 0 && id < len(tokens) {
			return tokens[id]
		}
		return ""
	}

	unkID := params.UnknownTokenID
	unkStr := tokenAt(unkID)

	var tj TokenizerJSON
	tj.Version = "1.0"
	tj.AddedTokens = addedTokens

	switch modelKind {
	case "bert", "wp":
		if unkStr == "" {
			if id, ok := vocab["[UNK]"]; ok {
				unkID = id
				unkStr = "[UNK]"
			}
		}
		doLowerCase := inferBertLowerCase(params, vocab)
		tj.Normalizer = &Normalizer{
			Type:               "BertNormalizer",
			CleanText:          true,
			HandleChineseChars: true,
			Lowercase:          doLowerCase,
		}
		tj.PreTokenizer = &PreTokenizer{
			Type: "BertPreTokenizer",
		}
		tj.Model = Model{
			Type:                    "WordPiece",
			Vocab:                   vocab,
			UnkToken:                unkStr,
			ContinuingSubwordPrefix: "##",
			MaxInputCharsPerWord:    100,
		}
		tj.Decoder = &Decoder{
			Type:   "WordPiece",
			Prefix: "##",
		}

		clsID := params.ClsTokenID
		if clsID < 0 {
			clsID = params.BosTokenID
		}
		if clsID < 0 {
			if id, ok := vocab["[CLS]"]; ok {
				clsID = id
			}
		}
		sepID := params.SepTokenID
		if sepID < 0 {
			sepID = params.EosTokenID
		}
		if sepID < 0 {
			if id, ok := vocab["[SEP]"]; ok {
				sepID = id
			}
		}
		tj.PostProcessor = buildBertPostProcessor(clsID, tokenAt(clsID), sepID, tokenAt(sepID))

	case "gpt2", "bpe":
		addSpacePrefix := false
		if params.AddSpacePrefix != nil {
			addSpacePrefix = *params.AddSpacePrefix
		}
		tj.PreTokenizer = &PreTokenizer{
			Type:           "ByteLevel",
			AddPrefixSpace: addSpacePrefix,
		}
		tj.Model = Model{
			Type:         "BPE",
			Vocab:        vocab,
			Merges:       params.Merges,
			UnkToken:     unkStr,
			ByteFallback: hasByteTokens,
		}
		tj.Decoder = &Decoder{
			Type: "ByteLevel",
		}

	case "llama", "spm", "t5":
		addSpacePrefix := true
		if params.AddSpacePrefix != nil {
			addSpacePrefix = *params.AddSpacePrefix
		}
		prependScheme := "always"
		if !addSpacePrefix {
			prependScheme = "never"
		}
		tj.PreTokenizer = &PreTokenizer{
			Type:           "Metaspace",
			Replacement:    "▁",
			PrependScheme:  prependScheme,
			AddPrefixSpace: addSpacePrefix,
		}
		tj.Model = Model{
			Type:         "Unigram",
			Vocab:        vocab,
			UnkToken:     unkStr,
			ByteFallback: hasByteTokens,
		}
		tj.Decoder = &Decoder{
			Type:          "Metaspace",
			Replacement:   "▁",
			PrependScheme: prependScheme,
			Split:         true,
		}

	default:
		return nil, errors.Errorf("hftokenizer.NewFromGGUF: unsupported tokenizer.ggml.model %q", params.Model)
	}

	if config == nil {
		config = &api.Config{}
	} else {
		cfgCopy := *config
		config = &cfgCopy
	}
	if params.AddBosToken != nil {
		config.AddBosToken = *params.AddBosToken
	} else if modelKind == "llama" && params.BosTokenID >= 0 {
		config.AddBosToken = true
	}
	if params.AddEosToken != nil {
		config.AddEosToken = *params.AddEosToken
	}

	tok, err := NewFromStruct(config, &tj)
	if err != nil {
		return nil, err
	}

	// Ensure every token ID from the GGUF array is present in idToToken even if empty/duplicate strings existed.
	for id, s := range tokens {
		tok.idToToken[id] = s
	}

	// Override special token IDs directly from GGUF metadata when present.
	if unkID >= 0 {
		tok.unkID = unkID
	}
	if params.PadTokenID >= 0 {
		tok.padID = params.PadTokenID
	}
	if params.BosTokenID >= 0 {
		tok.bosID = params.BosTokenID
	}
	if params.EosTokenID >= 0 {
		tok.eosID = params.EosTokenID
	}
	if params.ClsTokenID >= 0 {
		tok.clsID = params.ClsTokenID
	} else if modelKind == "bert" && params.BosTokenID >= 0 {
		tok.clsID = params.BosTokenID
	}
	if params.SepTokenID >= 0 {
		tok.sepID = params.SepTokenID
	} else if modelKind == "bert" && params.EosTokenID >= 0 {
		tok.sepID = params.EosTokenID
	}
	if params.MaskTokenID >= 0 {
		tok.maskID = params.MaskTokenID
	}

	return tok, nil
}

// normalizeBertGGUFTokens converts llama.cpp's PhantomSpace BERT GGUF vocabulary convention
// (where word-initial normal tokens start with "\u2581" and continuation subwords omit "##")
// back into standard HuggingFace WordPiece tokens ("word" and "##subword").
// If no token starts with "\u2581", rawTokens is returned unchanged.
func normalizeBertGGUFTokens(rawTokens []string, tokenTypes []int32) []string {
	usesPhantomSpace := false
	for _, t := range rawTokens {
		if strings.HasPrefix(t, "\u2581") {
			usesPhantomSpace = true
			break
		}
	}
	if !usesPhantomSpace {
		return rawTokens
	}
	out := make([]string, len(rawTokens))
	for id, g := range rawTokens {
		switch {
		case strings.HasPrefix(g, "\u2581"):
			out[id] = strings.TrimPrefix(g, "\u2581")
		case g == "" || (strings.HasPrefix(g, "[") && strings.HasSuffix(g, "]")) || (id < len(tokenTypes) && tokenTypes[id] != ggufTokenTypeNormal):
			out[id] = g
		default:
			out[id] = "##" + g
		}
	}
	return out
}

func inferBertLowerCase(params GGUFTokenizerParams, vocab map[string]int) bool {
	if params.DoLowerCase != nil {
		return *params.DoLowerCase
	}
	// Cased BERT models include capitalized English words such as "The" and "Of" in normal tokens;
	// uncased BERT models (such as all-MiniLM-L6-v2 and bert-base-uncased) do not.
	for _, probe := range []string{"The", "And", "Of", "In"} {
		if id, ok := vocab[probe]; ok {
			if id >= len(params.TokenTypes) || params.TokenTypes[id] == ggufTokenTypeNormal {
				return false
			}
		}
	}
	return true
}

func buildBertPostProcessor(clsID int, clsTok string, sepID int, sepTok string) *PostProcessor {
	if clsID < 0 && sepID < 0 {
		return nil
	}
	if clsTok == "" {
		clsTok = "[CLS]"
	}
	if sepTok == "" {
		sepTok = "[SEP]"
	}

	pp := &PostProcessor{
		Type:          "TemplateProcessing",
		SpecialTokens: make(map[string]PostProcSpecialToken, 2),
	}
	if clsID >= 0 {
		pp.SpecialTokens[clsTok] = PostProcSpecialToken{
			ID:     clsTok,
			IDs:    []int{clsID},
			Tokens: []string{clsTok},
		}
		pp.Single = append(pp.Single, PostProcItem{
			SpecialToken: &struct {
				ID     string `json:"id"`
				TypeID int    `json:"type_id"`
			}{ID: clsTok, TypeID: 0},
		})
		pp.Pair = append(pp.Pair, PostProcItem{
			SpecialToken: &struct {
				ID     string `json:"id"`
				TypeID int    `json:"type_id"`
			}{ID: clsTok, TypeID: 0},
		})
	}

	pp.Single = append(pp.Single, PostProcItem{
		Sequence: &struct {
			ID     string `json:"id"`
			TypeID int    `json:"type_id"`
		}{ID: "A", TypeID: 0},
	})
	pp.Pair = append(pp.Pair, PostProcItem{
		Sequence: &struct {
			ID     string `json:"id"`
			TypeID int    `json:"type_id"`
		}{ID: "A", TypeID: 0},
	})

	if sepID >= 0 {
		pp.SpecialTokens[sepTok] = PostProcSpecialToken{
			ID:     sepTok,
			IDs:    []int{sepID},
			Tokens: []string{sepTok},
		}
		pp.Single = append(pp.Single, PostProcItem{
			SpecialToken: &struct {
				ID     string `json:"id"`
				TypeID int    `json:"type_id"`
			}{ID: sepTok, TypeID: 0},
		})
		pp.Pair = append(pp.Pair,
			PostProcItem{
				SpecialToken: &struct {
					ID     string `json:"id"`
					TypeID int    `json:"type_id"`
				}{ID: sepTok, TypeID: 0},
			},
			PostProcItem{
				Sequence: &struct {
					ID     string `json:"id"`
					TypeID int    `json:"type_id"`
				}{ID: "B", TypeID: 1},
			},
			PostProcItem{
				SpecialToken: &struct {
					ID     string `json:"id"`
					TypeID int    `json:"type_id"`
				}{ID: sepTok, TypeID: 1},
			},
		)
	} else {
		pp.Pair = append(pp.Pair, PostProcItem{
			Sequence: &struct {
				ID     string `json:"id"`
				TypeID int    `json:"type_id"`
			}{ID: "B", TypeID: 1},
		})
	}
	return pp
}
