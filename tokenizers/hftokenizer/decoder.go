package hftokenizer

import (
	"regexp"
	"strings"
	"fmt"
	"github.com/pkg/errors"
	"sort"
)

func compileDecoderRegex(decoder *Decoder) error {

	if decoder == nil {
		return nil
	}

	pattern := ""
	if decoder.Pattern != nil {
		if decoder.Pattern.Regex != "" {
			pattern = decoder.Pattern.Regex
		} else {
			pattern = regexp.QuoteMeta(decoder.Pattern.String)
		}
	}
	if pattern != "" {
		// Go regexes use RE2 standard, but some tokenizers use PCRE. We could fallback to https://github.com/dlclark/regexp2 in the future for PCRE support.
		re, err := regexp.Compile(pattern)
		if err != nil {
			return errors.Wrapf(err, "failed to compile regex %q: Go regex used here doesn't support PCRE lookahead/behind expressions, please open an issue to add support for this type of regex", pattern)
		}
		decoder.compiled = re
	}
	for _, childDecoder := range decoder.Decoders {
		err := compileDecoderRegex(childDecoder)
		if err != nil {
			return err
		}
	}
	return nil
}

// Decode converts a sequence of token IDs back to text.
func (t *Tokenizer) Decode(ids []int) string {
	var tokens []string
	for _, id := range ids {
		if token, ok := t.idToToken[id]; ok {
			tokens = append(tokens, token)
		}
	}

	// Apply decoder
	result := t.applyDecoder(tokens)
	return result
}

// applyDecoder applies the decoder to convert tokens back to text.
func (t *Tokenizer) applyDecoder(tokens []string) string {
	if t.tokenizer.Decoder == nil {
		// Default: handle WordPiece-style decoding
		return t.defaultDecode(tokens)
	}

	switch t.tokenizer.Decoder.Type {
	case "WordPiece":
		return t.wordPieceDecode(tokens)
	case "ByteLevel":
		return t.byteLevelDecode(tokens)
	case "Metaspace":
		return t.metaspaceDecode(tokens)
	case "BPEDecoder":
		return t.bpeDecode(tokens)
	case "Sequence":
		result := tokens
		for _, dec := range t.tokenizer.Decoder.Decoders {
			result = t.applyDecoderStep(result, dec)
		}
		return strings.Join(result, "")
	default:
		return t.defaultDecode(tokens)
	}
}

func (t *Tokenizer) applyDecoderStep(tokens []string, d *Decoder) []string {
	switch d.Type {
	case "Replace":
		// Replace pattern in tokens
		var result []string
		for _, tok := range tokens {
			result = append(result, d.compiled.ReplaceAllString(tok, d.Content))
		}
		return result
	case "Strip":
		// Strip characters
		content := d.Content
		if content == "" {
			content = " \t\n\r"
		}
		var result []string
		for _, tok := range tokens {
			result = append(result, strings.Trim(tok, content))
		}
		return result
	case "ByteFallback":
		// Handle byte fallback decoding
		// In byte fallback, tokens that represent a single byte are encoded as <0xXX>
		var result []string
		for _, tok := range tokens {
			if len(tok) == 6 && strings.HasPrefix(tok, "<0x") && strings.HasSuffix(tok, ">") {
				// Potential byte fallback token
				hex := tok[3:5]
				var b byte
				_, err := fmt.Sscanf(hex, "%02x", &b)
				if err == nil {
					result = append(result, string([]byte{b}))
					continue
				}
			}
			result = append(result, tok)
		}
		// If consecutive tokens are single bytes, they might form a multi-byte UTF-8 character.
		// However, standard ByteFallback decoder in HuggingFace usually just converts them to bytes.
		// The final join will then be a sequence of bytes which might be valid UTF-8.
		return result
	case "Metaspace":
		// Metaspace replaces leading space with a replacement character (default \u2581)
		replacement := d.Replacement
		if replacement == "" {
			replacement = "\u2581"
		}
		var result []string
		for i, tok := range tokens {
			decoded := strings.ReplaceAll(tok, replacement, " ")
			if i == 0 && d.PrependScheme == "always" && strings.HasPrefix(decoded, " ") {
				decoded = strings.TrimPrefix(decoded, " ")
			}
			result = append(result, decoded)
		}
		return result
	default:
		return tokens
	}
}

func (t *Tokenizer) defaultDecode(tokens []string) string {
	prefix := t.tokenizer.Model.ContinuingSubwordPrefix
	if prefix == "" {
		prefix = "##"
	}

	var result strings.Builder
	for i, token := range tokens {
		if strings.HasPrefix(token, prefix) {
			result.WriteString(strings.TrimPrefix(token, prefix))
		} else {
			if i > 0 {
				result.WriteString(" ")
			}
			result.WriteString(token)
		}
	}
	return result.String()
}

func (t *Tokenizer) wordPieceDecode(tokens []string) string {
	prefix := t.tokenizer.Decoder.Prefix
	if prefix == "" {
		prefix = "##"
	}

	var result strings.Builder
	for i, token := range tokens {
		if strings.HasPrefix(token, prefix) {
			result.WriteString(strings.TrimPrefix(token, prefix))
		} else {
			if i > 0 {
				result.WriteString(" ")
			}
			result.WriteString(token)
		}
	}
	return result.String()
}

func (t *Tokenizer) byteLevelDecode(tokens []string) string {
	// Join tokens and decode byte-level representation
	text := strings.Join(tokens, "")
	// The byte-level encoding uses special unicode characters
	// We need to map them back to bytes
	return byteLevelDecode(text)
}

func (t *Tokenizer) metaspaceDecode(tokens []string) string {
	replacement := t.tokenizer.Decoder.Replacement
	if replacement == "" {
		replacement = "\u2581"
	}
	prependScheme := t.tokenizer.Decoder.PrependScheme
	var result strings.Builder
	for i, token := range tokens {
		// Metaspace replaces leading space with special char
		decoded := strings.ReplaceAll(token, replacement, " ")
		if i == 0 && prependScheme == "always" {
			decoded = strings.TrimPrefix(decoded, " ")
		} else if i == 0 && t.tokenizer.PreTokenizer != nil && (t.tokenizer.PreTokenizer.AddPrefixSpace || t.tokenizer.PreTokenizer.PrependScheme == "always") {
			// Also check pre-tokenizer for compatibility
			decoded = strings.TrimPrefix(decoded, " ")
		}
		result.WriteString(decoded)
	}
	return result.String()
}

func (t *Tokenizer) bpeDecode(tokens []string) string {
	suffix := t.tokenizer.Model.EndOfWordSuffix

	var result strings.Builder
	for i, token := range tokens {
		if suffix != "" && strings.HasSuffix(token, suffix) {
			result.WriteString(strings.TrimSuffix(token, suffix))
			if i < len(tokens)-1 {
				result.WriteString(" ")
			}
		} else {
			result.WriteString(token)
		}
	}
	return result.String()
}

func byteLevelDecode(text string) string {
	var result []byte
	for _, r := range text {
		if b, ok := unicodeToByte[r]; ok {
			result = append(result, b)
		} else {
			// Fallback for characters not in the mapping
			result = append(result, []byte(string(r))...)
		}
	}
	return string(result)
}

// GetTokenizerType returns the model type (WordPiece, BPE, Unigram).
func (t *Tokenizer) GetTokenizerType() string {
	return t.tokenizer.Model.Type
}

// TokenToID converts a token string to its ID.
func (t *Tokenizer) TokenToID(token string) (int, bool) {
	if id, ok := t.addedTokens[token]; ok {
		return id, true
	}
	id, ok := t.tokenizer.Model.Vocab[token]
	return id, ok
}

// IDToToken converts a token ID to its string.
func (t *Tokenizer) IDToToken(id int) (string, bool) {
	token, ok := t.idToToken[id]
	return token, ok
}

// AddedTokensList returns the list of added tokens sorted by ID.
func (t *Tokenizer) AddedTokensList() []AddedToken {
	result := make([]AddedToken, len(t.tokenizer.AddedTokens))
	copy(result, t.tokenizer.AddedTokens)
	sort.Slice(result, func(i, j int) bool {
		return result[i].ID < result[j].ID
	})
	return result
}
