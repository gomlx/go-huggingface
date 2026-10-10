package hftokenizer

import (
	"testing"

	"github.com/gomlx/go-huggingface/tokenizers/api"
)

// These tests guard against a byte/rune accounting bug in
// applyNormalizerWithOffsets: the returned offsets slice must have exactly
// one entry per BYTE of the returned normalized string, because downstream
// code (encodeCore, preTokenizeWithOffsets, tokenizeWordWithSpans) treats
// offsets[i] as a direct byte-slice index into the normalized text.
//
// The bug: for a "BertNormalizer" configured like a real cased model
// (dslim/bert-base-NER's actual tokenizer.json: strip_accents=null,
// lowercase=false, handle_chinese_chars=true — every accented Latin
// character and CJK character passes through UNCHANGED or is re-emitted
// as-is), the code appended exactly ONE offset entry per RUNE (`for range
// s` / one `WriteRune(r)` call) while writing potentially MULTIPLE bytes
// (`result.WriteString(s)` / a multi-byte `WriteRune(r)`). Every
// multi-byte character under-fills offsets by one entry relative to
// len(result). The deficit compounds across the string until downstream
// code indexes offsets past its (too-short) length, which is the direct
// mechanism behind the "slice bounds out of range" panics observed live in
// github.com/knights-analytics/hugot-based NER pipelines on
// Romanian/Spanish/Portuguese/CJK-heavy text.

// bertNERNormalizer mirrors the ACTUAL normalizer block from the
// optimum/bert-base-NER tokenizer.json used in production by
// baditaflorin/go_named_entity_recognizer: a cased model that neither
// strips accents nor lowercases.
func bertNERNormalizer() *Normalizer {
	return &Normalizer{
		Type:               "BertNormalizer",
		CleanText:          true,
		HandleChineseChars: true,
		StripAccents:       nil, // exactly matches optimum/bert-base-NER's tokenizer.json: "strip_accents": null
		Lowercase:          false,
	}
}

// assertOffsetsCoverBytes is the core invariant this bug violates: offsets
// must have exactly one entry per byte of the normalized output.
func assertOffsetsCoverBytes(t *testing.T, label, input, normalized string, offsets []int) {
	t.Helper()
	if len(offsets) != len(normalized) {
		t.Fatalf("%s: len(offsets)=%d != len(normalized)=%d (input=%q, normalized=%q, offsets=%v)",
			label, len(offsets), len(normalized), input, normalized, offsets)
	}
	// Every offset must be a valid byte index into the ORIGINAL input
	// (or, for the identity case, equal to len(input) is never expected
	// here since offsets record start positions of source runes).
	for i, off := range offsets {
		if off < 0 || off > len(input) {
			t.Fatalf("%s: offsets[%d]=%d out of range for input length %d (input=%q)", label, i, off, len(input), input)
		}
	}
}

func TestApplyNormalizerWithOffsets_BertNormalizer_AccentedLatin(t *testing.T) {
	tok := &Tokenizer{}
	n := bertNERNormalizer()

	cases := []struct {
		name string
		text string
	}{
		{"single 2-byte accented char", "Ștefan"},
		{"name with multiple accents", "Ștefan Popescu"},
		{"german umlaut", "München"},
		{"spanish tilde and accents", "Múnich Íñigo"},
		{"portuguese", "São Paulo Brasília"},
		{"romanian full sentence", "Ștefan Popescu s-a născut în România și a studiat la Universitatea din București."},
		{"spanish full sentence", "El presidente de España visitó Múnich la semana pasada junto a Íñigo Fernández."},
		{"portuguese full sentence", "O diretor da empresa em São Paulo anunciou investimentos em Brasília e no Porto."},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			normalized, offsets := tok.applyNormalizerWithOffsets(tc.text, n)
			assertOffsetsCoverBytes(t, tc.name, tc.text, normalized, offsets)
		})
	}
}

func TestApplyNormalizerWithOffsets_BertNormalizer_ChineseChars(t *testing.T) {
	tok := &Tokenizer{}
	n := bertNERNormalizer()

	cases := []struct {
		name string
		text string
	}{
		{"cjk only", "习近平"},
		{"cjk sentence", "习近平主席访问了北京大学并会见了马云和李彦宏。"},
		{"mixed latin and cjk", "Tim Cook visited 北京 for Apple."},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			normalized, offsets := tok.applyNormalizerWithOffsets(tc.text, n)
			assertOffsetsCoverBytes(t, tc.name, tc.text, normalized, offsets)
		})
	}
}

// TestApplyNormalizerWithOffsets_BertNormalizer_OffsetsResolveCorrectSubstring
// goes one step further than byte-count parity: it confirms that for every
// byte position in the normalized string, offsets[pos] actually points at
// a byte position in the original text that is part of the SAME source
// rune — i.e. the mapping is not just the right length, it is right.
func TestApplyNormalizerWithOffsets_BertNormalizer_OffsetsResolveCorrectSubstring(t *testing.T) {
	tok := &Tokenizer{}
	n := bertNERNormalizer()

	text := "Ștefan Popescu vizitează Bucureşti."
	normalized, offsets := tok.applyNormalizerWithOffsets(text, n)
	assertOffsetsCoverBytes(t, "resolve-substring", text, normalized, offsets)

	// Since strip_accents=false and lowercase=false, BertNormalizer with
	// clean text + no whitespace collapsing beyond ASCII space should
	// reproduce the input verbatim (modulo whitespace normalization to a
	// single ASCII space, which this input doesn't exercise).
	if normalized != text {
		t.Fatalf("expected byte-identical passthrough for a cased normalizer with no accent-stripping/lowercasing, got %q from %q", normalized, text)
	}
	// Every byte of a given source rune's output maps back to that rune's
	// START byte position (matching the convention already used for the
	// whitespace/Chinese-char branches, which assign the same origPos to
	// every entry they append for one source rune) — NOT to its own byte
	// index. Compute the expected start-of-rune position for every byte of
	// the input independently (via utf8 rune boundaries) and compare.
	wantStart := make([]int, len(text))
	pos := 0
	for _, r := range text {
		n := len(string(r))
		for k := 0; k < n; k++ {
			wantStart[pos+k] = pos
		}
		pos += n
	}
	for i, off := range offsets {
		if off != wantStart[i] {
			t.Errorf("offsets[%d]=%d, want %d (should map to the start byte of the source rune covering output byte %d)", i, off, wantStart[i], i)
		}
	}
}

// TestApplyNormalizerWithOffsets_Lowercase_MultiByteResult guards the same
// class of bug in the "Lowercase" branch: offsets must be filled one entry
// per output BYTE, not one per output RUNE, even though the offsets slice
// itself is (correctly) pre-sized to len(normalized) bytes. Before the fix
// this branch didn't panic (bounds-checked) but silently left trailing
// entries at their zero value for any text containing multi-byte
// lowercased characters, corrupting the offset mapping.
func TestApplyNormalizerWithOffsets_Lowercase_MultiByteResult(t *testing.T) {
	tok := &Tokenizer{}
	n := &Normalizer{Type: "Lowercase"}

	cases := []string{
		"MÜNCHEN",
		"ÎNTÂLNIRE",
		"São Paulo Ñandú",
	}
	for _, text := range cases {
		t.Run(text, func(t *testing.T) {
			normalized, offsets := tok.applyNormalizerWithOffsets(text, n)
			assertOffsetsCoverBytes(t, text, text, normalized, offsets)
			// No trailing zero-valued (unfilled) entries once real content
			// has already advanced origPos past 0: every offset actually
			// present in the source should be non-decreasing.
			for i := 1; i < len(offsets); i++ {
				if offsets[i] < offsets[i-1] {
					t.Errorf("offsets not monotonic at %d: offsets[%d]=%d < offsets[%d]=%d (likely an unfilled/zero-valued trailing entry)",
						i, i, offsets[i], i-1, offsets[i-1])
				}
			}
		})
	}
}

// TestIssue67_GemmaReplaceNormalizerSpans reproduces GitHub issue #67:
// When normalizer replaces " " with "▁" (as in Gemma's tokenizer),
// token byte spans drift because they are tracked in normalized text where "▁"
// takes 3 bytes instead of 1 byte in the original text.
func TestIssue67_GemmaReplaceNormalizerSpans(t *testing.T) {
	const gemmaTokenizerJSON = `{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<eos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "<bos>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 3, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 107, "content": "\n", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
  ],
  "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
  "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": false},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}],
    "pair": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}}, {"SpecialToken": {"id": "<bos>", "type_id": 1}}, {"Sequence": {"id": "B", "type_id": 1}}],
    "special_tokens": {"<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]}}
  },
  "decoder": {
    "type": "Sequence",
    "decoders": [
      {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
      {"type": "ByteFallback"},
      {"type": "Fuse"}
    ]
  },
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "<unk>",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": true,
    "byte_fallback": true,
    "ignore_merges": false,
    "vocab": {
      "<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3,
      "T": 4, "h": 5, "e": 6, "▁": 7, "q": 8, "u": 9, "i": 10, "c": 11, "k": 12, "b": 13, "r": 14, "o": 15, "w": 16, "n": 17,
      "Th": 18, "The": 19, "▁q": 20, "▁qu": 21, "▁qui": 22, "▁quic": 23, "▁quick": 24,
      "▁b": 25, "▁br": 26, "▁bro": 27, "▁brow": 28, "▁brown": 29, "\n": 107
    },
    "merges": [
      ["T", "h"], ["Th", "e"], ["▁", "q"], ["▁q", "u"], ["▁qu", "i"], ["▁qui", "c"], ["▁quic", "k"],
      ["▁", "b"], ["▁b", "r"], ["▁br", "o"], ["▁bro", "w"], ["▁brow", "n"]
    ]
  }
}`
	tok, err := NewFromContent(nil, []byte(gemmaTokenizerJSON))
	if err != nil {
		t.Fatalf("NewFromContent failed: %v", err)
	}
	if err := tok.With(api.EncodeOptions{AddSpecialTokens: true, IncludeSpans: true}); err != nil {
		t.Fatalf("With options failed: %v", err)
	}

	text := "The quick brown\nThe quick brown"
	enc := tok.EncodeWithAnnotations(text)

	wantIDs := []int{2, 19, 24, 29, 107, 19, 24, 29}
	if !intSliceEqual(enc.IDs, wantIDs) {
		t.Errorf("IDs = %v, want %v", enc.IDs, wantIDs)
	}

	wantSpans := []api.TokenSpan{
		{Start: -1, End: -1}, // <bos>
		{Start: 0, End: 3},   // "The"
		{Start: 3, End: 9},   // " quick"
		{Start: 9, End: 15},  // " brown"
		{Start: 15, End: 16}, // "\n"
		{Start: 16, End: 19}, // "The"
		{Start: 19, End: 25}, // " quick"
		{Start: 25, End: 31}, // " brown"
	}
	if !spansEqual(enc.Spans, wantSpans) {
		t.Errorf("Spans = %v, want %v", enc.Spans, wantSpans)
	}

	// Verify that text[span] actually corresponds to each token decoded text:
	for i, id := range enc.IDs {
		sp := enc.Spans[i]
		if sp.Start == -1 && sp.End == -1 {
			continue
		}
		if sp.Start < 0 || sp.End > len(text) || sp.Start > sp.End {
			t.Errorf("token %d (%q): invalid span [%d, %d] for text length %d",
				i, tok.Decode([]int{id}), sp.Start, sp.End, len(text))
			continue
		}
		gotText := text[sp.Start:sp.End]
		t.Logf("token %-9q span [%2d,%2d) -> text[span] = %q", tok.Decode([]int{id}), sp.Start, sp.End, gotText)
	}
}

func TestNormalizationType_StringAndParse(t *testing.T) {
	types := []struct {
		typ NormalizationType
		str string
	}{
		{NormalizationLowercase, "Lowercase"},
		{NormalizationBert, "BertNormalizer"},
		{NormalizationNFD, "NFD"},
		{NormalizationNFC, "NFC"},
		{NormalizationNFKD, "NFKD"},
		{NormalizationNFKC, "NFKC"},
		{NormalizationStripAccents, "StripAccents"},
		{NormalizationSequence, "Sequence"},
		{NormalizationReplace, "Replace"},
		{NormalizationPrepend, "Prepend"},
	}

	for _, tc := range types {
		if tc.typ.String() != tc.str {
			t.Errorf("%v.String() = %q, want %q", tc.typ, tc.typ.String(), tc.str)
		}
		if parsed := ParseNormalizationType(tc.str); parsed != tc.typ {
			t.Errorf("ParseNormalizationType(%q) = %v, want %v", tc.str, parsed, tc.typ)
		}
		n := &Normalizer{Type: tc.str}
		if n.NormalizationType() != tc.typ {
			t.Errorf("Normalizer.NormalizationType() = %v, want %v", n.NormalizationType(), tc.typ)
		}
	}

	if ParseNormalizationType("NonExistent") != NormalizationUnknown {
		t.Errorf("ParseNormalizationType(unknown) should be NormalizationUnknown")
	}
}

func TestApplyNormalizerWithOffsets_Replace(t *testing.T) {
	tok := &Tokenizer{}

	// Test literal string replacement: " " -> "▁"
	nString := &Normalizer{
		Type:    "Replace",
		Pattern: &Pattern{String: " "},
		Content: "▁",
	}
	text := "a b"
	norm, offsets := tok.applyNormalizerWithOffsets(text, nString)
	assertOffsetsCoverBytes(t, "replace-string", text, norm, offsets)
	// norm is "a▁b", 5 bytes: 'a' (1) + '▁' (3) + 'b' (1)
	wantOffsets := []int{0, 1, 1, 1, 2}
	if !intSliceEqual(offsets, wantOffsets) {
		t.Errorf("offsets = %v, want %v", offsets, wantOffsets)
	}

	// Test regex replacement: multiple digits to "NUM"
	nRegex := &Normalizer{
		Type:    "Replace",
		Pattern: &Pattern{Regex: `\d+`},
		Content: "NUM",
	}
	text2 := "page 123 end"
	norm2, offsets2 := tok.applyNormalizerWithOffsets(text2, nRegex)
	assertOffsetsCoverBytes(t, "replace-regex", text2, norm2, offsets2)
	// "page NUM end"
	// "page " (5 bytes: 0..4) -> 0, 1, 2, 3, 4
	// "NUM" (3 bytes) -> 5, 5, 5
	// " end" (4 bytes: 8..11) -> 8, 9, 10, 11
	wantOffsets2 := []int{0, 1, 2, 3, 4, 5, 5, 5, 8, 9, 10, 11}
	if !intSliceEqual(offsets2, wantOffsets2) {
		t.Errorf("offsets2 = %v, want %v", offsets2, wantOffsets2)
	}
}

