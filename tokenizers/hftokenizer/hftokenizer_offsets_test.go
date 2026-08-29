package hftokenizer

import "testing"

// These tests guard against a byte/rune accounting bug in
// applyNormalizerWithSpans: the returned offsets slice must have exactly
// one entry per BYTE of the returned normalized string, because downstream
// code (encodeCore, preTokenizeWithSpans, tokenizeWordWithSpans) treats
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

func TestApplyNormalizerWithSpans_BertNormalizer_AccentedLatin(t *testing.T) {
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
			normalized, offsets := tok.applyNormalizerWithSpans(tc.text, n)
			assertOffsetsCoverBytes(t, tc.name, tc.text, normalized, offsets)
		})
	}
}

func TestApplyNormalizerWithSpans_BertNormalizer_ChineseChars(t *testing.T) {
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
			normalized, offsets := tok.applyNormalizerWithSpans(tc.text, n)
			assertOffsetsCoverBytes(t, tc.name, tc.text, normalized, offsets)
		})
	}
}

// TestApplyNormalizerWithSpans_BertNormalizer_OffsetsResolveCorrectSubstring
// goes one step further than byte-count parity: it confirms that for every
// byte position in the normalized string, offsets[pos] actually points at
// a byte position in the original text that is part of the SAME source
// rune — i.e. the mapping is not just the right length, it is right.
func TestApplyNormalizerWithSpans_BertNormalizer_OffsetsResolveCorrectSubstring(t *testing.T) {
	tok := &Tokenizer{}
	n := bertNERNormalizer()

	text := "Ștefan Popescu vizitează Bucureşti."
	normalized, offsets := tok.applyNormalizerWithSpans(text, n)
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

// TestApplyNormalizerWithSpans_Lowercase_MultiByteResult guards the same
// class of bug in the "Lowercase" branch: offsets must be filled one entry
// per output BYTE, not one per output RUNE, even though the offsets slice
// itself is (correctly) pre-sized to len(normalized) bytes. Before the fix
// this branch didn't panic (bounds-checked) but silently left trailing
// entries at their zero value for any text containing multi-byte
// lowercased characters, corrupting the offset mapping.
func TestApplyNormalizerWithSpans_Lowercase_MultiByteResult(t *testing.T) {
	tok := &Tokenizer{}
	n := &Normalizer{Type: "Lowercase"}

	cases := []string{
		"MÜNCHEN",
		"ÎNTÂLNIRE",
		"São Paulo Ñandú",
	}
	for _, text := range cases {
		t.Run(text, func(t *testing.T) {
			normalized, offsets := tok.applyNormalizerWithSpans(text, n)
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
