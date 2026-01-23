package sentencepiece

import (
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers/api"
)

// TestEncodeWithSpans_MatchesEncode verifies that EncodeWithSpans produces the same IDs as Encode.
func TestEncodeWithSpans_MatchesEncode(t *testing.T) {
	// Use a public model that has a sentencepiece tokenizer
	// google/flan-t5-small uses sentencepiece and is freely accessible
	repo := hub.New("google/flan-t5-small")
	if !repo.HasFile("tokenizer.model") {
		t.Skip("tokenizer.model not found in repo")
	}

	baseTok, err := New(nil, repo)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	tok := baseTok.(*Tokenizer)

	inputs := []string{
		"hello",
		"hello world",
		"The quick brown fox jumps over the lazy dog.",
		"Testing tokenization with offsets.",
		"Multiple  spaces   here",
	}

	for _, input := range inputs {
		t.Run(input, func(t *testing.T) {
			ids := tok.Encode(input)
			result := tok.EncodeWithSpans(input)
			if !intSliceEqual(ids, result.IDs) {
				t.Errorf("Encode(%q) = %v, EncodeWithSpans(%q).IDs = %v", input, ids, input, result.IDs)
			}
		})
	}
}

// TestEncodeWithSpans_ValidSpans verifies that spans are valid (within bounds).
func TestEncodeWithSpans_ValidSpans(t *testing.T) {
	repo := hub.New("google/flan-t5-small")
	if !repo.HasFile("tokenizer.model") {
		t.Skip("tokenizer.model not found in repo")
	}

	baseTok, err := New(nil, repo)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	tok := baseTok.(*Tokenizer)

	inputs := []string{
		"hello world",
		"The quick brown fox.",
		"Testing 123 numbers!",
	}

	for _, input := range inputs {
		t.Run(input, func(t *testing.T) {
			result := tok.EncodeWithSpans(input)

			if len(result.Spans) != len(result.IDs) {
				t.Errorf("len(Spans)=%d != len(IDs)=%d", len(result.Spans), len(result.IDs))
			}

			for i, off := range result.Spans {
				// Check bounds
				if off.Start < 0 {
					t.Errorf("Token %d: offset start %d is negative", i, off.Start)
				}
				if off.End > len(input) {
					t.Errorf("Token %d: offset end %d exceeds input length %d", i, off.End, len(input))
				}
				if off.Start > off.End {
					t.Errorf("Token %d: start %d > end %d", i, off.Start, off.End)
				}

				// Log for debugging
				if off.Start <= off.End && off.End <= len(input) {
					t.Logf("Token %d: ID=%d, offset=[%d,%d], text=%q",
						i, result.IDs[i], off.Start, off.End, input[off.Start:off.End])
				}
			}
		})
	}
}

// TestEncodeWithSpans_EmptyString verifies behavior with empty input.
func TestEncodeWithSpans_EmptyString(t *testing.T) {
	repo := hub.New("google/flan-t5-small")
	if !repo.HasFile("tokenizer.model") {
		t.Skip("tokenizer.model not found in repo")
	}

	baseTok, err := New(nil, repo)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	tok := baseTok.(*Tokenizer)

	result := tok.EncodeWithSpans("")
	if len(result.IDs) != 0 {
		t.Errorf("Expected empty IDs for empty input, got %v", result.IDs)
	}
	if len(result.Spans) != 0 {
		t.Errorf("Expected empty offsets for empty input, got %v", result.Spans)
	}
}

// TestEncodeWithSpans_Unicode verifies that Unicode text is handled correctly.
func TestEncodeWithSpans_Unicode(t *testing.T) {
	repo := hub.New("google/flan-t5-small")
	if !repo.HasFile("tokenizer.model") {
		t.Skip("tokenizer.model not found in repo")
	}

	baseTok, err := New(nil, repo)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	tok := baseTok.(*Tokenizer)

	inputs := []string{
		"Hello, world!",
		"Hello, 世界!",
		"日本語テスト",
		"Emoji: 🎉",
	}

	for _, input := range inputs {
		t.Run(input, func(t *testing.T) {
			result := tok.EncodeWithSpans(input)

			// Verify IDs match Encode
			ids := tok.Encode(input)
			if !intSliceEqual(ids, result.IDs) {
				t.Errorf("IDs mismatch for %q", input)
			}

			// Verify offsets are valid
			for i, off := range result.Spans {
				if off.Start < 0 || off.End > len(input) || off.Start > off.End {
					t.Errorf("Invalid offset at %d: [%d, %d] for input length %d",
						i, off.Start, off.End, len(input))
				}
			}
		})
	}
}

// TestTokenizerWithSpansInterface verifies the interface is correctly implemented.
func TestTokenizerWithSpansInterface(t *testing.T) {
	repo := hub.New("google/flan-t5-small")
	if !repo.HasFile("tokenizer.model") {
		t.Skip("tokenizer.model not found in repo")
	}

	tok, err := New(nil, repo)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	// Verify the tokenizer implements TokenizerWithSpans
	var _ api.TokenizerWithSpans = tok.(*Tokenizer)
}

func intSliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
