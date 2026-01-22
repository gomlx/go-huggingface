package sentencepiece

import (
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/tokenizers/api"
)

// TestEncodeWithOffsets_MatchesEncode verifies that EncodeWithOffsets produces the same IDs as Encode.
func TestEncodeWithOffsets_MatchesEncode(t *testing.T) {
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
			result := tok.EncodeWithOffsets(input)
			if !intSliceEqual(ids, result.IDs) {
				t.Errorf("Encode(%q) = %v, EncodeWithOffsets(%q).IDs = %v", input, ids, input, result.IDs)
			}
		})
	}
}

// TestEncodeWithOffsets_ValidOffsets verifies that offsets are valid (within bounds and non-overlapping).
func TestEncodeWithOffsets_ValidOffsets(t *testing.T) {
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
			result := tok.EncodeWithOffsets(input)

			if len(result.Offsets) != len(result.IDs) {
				t.Errorf("len(Offsets)=%d != len(IDs)=%d", len(result.Offsets), len(result.IDs))
			}

			for i, off := range result.Offsets {
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

// TestEncodeWithOffsets_EmptyString verifies behavior with empty input.
func TestEncodeWithOffsets_EmptyString(t *testing.T) {
	repo := hub.New("google/flan-t5-small")
	if !repo.HasFile("tokenizer.model") {
		t.Skip("tokenizer.model not found in repo")
	}

	baseTok, err := New(nil, repo)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	tok := baseTok.(*Tokenizer)

	result := tok.EncodeWithOffsets("")
	if len(result.IDs) != 0 {
		t.Errorf("Expected empty IDs for empty input, got %v", result.IDs)
	}
	if len(result.Offsets) != 0 {
		t.Errorf("Expected empty offsets for empty input, got %v", result.Offsets)
	}
}

// TestEncodeWithOffsets_Unicode verifies that Unicode text is handled correctly.
func TestEncodeWithOffsets_Unicode(t *testing.T) {
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
			result := tok.EncodeWithOffsets(input)

			// Verify IDs match Encode
			ids := tok.Encode(input)
			if !intSliceEqual(ids, result.IDs) {
				t.Errorf("IDs mismatch for %q", input)
			}

			// Verify offsets are valid
			for i, off := range result.Offsets {
				if off.Start < 0 || off.End > len(input) || off.Start > off.End {
					t.Errorf("Invalid offset at %d: [%d, %d] for input length %d",
						i, off.Start, off.End, len(input))
				}
			}
		})
	}
}

// TestTokenizerWithOffsetsInterface verifies the interface is correctly implemented.
func TestTokenizerWithOffsetsInterface(t *testing.T) {
	repo := hub.New("google/flan-t5-small")
	if !repo.HasFile("tokenizer.model") {
		t.Skip("tokenizer.model not found in repo")
	}

	tok, err := New(nil, repo)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	// Verify the tokenizer implements TokenizerWithOffsets
	var _ api.TokenizerWithOffsets = tok.(*Tokenizer)
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
