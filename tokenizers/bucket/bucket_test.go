package bucket

import (
	"testing"
	"testing/synctest"
	"time"

	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestByPower(t *testing.T) {
	b := &Bucketizer{}
	b.ByPower(32, 8, 2.0)

	testCases := []struct {
		length int
		want   Shape
	}{
		{length: 1, want: Shape{BatchSize: 32, SentenceLength: 8}}, // max(1, 8) -> 8
		{length: 7, want: Shape{BatchSize: 32, SentenceLength: 8}},
		{length: 8, want: Shape{BatchSize: 32, SentenceLength: 8}},
		{length: 9, want: Shape{BatchSize: 32, SentenceLength: 16}},
		{length: 15, want: Shape{BatchSize: 32, SentenceLength: 16}},
		{length: 16, want: Shape{BatchSize: 32, SentenceLength: 16}},
		{length: 17, want: Shape{BatchSize: 32, SentenceLength: 32}},
		{length: 32, want: Shape{BatchSize: 32, SentenceLength: 32}},
		{length: 33, want: Shape{BatchSize: 32, SentenceLength: 64}},
	}

	for _, tc := range testCases {
		got := b.shapeFn(tc.length)
		if got != tc.want {
			t.Errorf("ByPower(length: %d) = %+v, want %+v", tc.length, got, tc.want)
		}
	}

	// Test base 10
	b10 := &Bucketizer{}
	b10.ByPower(32, 10, 10.0)
	testCases10 := []struct {
		length int
		want   Shape
	}{
		{length: 1, want: Shape{BatchSize: 32, SentenceLength: 10}},
		{length: 10, want: Shape{BatchSize: 32, SentenceLength: 10}},
		{length: 11, want: Shape{BatchSize: 32, SentenceLength: 100}},
		{length: 100, want: Shape{BatchSize: 32, SentenceLength: 100}},
		{length: 101, want: Shape{BatchSize: 32, SentenceLength: 1000}},
	}
	for _, tc := range testCases10 {
		got := b10.shapeFn(tc.length)
		if got != tc.want {
			t.Errorf("ByPower_Base10(length: %d) = %+v, want %+v", tc.length, got, tc.want)
		}
	}
}

func TestByPowerBudget(t *testing.T) {
	b := &Bucketizer{}
	b.ByPowerBudget(256, 8, 2.0)

	testCases := []struct {
		length int
		want   Shape
	}{
		{length: 1, want: Shape{BatchSize: 32, SentenceLength: 8}},
		{length: 8, want: Shape{BatchSize: 32, SentenceLength: 8}},
		{length: 9, want: Shape{BatchSize: 16, SentenceLength: 16}},
		{length: 16, want: Shape{BatchSize: 16, SentenceLength: 16}},
		{length: 17, want: Shape{BatchSize: 8, SentenceLength: 32}},
		{length: 256, want: Shape{BatchSize: 1, SentenceLength: 256}},
		{length: 257, want: Shape{BatchSize: 1, SentenceLength: 512}},
	}

	for _, tc := range testCases {
		got := b.shapeFn(tc.length)
		if got != tc.want {
			t.Errorf("ByPowerBudget(length: %d) = %+v, want %+v", tc.length, got, tc.want)
		}
	}

	// Test base 10
	b10 := &Bucketizer{}
	b10.ByPowerBudget(300, 10, 10.0)
	testCases10 := []struct {
		length int
		want   Shape
	}{
		{length: 1, want: Shape{BatchSize: 30, SentenceLength: 10}},
		{length: 10, want: Shape{BatchSize: 30, SentenceLength: 10}},
		{length: 11, want: Shape{BatchSize: 3, SentenceLength: 100}},
		{length: 100, want: Shape{BatchSize: 3, SentenceLength: 100}},
		{length: 101, want: Shape{BatchSize: 1, SentenceLength: 1000}}, // 300/1000 -> max(0, 1) -> 1
	}
	for _, tc := range testCases10 {
		got := b10.shapeFn(tc.length)
		if got != tc.want {
			t.Errorf("ByPowerBudget_Base10(length: %d) = %+v, want %+v", tc.length, got, tc.want)
		}
	}
}

// mockTokenizer for testing. It returns TokenIDs [1, 2, ..., len(text)].
type mockTokenizer struct {
	padID int
}

func (m *mockTokenizer) Encode(text string) []int {
	ids := make([]int, len(text))
	for i := range ids {
		ids[i] = i + 1
	}
	return ids
}

func (m *mockTokenizer) EncodeWithOptions(text string, addSpecialTokens bool) []int {
	return m.Encode(text)
}

func (m *mockTokenizer) Decode(ids []int) string { return "" }

func (m *mockTokenizer) SpecialTokenID(token api.SpecialToken) (int, error) {
	if token == api.TokPad {
		return m.padID, nil
	}
	return 0, nil
}

func TestBucketizerRun(t *testing.T) {
	mockTok := &mockTokenizer{padID: 99}
	// Shape: batchSize=2, length=min(8, 2^ceil(log2(len))) => lengths: 8, 16, 32...
	b := New(mockTok).ByPower(2, 8, 2.0).WithMaxParallelization(1)

	input := make(chan SentenceRef, 5)
	output := make(chan Bucket, 5)

	done := make(chan struct{})
	go func() {
		b.Run(input, output)
		close(done)
	}()

	// lengths shape into 8.
	input <- SentenceRef{Sentence: "123", Reference: "ref1"}   // len 3 => shape length 8
	input <- SentenceRef{Sentence: "12345", Reference: "ref2"} // len 5 => shape length 8
	// This completes 1 bucket of size 2.

	// length shape into 16.
	input <- SentenceRef{Sentence: "12345678901", Reference: "ref3"} // len 11 => shape length 16

	close(input)
	<-done

	var buckets []Bucket
	for bucket := range output {
		buckets = append(buckets, bucket)
	}

	require.Len(t, buckets, 2)

	// Since we use maxParallelization=1, we can predictably assume order of insertion into buckets.
	// But maps iteration is random, so we need to match them.
	var b8, b16 Bucket
	for _, bucket := range buckets {
		if bucket.Shape.SentenceLength == 8 {
			b8 = bucket
		} else if bucket.Shape.SentenceLength == 16 {
			b16 = bucket
		}
	}

	assert.Equal(t, 2, b8.Shape.BatchSize)
	assert.Equal(t, 1, b16.Shape.BatchSize, "Incomplete batch with useBatchPadding=false should have BatchSize=1")

	assert.Equal(t, []Reference{"ref1", "ref2"}, b8.References)
	assert.Equal(t, []Reference{"ref3"}, b16.References)

	// verify padding in b8
	// Expected Batch for b8:
	// ref1: [1, 2, 3, 99, 99, 99, 99, 99]
	// ref2: [1, 2, 3, 4, 5, 99, 99, 99]
	assert.Equal(t, []int{
		1, 2, 3, 99, 99, 99, 99, 99,
		1, 2, 3, 4, 5, 99, 99, 99,
	}, b8.Batch)

	// verify padding in b16
	// Expected Batch for b16 (BatchSize=1):
	// ref3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99, 99, 99, 99, 99]
	assert.Equal(t, []int{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99, 99, 99, 99, 99,
	}, b16.Batch)
}

func TestBucketizerRunWithMaxDelay(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		mockTok := &mockTokenizer{padID: 99}
		b := New(mockTok).ByPower(2, 8, 2.0).
			WithMaxParallelization(1).
			WithMaxDelay(100*time.Millisecond, true) // useBatchPadding = true

		input := make(chan SentenceRef, 5)
		output := make(chan Bucket, 5)

		done := make(chan struct{})
		go func() {
			b.Run(input, output)
			close(done)
		}()

		input <- SentenceRef{Sentence: "123", Reference: "ref1"}

		synctest.Wait() // wait until background goroutines block

		select {
		case <-output:
			t.Fatal("Bucket should not have been emitted yet")
		default:
		}

		// Advance time past the max delay
		time.Sleep(150 * time.Millisecond)
		synctest.Wait() // wait until it processes

		// Now we should receive the incomplete bucket.
		var bucket Bucket
		select {
		case bucket = <-output:
		default:
			t.Fatal("Bucket should have been emitted due to maxDelay")
		}

		assert.Equal(t, 2, bucket.Shape.BatchSize, "useBatchPadding=true should keep original BatchSize")
		require.Len(t, bucket.References, 2)
		assert.Equal(t, Reference("ref1"), bucket.References[0])
		assert.Nil(t, bucket.References[1])

		// Expected padding out
		assert.Equal(t, []int{
			// item 1
			1, 2, 3, 99, 99, 99, 99, 99,
			// item 2 (completely padded)
			99, 99, 99, 99, 99, 99, 99, 99,
		}, bucket.Batch)

		// Close cleanly
		close(input)
		<-done
	})
}
