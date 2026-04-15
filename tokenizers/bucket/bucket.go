// Package bucket implements a streaming tokenizer of sentences into buckets (or batches) of discrete sizes, to minimize padding.
//
// Fundamentally, it takes as input a channel of sentences and their references (of type `any`, opaque to this library),
// and it sends on an output channel the sentences tokenized/padded to the same length in the corresponding buckets,
// along with their corresponding buckets. The order is lost because of the buffering of buckets until they fill.
//
// Example usage:
//
//	b := bucket.New(tokenizer).
//		ByPower(32, 8, 2).
//		WithMaxDelay(100*time.Millisecond, true)
//
//	input := make(chan bucket.SentenceRef)
//	output := make(chan bucket.Bucket, 10)
//
//	go func() {
//		defer close(input)
//		for i, text := range sentences {
//			input <- bucket.SentenceRef{Sentence: text, Reference: i}
//		}
//	}()
//
//	go b.Run(input, output)
//
//	for bk := range output {
//		// Process bk.Batch, bk.Shape, bk.References...
//	}
package bucket

import (
	"math"
	"math/bits"
	"runtime"
	"sync"
	"time"

	"github.com/gomlx/go-huggingface/tokenizers/api"
)

// Bucketizer represents the streaming bucketizer.
type Bucketizer struct {
	tokenizer api.Tokenizer

	// shapeFn maps sentence length to corresponding bucket shape.
	shapeFn ShapeFn

	// maxParallelization is the maximum number of sentences to tokenize in parallel.
	// Set to -1 to use runtime.NumCPU().
	maxParallelization int

	// maxDelay is the maximum delay between the first and last sentence in a bucket.
	// If this delay is reached, the bucket is flushed partially.
	//
	// This is useful for real-time applications where latency is more important than efficiency.
	maxDelay time.Duration

	// useBatchPadding is true if partially emitted batches should be padded to match the maximum BatchSize with empty sentences.
	useBatchPadding bool

	// Runtime state for Run
	output        chan<- Bucket
	padID         int
	activeBuckets map[Shape]*pendingBucket
	timer         *time.Timer
	timerChannel  <-chan time.Time
	timerDeadline time.Time
}

// pendingBucket holds the pre-allocated bucket out of which batches are yielded.
type pendingBucket struct {
	shape     Shape
	startedAt time.Time
	bucket    Bucket
	count     int
}

// Reference is an opaque type that is to conciliate sentences streamed into input -- since their order is lost
// during the bucketing.
type Reference any

// SentenceRef pair are the inputs to be tokenized and bucketed.
type SentenceRef struct {
	Sentence  string
	Reference any
}

// Shape describes the shape of a bucket.
type Shape struct {
	BatchSize, SentenceLength int
}

// ShapeFn is a function that determines the shape of a bucket given the sentence length.
//
// You can use arbitrary ShapeFn with the Bucketizer, see Bucketizer.WithShapeFn.
type ShapeFn func(sentenceLength int) Shape

// Bucket contains the tokenized sentences and their references.
type Bucket struct {
	// Shape of this bucket.
	Shape

	// Batch, with [Shape.BatchSize x Shape.SentenceLength] tokens, padded.
	Batch []int

	// References, with Shape.BatchSize references.
	References []Reference

	// NonPadTokens is the number of non-pad tokens in the batch.
	NonPadTokens int

	// Error, if any, that occurred during tokenization.
	// Errors are returned in individual buckets, with Shape.BatchSize == 1,
	// and References set to the failing SentenceRef.
	Error error
}

// New creates a new Bucketizer.
//
// By default it starts with a shape function grouping up to 32 elements with lengths rounded
// up to the lowest power of 2, starting with a minimum of 8 tokens (so 8, 16, 32, ...).
// It also sets the maximum tokenization parallelization to the number of available CPUs.
//
// Example usage:
//
//	b := bucket.New(tokenizer).
//		ByPower(32, 8, 2).
//		WithMaxDelay(100*time.Millisecond, true)
//
//	input := make(chan bucket.SentenceRef)
//	output := make(chan bucket.Bucket, 10)
//
//	go func() {
//		defer close(input)
//		for i, text := range sentences {
//			input <- bucket.SentenceRef{Sentence: text, Reference: i}
//		}
//	}()
//
//	go b.Run(input, output)
//
//	for bk := range output {
//		// Process bk.Batch, bk.Shape, bk.References...
//	}
func New(tokenizer api.Tokenizer) *Bucketizer {
	return (&Bucketizer{
		tokenizer: tokenizer,
	}).ByPower(32, 8, 2).WithMaxParallelization(-1)
}

// WithShapeFn sets the shape function to be used by the bucketizer.
//
// See also ByPower and ByPowerBudget for common shape functions.
func (b *Bucketizer) WithShapeFn(shapeFn ShapeFn) *Bucketizer {
	b.shapeFn = shapeFn
	return b
}

// ByPower configures the bucketizer to group sentences in buckets into fixed batchSize,
// where the sentence length is rounded up to the next power of base.
//
// The minSentenceLength is the smallest sentenceLength bucket.
//
// See also ByPowerBudget, if you want to round to a fixed tokens budget.
//
// For full control on the bucketing function, see WithShapeFn.
func (b *Bucketizer) ByPower(batchSize, minSentenceLength int, base float64) *Bucketizer {
	return b.WithShapeFn(func(sentenceLength int) Shape {
		sentenceLength = max(sentenceLength, minSentenceLength)
		bucketSentenceLen := int(math.Pow(base, math.Ceil(math.Log(float64(sentenceLength))/math.Log(base))))
		return Shape{
			BatchSize:      batchSize,
			SentenceLength: bucketSentenceLen,
		}
	})
}

// ByPowerBudget configures the bucketizer to group sentences in buckets with a fixed total number of
// tokens (the tokensBudget), and where the sentence length is rounded up to the next power of base.
//
// That means the batchSize is adjusted to keep the total number of tokens in the bucket close to tokensBudget,
// and hopefully the downstream tasks run on +/- constant time.
//
// For sentence lengths > tokenBudget, it simply uses batchSize = 1.
//
// For full control on the bucketing function, see WithShapeFn.
func (b *Bucketizer) ByPowerBudget(tokensBudget, minSentenceLength int, base float64) *Bucketizer {
	return b.WithShapeFn(func(sentenceLength int) Shape {
		sentenceLength = max(sentenceLength, minSentenceLength)
		bucketSentenceLen := int(math.Pow(base, math.Ceil(math.Log(float64(sentenceLength))/math.Log(base))))
		batchSize := max(tokensBudget/bucketSentenceLen, 1)
		return Shape{
			BatchSize:      batchSize,
			SentenceLength: bucketSentenceLen,
		}
	})
}

// ByTwoBitBucket configures the bucketizer to use buckets of sentence-length sized to the
// next value that can be represented with 2 bits. So: 1, 2, 3, 4, 6, 8, 12, 16, ...
//
// This is a "2-bit semi-log bucketing", and each size is separated from the other by
// a factor of 1.5 or 1.333 alternatingly, on average, by a factor of 1.414 (sqrt(2)),
// but results in numbers that are "friendlier" for binary addressing (and memory pages, etc.).
//
// For full control on the bucketing function, see WithShapeFn.
func (b *Bucketizer) ByTwoBitBucket(batchSize, minSentenceLength int) *Bucketizer {
	return b.WithShapeFn(func(sentenceLength int) Shape {
		sentenceLength = max(sentenceLength, minSentenceLength)
		bucketSentenceLen := TwoBitBucketLen(sentenceLength)
		return Shape{
			BatchSize:      batchSize,
			SentenceLength: bucketSentenceLen,
		}
	})
}

// ByTwoBitBucketBudget configures the bucketizer to use buckets of sentence-length sized to the
// next value that can be represented with 2 bits. So: 1, 2, 3, 4, 6, 8, 12, 16, ...
//
// This is a "2-bit semi-log bucketing", and each size is separated from the other by
// a factor of 1.5 or 1.333 alternatingly, on average, by a factor of 1.414 (sqrt(2)),
// but results in numbers that are "friendlier" for binary addressing (and memory pages, etc.).
//
// For full control on the bucketing function, see WithShapeFn.
func (b *Bucketizer) ByTwoBitBucketBudget(tokensBudget, minSentenceLength int) *Bucketizer {
	return b.WithShapeFn(func(sentenceLength int) Shape {
		sentenceLength = max(sentenceLength, minSentenceLength)
		bucketSentenceLen := TwoBitBucketLen(sentenceLength)
		batchSize := max(tokensBudget/bucketSentenceLen, 1)
		return Shape{
			BatchSize:      batchSize,
			SentenceLength: bucketSentenceLen,
		}
	})
}

// TwoBitBucketLen returns the smallest size >= unpaddedLen that uses only
// the two highest bits (either 2^n or 1.5 * 2^n).
//
// It is used by Bucketizer.ByTwoBitBucket and Bucketizer.ByTwoBitBucketBudget.
func TwoBitBucketLen(unpaddedLen int) int {
	if unpaddedLen <= 2 {
		return unpaddedLen
	}

	// Find the position of the most significant bit (MSB).
	// bits.Len returns the number of bits required to represent the uint.
	// For 5 (101), Len is 3.
	msbPos := bits.Len(uint(unpaddedLen)) - 1
	msbValue := 1 << msbPos

	// Case 1: Exact power of 2
	if unpaddedLen == msbValue {
		return msbValue
	}

	// Case 2: Check the "1.5" threshold (the two highest bits)
	// Example: If msbValue is 4 (100), threshold is 6 (110)
	threshold := msbValue | (msbValue >> 1)

	if unpaddedLen <= threshold {
		return threshold
	}

	// Case 3: Above the 1.5 threshold, jump to the next power of 2
	return msbValue << 1
}

// WithMaxParallelization sets the maximum number of sentences to tokenize in parallel.
//
// Set to -1 to use runtime.NumCPU().
func (b *Bucketizer) WithMaxParallelization(maxParallelization int) *Bucketizer {
	b.maxParallelization = maxParallelization
	return b
}

// WithBatchPadding sets whether partially emitted batches should be padded with
// empty sentences to match the maximum BatchSize.
//
// This can be useful with WithMaxDelay -- it can also be configured there --
// and if processing batches, to use on the final batches, when flushing the
// last buffered sentences.
func (b *Bucketizer) WithBatchPadding(useBatchPadding bool) *Bucketizer {
	b.useBatchPadding = useBatchPadding
	return b
}

// WithMaxDelay sets the maximum time a bucket can wait for more sentences
// before it is emitted partially.
//
// A value of 0 (default) means no timeout—buckets only emit when full or closed.
//
// An incomplete batch can either be returned partially (useBatchPadding == false) or
// padded with empty sentences (useBatchPadding == true). This is important if used
// fixed-shapes backends (e.g. XLA), which may expect full batches.
func (b *Bucketizer) WithMaxDelay(d time.Duration, useBatchPadding bool) *Bucketizer {
	b.maxDelay = d
	b.useBatchPadding = useBatchPadding
	return b
}

func (b *Bucketizer) newPendingBucket(shape Shape) *pendingBucket {
	batch := make([]int, shape.BatchSize*shape.SentenceLength)
	if b.padID != 0 {
		for i := range batch {
			batch[i] = b.padID
		}
	}
	refs := make([]Reference, shape.BatchSize)

	return &pendingBucket{
		shape:     shape,
		startedAt: time.Now(),
		count:     0,
		bucket: Bucket{
			Shape:      shape,
			Batch:      batch,
			References: refs,
		},
	}
}

func (b *Bucketizer) addTokenized(pb *pendingBucket, ref SentenceRef, ids []int) {
	idx := pb.count
	pb.bucket.References[idx] = ref.Reference

	length := min(len(ids), pb.shape.SentenceLength)
	dst := pb.bucket.Batch[idx*pb.shape.SentenceLength : (idx+1)*pb.shape.SentenceLength]
	copy(dst, ids[:length])

	pb.bucket.NonPadTokens += length
	pb.count++
}

func (b *Bucketizer) flushBucket(pb *pendingBucket) {
	if pb.count == 0 {
		return
	}

	result := pb.bucket
	if !b.useBatchPadding && pb.count < pb.shape.BatchSize {
		result.Shape.BatchSize = pb.count
		result.Batch = result.Batch[:pb.count*pb.shape.SentenceLength]
		result.References = result.References[:pb.count]
	}

	b.output <- result
	delete(b.activeBuckets, pb.shape)
}

func (b *Bucketizer) updateTimer() {
	if b.maxDelay <= 0 || len(b.activeBuckets) == 0 {
		if b.timer != nil {
			b.timer.Stop()
			b.timerChannel = nil
		}
		b.timerDeadline = time.Time{}
		return
	}
	var earliest time.Time
	for _, pb := range b.activeBuckets {
		if earliest.IsZero() || pb.startedAt.Before(earliest) {
			earliest = pb.startedAt
		}
	}
	b.timerDeadline = earliest
	duration := time.Until(earliest.Add(b.maxDelay))
	if duration <= 0 {
		duration = 1 // fire almost immediately
	}
	if b.timer == nil {
		b.timer = time.NewTimer(duration)
	} else {
		b.timer.Reset(duration)
	}
	b.timerChannel = b.timer.C
}

// Run starts the streaming of the sentences into buckets.
//
// It loops reading from the input, processing it, and writing to the output.
// The loop terminates when the input channel is closed.
//
// The order of the output is very likely not the same as the input, but one can
// reconciliate the order by using the Reference.
//
// By default this parallelizes the tokenization of sentences, but it can be configured
// with MaxParallelization.
func (b *Bucketizer) Run(input <-chan SentenceRef, output chan<- Bucket) {
	defer close(output)

	// Prevent leaking of state from this Run execution if Bucketizer is reused.
	defer func() {
		b.output = nil
		b.activeBuckets = nil
		if b.timer != nil {
			b.timer.Stop()
			b.timer = nil
			b.timerChannel = nil
		}
	}()

	b.output = output
	b.activeBuckets = make(map[Shape]*pendingBucket)

	b.padID = 0
	if id, err := b.tokenizer.SpecialTokenID(api.TokPad); err == nil {
		b.padID = id
	}

	maxParallelization := b.maxParallelization
	if maxParallelization <= 0 {
		maxParallelization = runtime.NumCPU()
	}

	type tokenized struct {
		ref SentenceRef
		ids []int
	}

	results := make(chan tokenized, maxParallelization*2)
	var wg sync.WaitGroup

	for i := 0; i < maxParallelization; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ref := range input {
				ids := b.tokenizer.Encode(ref.Sentence)
				results <- tokenized{ref: ref, ids: ids}
			}
		}()
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	for {
		select {
		case item, ok := <-results:
			if !ok {
				for _, pb := range b.activeBuckets {
					b.flushBucket(pb)
				}
				if b.timer != nil {
					b.timer.Stop()
				}
				return
			}

			shape := b.shapeFn(len(item.ids))
			pb, exists := b.activeBuckets[shape]
			if !exists {
				pb = b.newPendingBucket(shape)
				b.activeBuckets[shape] = pb
				if len(b.activeBuckets) == 1 {
					b.updateTimer()
				}
			}
			b.addTokenized(pb, item.ref, item.ids)

			if pb.count == pb.shape.BatchSize {
				timerTriggered := b.timerDeadline.Equal(pb.startedAt)
				b.flushBucket(pb)
				if timerTriggered {
					b.updateTimer()
				} else if len(b.activeBuckets) == 0 && b.timer != nil {
					b.timer.Stop()
					b.timerChannel = nil
					b.timerDeadline = time.Time{}
				}
			}

		case <-b.timerChannel:
			now := time.Now()
			for _, pb := range b.activeBuckets {
				if now.Sub(pb.startedAt) >= b.maxDelay {
					b.flushBucket(pb)
				}
			}
			b.updateTimer()
		}
	}
}
