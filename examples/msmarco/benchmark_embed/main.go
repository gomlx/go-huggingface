package main

import (
	"flag"
	"fmt"
	"math/bits"
	"sync"
	"sync/atomic"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/lipgloss/table"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-huggingface/datasets"
	bge "github.com/gomlx/go-huggingface/examples/BAAI-bge-small-en-v1.5"
	"github.com/gomlx/go-huggingface/examples/kalmgemma3"
	"github.com/gomlx/go-huggingface/examples/msmarco"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/go-huggingface/tokenizers"
	tapi "github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/go-huggingface/tokenizers/bucket"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/support/humanize"
	"k8s.io/klog/v2"
)

var (
	flagModel        = flag.String("model", kalmgemma3.Repository, fmt.Sprintf("Model repository to use. Examples: %q, %q", kalmgemma3.Repository, bge.Repository))
	flagLimit        = flag.Int("limit", -1, "Limit the number of queries indexed. Set <= 0 to use all.")
	flagMSMarcoSplit = flag.String("msmarco_split", msmarco.TrainSplit, "Split to read from MS MARCO dataset (e.g. 'train', 'validation', 'test')")
	flagBucketBudget = flag.Int("bucket", 8*1024,
		"Bucket budget in number of tokens: sentences will be batched in buckets of this number of tokens in total. "+
			"So sentences with 128 tokens, if the budget is 1K, the batchsSize will be 8. "+
			"The buckets use the 'two-bits' algorithm to minimize padding -- e.g.: sizes 8, 12, 16, 24, 32, 48, etc.")
	flagParallelEmbedders = flag.Int("num_embedders", 1, "Number of parallel embedders. "+
		"The optimal value depends on the backend and the bucket size (-bucket), for GPUs usually 1 or 2 is enough.")
)

func MapHas[K comparable, V any](m map[K]V, k K) bool {
	_, ok := m[k]
	return ok
}

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	repo := hub.New(*flagModel)
	if err := repo.DownloadInfo(false); err != nil {
		klog.Fatalf("Failed to load repo info for %q: %v", *flagModel, err)
	}

	model := mustRunWithElapsedTime("Loading model configurations", func() (*transformer.Model, error) {
		return transformer.LoadModel(repo)
	})

	tokenizer := mustRunWithElapsedTime("Loading tokenizer", func() (tokenizers.Tokenizer, error) {
		return model.GetTokenizer()
	})

	backend := mustRunWithElapsedTime("Initializing backend", func() (compute.Backend, error) {
		return compute.New()
	})
	ctx := context.New()

	mustRunWithElapsedTime("Loading variables into context", func() (any, error) {
		return nil, model.LoadContext(backend, ctx)
	})

	padID := 0
	if id, err := tokenizer.SpecialTokenID(tapi.TokPad); err == nil {
		padID = id
	}

	embedExec, err := context.NewExec(backend, ctx.Checked(false), func(ctx *context.Context, tokens *graph.Node) *graph.Node {
		constPadID := graph.Scalar(tokens.Graph(), tokens.DType(), padID)
		mask := graph.NotEqual(tokens, constPadID)
		x := model.SentenceEmbeddingGraph(ctx, tokens, mask)
		return graph.ConvertDType(x, dtypes.Float32)
	})

	// Structured concurrency (keep track of goroutines).
	var wg sync.WaitGroup

	// Start bucket runner in a separate goroutine.
	bucketsInputChan := make(chan bucket.SentenceRef, 5)
	bucketsOutputChan := make(chan bucket.Bucket, 10)
	bkt := bucket.New(tokenizer).
		ByTwoBitBucketBudget(*flagBucketBudget, 8).
		WithMaxParallelization(-1).
		WithBatchPadding(true)

	wg.Go(func() {
		bkt.Run(bucketsInputChan, bucketsOutputChan)
	})

	// Dataset preparation and stats.
	ds := datasets.New(msmarco.ID)
	limit := *flagLimit
	dsInfo, err := ds.Info()
	if err != nil {
		klog.Fatalf("Failed to get dataset info: %v", err)
	}
	if !MapHas(dsInfo.DatasetInfo, msmarco.Config) || !MapHas(dsInfo.DatasetInfo[msmarco.Config].Splits, *flagMSMarcoSplit) {
		klog.Fatalf("Dataset %q doesn't contents for config=%q / split=%q", ds.ID, msmarco.Config, *flagMSMarcoSplit)
	}
	splitInfo := dsInfo.DatasetInfo[msmarco.Config].Splits[*flagMSMarcoSplit]
	totalQueries := splitInfo.NumExamples
	fmt.Printf("- Dataset %q, split %q: %d queries in total\n", ds.ID, *flagMSMarcoSplit, totalQueries)
	if limit > 0 {
		limit = min(limit, int(totalQueries))
	}

	var numSentencesRead int32

	// Start goroutine that feeds the bucket runner with passages.
	wg.Go(func() {
		defer close(bucketsInputChan)

		var numQueriesRead int
		for record, err := range datasets.IterParquetFromDatasetAt[msmarco.MsMarcoRecord](ds, msmarco.Config, *flagMSMarcoSplit, 0) {
			if err != nil {
				klog.Fatalf("Dataset iterator error: %v", err)
			}

			numQueriesRead++

			// There should be at most 10 passages per query in the datasets, but
			// just in case we enforce the limit.
			pLens := len(record.Passages.PassageText)
			for queryPassageIdx := range pLens {
				text := record.Passages.PassageText[queryPassageIdx]
				if text == "" {
					continue
				}

				bucketsInputChan <- bucket.SentenceRef{
					Sentence:  text,
					Reference: atomic.AddInt32(&numSentencesRead, 1),
				}
			}

			if limit > 0 && numQueriesRead >= limit {
				break
			}
		}
	})

	// Process batches and observe performance.
	startTime := time.Now()
	var numTokensProcessed, numNonPadTokensProcessed int64
	var numSentencesProcessed int
	expectedNumQueries := limit
	if expectedNumQueries <= 0 {
		expectedNumQueries = int(totalQueries)
	}
	var emaSpeed float64
	var emaInitialized bool

	fmt.Printf("- Starting processing:\n")
	numSentencesProcessedChan := make(chan int)
	var emaMu sync.RWMutex

	wg.Go(func() {
		lastReportTime := time.Now()
		var sentencesPerSecond float64
		for count := range numSentencesProcessedChan {
			// fmt.Printf("\r- Got %d sentences processed%s\n", count, humanize.EraseToEndOfLine)
			numSentencesProcessed += count

			// Report progress every second.
			if time.Since(lastReportTime) > time.Second {
				lastReportTime = time.Now()

				// ETA estimation.
				sentencesPerSecond = float64(numSentencesProcessed) / time.Since(startTime).Seconds()
				eta := "Unknown"
				expectedTotalSentences := 10 * expectedNumQueries
				if numSentencesProcessed > 0 {
					remainingSeconds := float64(expectedTotalSentences-numSentencesProcessed) / sentencesPerSecond
					if remainingSeconds > 0 {
						eta = humanize.Duration(time.Duration(int64(remainingSeconds*1e9)) * time.Nanosecond)
					} else {
						eta = "done"
					}
				}
				emaMu.RLock()
				speed := emaSpeed
				emaMu.RUnlock()
				fmt.Printf("\r   - Processed %s / %s passages (%s, %s non-padding) -- ETA %s ...%s",
					humanize.Count(int64(numSentencesProcessed)), humanize.Count(int64(expectedTotalSentences)), humanize.Speed(sentencesPerSecond, "passages"),
					humanize.Speed(speed, "tokens"), eta, humanize.EraseToEndOfLine)
			}
		}
		expectedTotalSentences := 10 * expectedNumQueries
		emaMu.RLock()
		speed := emaSpeed
		emaMu.RUnlock()
		fmt.Printf("\r  ✅ Processed %s / %s passages (%s, %s non-padding) -- done.%s\n",
			humanize.Count(int64(numSentencesProcessed)), humanize.Count(int64(expectedTotalSentences)), humanize.Speed(sentencesPerSecond, "passages"),
			humanize.Speed(speed, "tokens"), humanize.EraseToEndOfLine)
	})

	var embeddersWg sync.WaitGroup
	for i := 0; i < *flagParallelEmbedders; i++ {
		embeddersWg.Go(func() {
			for bk := range bucketsOutputChan {
				if bk.Error != nil {
					klog.Fatalf("Tokenization error: %v", bk.Error)
				}
				// fmt.Printf("\r- Processing batch shaped [%d, %d] (%d non-padding tokens)%s\n",
				// 	bk.Shape.BatchSize, bk.Shape.SentenceLength, bk.NonPadTokens, humanize.EraseToEndOfLine)

				batchSize := bk.Shape.BatchSize
				seqLen := bk.Shape.SentenceLength

				rawData := dtypes.UnsafeByteSlice(bk.Batch)
				var dtype dtypes.DType
				switch bits.UintSize {
				case 32:
					dtype = dtypes.Int32
				case 64:
					dtype = dtypes.Int64
				default:
					klog.Fatalf("Unsupported int of %d-bits architecture", bits.UintSize)
				}
				inputTensor, err := tensors.FromRaw(backend, 0, shapes.Make(dtype, batchSize, seqLen), rawData)
				if err != nil {
					klog.Fatalf("Failed to create input tensor: %+v", err)
				}

				batchStartTime := time.Now()
				var outTensor *tensors.Tensor
				outTensor, err = embedExec.Exec1(inputTensor)
				if err != nil {
					fmt.Println()
					klog.Fatalf("Failed to execute embeddings for %s: %+v", inputTensor.Shape(), err)
				}
				atomic.AddInt64(&numTokensProcessed, int64(len(bk.Batch)))
				atomic.AddInt64(&numNonPadTokensProcessed, int64(bk.NonPadTokens))

				// Here we simply discard the embeddings.
				// In a real application, you would save them or use them.

				inputTensor.FinalizeAll()
				outTensor.FinalizeAll()

				// Moving average of (non-padding) tokens per second speed.
				batchDuration := time.Since(batchStartTime).Seconds()
				if batchDuration > 0 {
					currentSpeed := float64(bk.NonPadTokens) / batchDuration
					emaMu.Lock()
					if !emaInitialized {
						emaSpeed = currentSpeed
						emaInitialized = true
					} else {
						emaSpeed = 0.1*currentSpeed + 0.9*emaSpeed
					}
					emaMu.Unlock()
				}

				// Count sentences processed.
				count := 0
				for _, ref := range bk.References {
					if ref != nil {
						count++
					}
				}
				numSentencesProcessedChan <- count
			}
		})
	}

	wg.Go(func() {
		embeddersWg.Wait()
		close(numSentencesProcessedChan)
	})

	wg.Wait()
	elapsed := time.Since(startTime)
	fmt.Printf("Total duration: %v\n", humanize.Duration(elapsed))

	// Print nice report table with counts and speeds.
	names := []string{"Passages", "Tokens", "Non-Pad Tokens"}
	totals := []string{humanize.Count(int64(numSentencesProcessed)), humanize.Count(numTokensProcessed), humanize.Count(numNonPadTokensProcessed)}
	speeds := []string{
		humanize.Speed(float64(numSentencesProcessed)/elapsed.Seconds(), " items"),
		humanize.Speed(float64(numTokensProcessed)/elapsed.Seconds(), "tokens"),
		humanize.Speed(float64(numNonPadTokensProcessed)/elapsed.Seconds(), "tokens"),
	}
	baseStyle := lipgloss.NewStyle().Padding(0, 1)
	t := table.New().
		Border(lipgloss.NormalBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(lipgloss.Color("238"))).
		Headers("Metric", "Total", "Speed").
		StyleFunc(func(row, col int) lipgloss.Style {
			s := baseStyle
			if col > 0 && row != table.HeaderRow {
				s = s.Align(lipgloss.Right)
			}
			if row == table.HeaderRow {
				headerStyle := s.Foreground(lipgloss.Color("252")).Bold(true)
				if col > 0 {
					headerStyle = headerStyle.Align(lipgloss.Center)
				}
				return headerStyle
			}
			return s
		})
	for i := range names {
		t.Row(names[i], totals[i], speeds[i])
	}
	fmt.Println(t)
}

func mustRunWithElapsedTime[T any](name string, f func() (T, error)) T {
	fmt.Printf("%s...", name)
	start := time.Now()
	ret, err := f()
	if err != nil {
		klog.Fatalf("failed: %v\n", err)
	}
	fmt.Printf("done (%s)\n", humanize.Duration(time.Since(start)))
	return ret
}
