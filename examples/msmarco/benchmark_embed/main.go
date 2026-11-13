package main

import (
	"flag"
	"fmt"
	"math"
	"math/bits"
	"net/http"
	_ "net/http/pprof"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/lipgloss/table"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/datasets"
	bge "github.com/gomlx/go-huggingface/examples/BAAI-bge-small-en-v1.5"
	"github.com/gomlx/go-huggingface/examples/kalmgemma3"
	minilmgguf "github.com/gomlx/go-huggingface/examples/minilm-l6-v2-gguf"
	"github.com/gomlx/go-huggingface/examples/msmarco"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/go-huggingface/tokenizers"
	tapi "github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/go-huggingface/tokenizers/bucket"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/onnx-gomlx/onnx"
	"github.com/gomlx/onnx-gomlx/onnx/parser"
	"k8s.io/klog/v2"
)

var (
	flagModel    = flag.String("model", kalmgemma3.Repository, fmt.Sprintf("Model repository to use. Examples: %q, %q, %q", kalmgemma3.Repository, bge.Repository, minilmgguf.Repository))
	flagGGUFFile = flag.String("gguf_file", "",
		"Specific .gguf filename to load when -model is a multi-file GGUF repository. "+
			"When empty, selects by community preference order: "+
			"Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q8_0, Q6_K, Q5_1, Q5_0, Q4_1, Q4_0, IQ4_NL, Q3_K_M, Q3_K_L, Q3_K_S, Q2_K, F16, BF16, F32 "+
			"(falling back to the first non-mmproj .gguf file).")
	flagApproxGelu = flag.Bool("approx_gelu", true,
		"Use tanh-approximate GELU (matches GoMLX default and llama.cpp ggml_gelu); set false for exact erf GELU.")
	flagLimit        = flag.Int("limit", -1, "Limit the number of queries indexed. Set <= 0 to use all.")
	flagMSMarcoSplit = flag.String("msmarco_split", msmarco.TrainSplit, "Split to read from MS MARCO dataset (e.g. 'train', 'validation', 'test')")
	flagBucketBudget = flag.Int("bucket", 8*1024,
		"Bucket budget in number of tokens: sentences will be batched in buckets of this number of tokens in total. "+
			"So sentences with 128 tokens, if the budget is 1K, the batchsSize will be 8. "+
			"The buckets use the 'two-bits' algorithm to minimize padding -- e.g.: sizes 8, 12, 16, 24, 32, 48, etc. "+
			"Set <= 0 to embed each sentence individually (with -dynamic=true on dynamic-shape backends, no padding is used).")
	flagParallelEmbedders = flag.Int("num_embedders", 1, "Number of parallel embedders. "+
		"The optimal value depends on the backend and the bucket size (-bucket), for GPUs usually 1 or 2 is enough.")
	flagProfile     = flag.Bool("profile", false, "Enable profiling.")
	flagCPUProf     = flag.String("cpuprof", "", "Write CPU profile to file.")
	flagWarmup      = flag.Bool("warmup", false, "Do a warmup run over the dataset first, compiling the execution graph for each new batch shape encountered.")
	flagShapes      = flag.Bool("shapes", false, "Report the shapes of the batches encountered during the warmup run.")
	flagDynamic     = flag.Bool("dynamic", true, "Use dynamic shapes if supported by the backend.")
	flagTestBackend = flag.String("test_backend", "", "Optional secondary backend config (e.g. 'go' or 'xla:cpu') to compare embeddings against the default backend.")
)

const execMaxCacheSize = 64

func MapHas[K comparable, V any](m map[K]V, k K) bool {
	_, ok := m[k]
	return ok
}

func createModelStore() *model.Store {
	store := model.NewStore()
	store.SetParams(map[string]any{
		layers.ParamQuantizedMatmul: true,
	})
	return store
}

func main() {
	store := createModelStore()
	settings := commandline.CreateSettingsFlag(store, "")
	klog.InitFlags(nil)
	flag.Parse()
	paramsSet, err := commandline.ParseSettings(store, *settings)
	if err != nil {
		klog.Fatalf("Failed to parse -set settings: %+v", err)
	}

	fmt.Printf("- Model %q, bucket %d, parallel embedders %d, dynamic shapes %t, warmup %t, shapes %t\n",
		*flagModel, *flagBucketBudget, *flagParallelEmbedders, *flagDynamic, *flagWarmup, *flagShapes)
	if len(paramsSet) > 0 {
		fmt.Printf("- Modified settings:\n%s\n", commandline.SprintModifiedSettings(store, paramsSet))
	}

	if *flagCPUProf != "" {
		f, err := os.Create(*flagCPUProf)
		if err != nil {
			klog.Fatalf("Could not create CPU profile: %v", err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			klog.Fatalf("Could not start CPU profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	if *flagProfile {
		go func() {
			klog.Info("Starting pprof server on :6060")
			klog.Fatal(http.ListenAndServe("localhost:6060", nil))
		}()
	}

	repo := hub.New(*flagModel)
	if err := repo.DownloadInfo(false); err != nil {
		klog.Fatalf("Failed to load repo info for %q: %v", *flagModel, err)
	}

	hfModel := mustRunWithElapsedTime("Loading model configurations", func() (*transformer.Model, error) {
		m, err := transformer.LoadModel(repo)
		if err != nil {
			return nil, err
		}
		m.WithApproximateGelu(*flagApproxGelu)
		if *flagGGUFFile != "" {
			m.WithGGUFFile(*flagGGUFFile)
		}
		return m, nil
	})
	if hfModel.IsGGUF() {
		fmt.Printf("- GGUF file: %s (approx_gelu=%t)\n", hfModel.GGUFFile(), *flagApproxGelu)
	}

	tokenizer := mustRunWithElapsedTime("Loading tokenizer", func() (tokenizers.Tokenizer, error) {
		return hfModel.GetTokenizer()
	})

	backend := mustRunWithElapsedTime("Initializing backend", func() (compute.Backend, error) {
		return compute.New()
	})
	fmt.Printf("- Backend: %s\n", backend.Description())

	unpadded := *flagBucketBudget <= 0 && *flagDynamic && backend.Capabilities().HasDynamicShapes()
	if unpadded {
		fmt.Printf("- Unpadded single-sentence mode enabled (batchSize=1 static, seqLen dynamic)\n")
	}

	padID := 0
	if id, err := tokenizer.SpecialTokenID(tapi.TokPad); err == nil {
		padID = id
	}

	embedExec := createEmbedExec(backend, store, repo, hfModel, padID, unpadded)

	var testBackend compute.Backend
	var testEmbedExec *model.Exec
	if *flagTestBackend != "" {
		testBackend = mustRunWithElapsedTime(fmt.Sprintf("Initializing test backend (%s)", *flagTestBackend), func() (compute.Backend, error) {
			return compute.NewWithConfig(*flagTestBackend)
		})
		fmt.Printf("- Test Backend: %s\n", testBackend.Description())
		testStore := createModelStore()
		if _, err := commandline.ParseSettings(testStore, *settings); err != nil {
			klog.Fatalf("Failed to parse -set settings for test store: %+v", err)
		}
		testEmbedExec = createEmbedExec(testBackend, testStore, repo, hfModel, padID, unpadded)
	}

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

	if *flagWarmup {
		fmt.Printf("\n--- Warmup Run ---\n")
		runBenchmark(backend, embedExec, testBackend, testEmbedExec, tokenizer, ds, limit, int(totalQueries), unpadded, true)
	}

	fmt.Printf("\n--- Benchmark Run ---\n")
	runBenchmark(backend, embedExec, testBackend, testEmbedExec, tokenizer, ds, limit, int(totalQueries), unpadded, false)
}

func createEmbedExec(backend compute.Backend, store *model.Store, repo *hub.Repo, hfModel *transformer.Model, padID int, unpadded bool) *model.Exec {
	var embedExec *model.Exec
	var err error
	if hfModel.IsGGUF() || repoHasSafetensors(repo) {
		mustRunWithElapsedTime(fmt.Sprintf("Loading variables into store (%s)", backend.Name()), func() (any, error) {
			return nil, hfModel.LoadStore(backend, store)
		})

		embedExec, err = model.NewExec(backend, store, func(scope *model.Scope, tokens *graph.Node) *graph.Node {
			var seqLen *graph.Node
			if !unpadded {
				constPadID := graph.Scalar(tokens.Graph(), tokens.DType(), padID)
				seqLen = graph.ReduceSum(graph.ConvertDType(graph.NotEqual(tokens, constPadID), dtypes.Int32), 1)
			}
			x := hfModel.SentenceEmbeddingGraph(scope, tokens, seqLen)
			return graph.ConvertDType(x, dtypes.Float32)
		})
		if err != nil {
			klog.Fatalf("Failed to create model executor (%s): %v", backend.Name(), err)
		}
	} else if onnxFile, ok := findONNXFile(repo); ok {
		onnxModel := mustRunWithElapsedTime(fmt.Sprintf("Loading ONNX model (%s)", onnxFile), func() (onnx.Model, error) {
			localPath, err := repo.DownloadFile(onnxFile)
			if err != nil {
				return nil, err
			}
			return parser.ParseFile(localPath)
		})
		mustRunWithElapsedTime(fmt.Sprintf("Loading ONNX variables into store (%s)", backend.Name()), func() (any, error) {
			return nil, onnxModel.VariablesToScope(store.RootScope())
		})

		embedExec, err = model.NewExec(backend, store, func(scope *model.Scope, tokens *graph.Node) *graph.Node {
			g := tokens.Graph()
			scope.SetTraining(g, false)
			var notPad, seqLen *graph.Node
			if !unpadded {
				constPadID := graph.Scalar(g, tokens.DType(), padID)
				notPad = graph.NotEqual(tokens, constPadID)
				seqLen = graph.ReduceSum(graph.ConvertDType(notPad, dtypes.Int32), 1)
			}

			inputNames, inputShapes := onnxModel.Inputs()
			onnxInputs := make(map[string]*graph.Node, len(inputNames))
			for i, name := range inputNames {
				dtype := inputShapes[i].DType
				switch name {
				case "input_ids":
					onnxInputs[name] = graph.ConvertDType(tokens, dtype)
				case "attention_mask":
					if unpadded {
						onnxInputs[name] = graph.OnesLike(graph.ConvertDType(tokens, dtype))
					} else {
						onnxInputs[name] = graph.ConvertDType(notPad, dtype)
					}
				case "token_type_ids":
					onnxInputs[name] = graph.ZerosLike(graph.ConvertDType(tokens, dtype))
				case "position_ids":
					onnxInputs[name] = graph.DynamicIota(g, dtype, 1, graph.DimensionSpecFor(tokens, 0), graph.DimensionSpecFor(tokens, 1))
				}
			}
			outputs := onnxModel.CallGraph(scope, g, onnxInputs)
			x := outputs[0]
			if x.Rank() == 3 {
				x = hfModel.SentenceEmbeddingFromLastHiddenState(x, seqLen)
			}
			return graph.ConvertDType(x, dtypes.Float32)
		})
		if err != nil {
			klog.Fatalf("Failed to create ONNX model executor (%s): %v", backend.Name(), err)
		}
	} else {
		klog.Fatalf("No safetensors, GGUF, or ONNX model files found in repository %q", *flagModel)
	}
	if unpadded && !backend.Capabilities().HasDynamicShapes() {
		embedExec.SetMaxCache(-1)
	} else {
		embedExec.SetMaxCache(execMaxCacheSize)
	}
	embedExec.DonateStore()
	if *flagDynamic && backend.Capabilities().HasDynamicShapes() {
		if unpadded {
			embedExec.WithDynamicAxes([]string{"", "seq_len"})
		} else {
			embedExec.WithDynamicAxes([]string{"batch", "seq_len"})
		}
	}
	return embedExec
}

type shapeKey struct {
	batchSize, seqLen int
}

// SpeedStats holds counters for the benchmark speed report.
type SpeedStats struct {
	NumSentencesProcessed    int
	NumTokensProcessed       int64
	NumNonPadTokensProcessed int64
	Elapsed                  time.Duration
	CPUElapsed               time.Duration
	MaxRSSBytes              int64
	IsWarmup                 bool
	UniqueShapes             int
}

// DiffStats holds accumulators for comparing embeddings between two backends.
type DiffStats struct {
	mu sync.Mutex

	NumSentences int64
	NumElements  int64

	// Element-wise differences: |u_i - v_i| and (u_i - v_i)^2
	SumAbsDiff float64
	MaxAbsDiff float64
	SumSqDiff  float64
	MaxSqDiff  float64

	// Per-passage vector differences
	SumL2Dist     float64
	MaxL2Dist     float64
	SumRelL2Dist  float64
	MaxRelL2Dist  float64
	SumCosineDist float64
	MaxCosineDist float64
}

func (d *DiffStats) UpdateBatch(refTensor, testTensor *tensors.Tensor, references []bucket.Reference) {
	var refFlat, testFlat []float32
	refTensor.MustConstFlatData(func(flat any) {
		refFlat = flat.([]float32)
	})
	testTensor.MustConstFlatData(func(flat any) {
		testFlat = flat.([]float32)
	})

	batchSize := len(references)
	if batchSize == 0 || len(refFlat) == 0 || len(refFlat) != len(testFlat) {
		klog.Fatalf("Mismatched output tensor sizes: ref=%d floats, test=%d floats (batchSize=%d)",
			len(refFlat), len(testFlat), batchSize)
	}
	embedDim := len(refFlat) / batchSize

	var (
		localSentences  int64
		localElements   int64
		localSumAbsDiff float64
		localMaxAbsDiff float64
		localSumSqDiff  float64
		localMaxSqDiff  float64
		localSumL2      float64
		localMaxL2      float64
		localSumRelL2   float64
		localMaxRelL2   float64
		localSumCosDist float64
		localMaxCosDist float64
	)

	for rowIdx, ref := range references {
		if ref == nil {
			// Padded row in bucket, skip.
			continue
		}
		localSentences++
		localElements += int64(embedDim)

		offset := rowIdx * embedDim
		u := refFlat[offset : offset+embedDim]
		v := testFlat[offset : offset+embedDim]

		var dotUV, normUSq, normVSq, diffSqSum float64
		for i := range embedDim {
			ui := float64(u[i])
			vi := float64(v[i])
			diff := math.Abs(ui - vi)
			sqDiff := diff * diff

			localSumAbsDiff += diff
			if diff > localMaxAbsDiff {
				localMaxAbsDiff = diff
			}
			localSumSqDiff += sqDiff
			if sqDiff > localMaxSqDiff {
				localMaxSqDiff = sqDiff
			}

			diffSqSum += sqDiff
			dotUV += ui * vi
			normUSq += ui * ui
			normVSq += vi * vi
		}

		l2 := math.Sqrt(diffSqSum)
		localSumL2 += l2
		if l2 > localMaxL2 {
			localMaxL2 = l2
		}

		normU := math.Sqrt(normUSq)
		normV := math.Sqrt(normVSq)
		denomNorm := max(normU, 1e-12)
		relL2 := l2 / denomNorm
		localSumRelL2 += relL2
		if relL2 > localMaxRelL2 {
			localMaxRelL2 = relL2
		}

		var cosDist float64
		if normU > 1e-12 && normV > 1e-12 {
			cosSim := dotUV / (normU * normV)
			// Clamp only above 1.0 to avoid negative cosine distance from float rounding.
			cosDist = max(0.0, 1.0-cosSim)
		}
		localSumCosDist += cosDist
		if cosDist > localMaxCosDist {
			localMaxCosDist = cosDist
		}
	}

	if localSentences == 0 {
		return
	}

	d.mu.Lock()
	defer d.mu.Unlock()
	d.NumSentences += localSentences
	d.NumElements += localElements
	d.SumAbsDiff += localSumAbsDiff
	if localMaxAbsDiff > d.MaxAbsDiff {
		d.MaxAbsDiff = localMaxAbsDiff
	}
	d.SumSqDiff += localSumSqDiff
	if localMaxSqDiff > d.MaxSqDiff {
		d.MaxSqDiff = localMaxSqDiff
	}
	d.SumL2Dist += localSumL2
	if localMaxL2 > d.MaxL2Dist {
		d.MaxL2Dist = localMaxL2
	}
	d.SumRelL2Dist += localSumRelL2
	if localMaxRelL2 > d.MaxRelL2Dist {
		d.MaxRelL2Dist = localMaxRelL2
	}
	d.SumCosineDist += localSumCosDist
	if localMaxCosDist > d.MaxCosineDist {
		d.MaxCosineDist = localMaxCosDist
	}
}

func runBenchmark(backend compute.Backend, embedExec *model.Exec, testBackend compute.Backend, testEmbedExec *model.Exec, tokenizer tokenizers.Tokenizer, ds *datasets.Dataset, limit, totalQueries int, unpadded, isWarmup bool) {
	// Structured concurrency (keep track of goroutines).
	var wg sync.WaitGroup

	// Start bucket runner in a separate goroutine.
	bucketsInputChan := make(chan bucket.SentenceRef, 5)
	bucketsOutputChan := make(chan bucket.Bucket, 10)
	bkt := bucket.New(tokenizer).
		WithMaxParallelization(-1)
	if unpadded {
		bkt.Unpadded()
	} else {
		bkt.ByTwoBitBucketBudget(*flagBucketBudget, 8).
			WithBatchPadding(true)
	}

	wg.Go(func() {
		bkt.Run(bucketsInputChan, bucketsOutputChan)
	})

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
	startTime, startCPUTime := time.Now(), cpuTime()
	var speedStats SpeedStats
	speedStats.IsWarmup = isWarmup
	var diffStats DiffStats

	expectedNumQueries := limit
	if expectedNumQueries <= 0 {
		expectedNumQueries = totalQueries
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
			speedStats.NumSentencesProcessed += count

			// Report progress every second.
			if time.Since(lastReportTime) > time.Second {
				lastReportTime = time.Now()

				// ETA estimation.
				sentencesPerSecond = float64(speedStats.NumSentencesProcessed) / time.Since(startTime).Seconds()
				eta := "Unknown"
				expectedTotalSentences := 10 * expectedNumQueries
				if speedStats.NumSentencesProcessed > 0 {
					remainingSeconds := float64(expectedTotalSentences-speedStats.NumSentencesProcessed) / sentencesPerSecond
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
					humanize.Count(int64(speedStats.NumSentencesProcessed)), humanize.Count(int64(expectedTotalSentences)), humanize.Speed(sentencesPerSecond, "passages"),
					humanize.Speed(speed, "tokens"), eta, humanize.EraseToEndOfLine)
			}
		}
		expectedTotalSentences := 10 * expectedNumQueries
		emaMu.RLock()
		speed := emaSpeed
		emaMu.RUnlock()
		fmt.Printf("\r  ✅ Processed %s / %s passages (%s, %s non-padding) -- done.%s\n",
			humanize.Count(int64(speedStats.NumSentencesProcessed)), humanize.Count(int64(expectedTotalSentences)), humanize.Speed(sentencesPerSecond, "passages"),
			humanize.Speed(speed, "tokens"), humanize.EraseToEndOfLine)
	})

	var seenShapes sync.Map
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

				if isWarmup {
					key := shapeKey{batchSize, seqLen}
					if _, loaded := seenShapes.LoadOrStore(key, true); loaded {
						// Already seen, discard batch
						count := 0
						for _, ref := range bk.References {
							if ref != nil {
								count++
							}
						}
						numSentencesProcessedChan <- count
						continue
					}
					if *flagShapes {
						fmt.Printf("\n  - Shape [%d, %d], %d total tokens\n", batchSize, seqLen, len(bk.Batch))
					}
				}

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
				batchShape := shapes.Make(dtype, batchSize, seqLen)
				inputTensor, err := tensors.FromRaw(backend, 0, batchShape, rawData)
				if err != nil {
					klog.Fatalf("Failed to create input tensor: %+v", err)
				}

				batchStartTime := time.Now()
				var outTensor *tensors.Tensor
				outTensor, err = embedExec.Call1(inputTensor)
				if err != nil {
					fmt.Println()
					klog.Fatalf("Failed to execute embeddings for %s: %+v", inputTensor.Shape(), err)
				}
				if testEmbedExec != nil {
					testInputTensor, err := tensors.FromRaw(testBackend, 0, batchShape, rawData)
					if err != nil {
						klog.Fatalf("Failed to create test input tensor: %+v", err)
					}
					testOutTensor, err := testEmbedExec.Call1(testInputTensor)
					if err != nil {
						fmt.Println()
						klog.Fatalf("Failed to execute test embeddings for %s: %+v", testInputTensor.Shape(), err)
					}
					diffStats.UpdateBatch(outTensor, testOutTensor, bk.References)
					testInputTensor.FinalizeAll()
					testOutTensor.FinalizeAll()
				}

				atomic.AddInt64(&speedStats.NumTokensProcessed, int64(len(bk.Batch)))
				atomic.AddInt64(&speedStats.NumNonPadTokensProcessed, int64(bk.NonPadTokens))

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
	speedStats.Elapsed = time.Since(startTime)
	speedStats.CPUElapsed = cpuTime() - startCPUTime
	speedStats.MaxRSSBytes = maxRSSBytes()
	if isWarmup {
		speedStats.UniqueShapes = mapLen(&seenShapes)
	}
	fmt.Printf("Total duration: %v\n", humanize.Duration(speedStats.Elapsed))

	if testEmbedExec != nil {
		ReportDifferences(&diffStats)
	} else {
		ReportSpeed(&speedStats)
	}
}

// ReportSpeed prints a summary table with counts and processing speeds.
func ReportSpeed(s *SpeedStats) {
	cpuTimePerPassage := "N/A"
	if s.NumSentencesProcessed > 0 {
		cpuTimePerPassage = fmt.Sprintf("%s/passage", humanize.Duration(s.CPUElapsed/time.Duration(s.NumSentencesProcessed)))
	}
	names := []string{"Passages", "Tokens", "Non-Pad Tokens", "CPU Time", "~Cores", "Max RSS Mem"}
	totals := []string{
		humanize.Count(int64(s.NumSentencesProcessed)),
		humanize.Count(s.NumTokensProcessed),
		humanize.Count(s.NumNonPadTokensProcessed),
		humanize.Duration(s.CPUElapsed),
		fmt.Sprintf("%.1f", s.CPUElapsed.Seconds()/s.Elapsed.Seconds()),
		humanize.Bytes(s.MaxRSSBytes),
	}
	speeds := []string{
		humanize.Speed(float64(s.NumSentencesProcessed)/s.Elapsed.Seconds(), " items"),
		humanize.Speed(float64(s.NumTokensProcessed)/s.Elapsed.Seconds(), "tokens"),
		humanize.Speed(float64(s.NumNonPadTokensProcessed)/s.Elapsed.Seconds(), "tokens"),
		cpuTimePerPassage,
		"N/A",
		"N/A",
	}
	if s.IsWarmup {
		names = append(names, "Unique Shapes")
		totals = append(totals, humanize.Count(s.UniqueShapes))
		speeds = append(speeds, "N/A")
	}
	baseStyle := lipgloss.NewStyle().Padding(0, 1)
	t := table.New().
		Border(lipgloss.NormalBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(lipgloss.Color("238"))).
		Headers("Metric", "Total", "Speed").
		StyleFunc(func(row, col int) lipgloss.Style {
			st := baseStyle
			if col > 0 && row != table.HeaderRow {
				st = st.Align(lipgloss.Right)
			}
			if row == table.HeaderRow {
				headerStyle := st.Foreground(lipgloss.Color("252")).Bold(true)
				if col > 0 {
					headerStyle = headerStyle.Align(lipgloss.Center)
				}
				return headerStyle
			}
			return st
		})
	for i := range names {
		t.Row(names[i], totals[i], speeds[i])
	}
	fmt.Println(t)
}

// ReportDifferences prints a summary table comparing embeddings between the reference and test backends.
func ReportDifferences(d *DiffStats) {
	if d.NumSentences == 0 || d.NumElements == 0 {
		fmt.Println("No passages processed to compare.")
		return
	}

	meanAbsDiff := d.SumAbsDiff / float64(d.NumElements)
	rmse := math.Sqrt(d.SumSqDiff / float64(d.NumElements))
	meanL2 := d.SumL2Dist / float64(d.NumSentences)
	meanRelL2 := d.SumRelL2Dist / float64(d.NumSentences)
	meanCosDist := d.SumCosineDist / float64(d.NumSentences)

	metrics := []string{
		"Element-wise Abs Diff (|u_i - v_i|)",
		"Element-wise Sq Diff (RMSE)",
		"Passage Euclidean Distance (L2)",
		"Passage Relative L2 (||u - v||_2 / ||u||_2)",
		"Passage Cosine Distance (1 - cos(u, v))",
	}
	means := []string{
		fmt.Sprintf("%.4e", meanAbsDiff),
		fmt.Sprintf("%.4e", rmse),
		fmt.Sprintf("%.4e", meanL2),
		fmt.Sprintf("%.4e", meanRelL2),
		fmt.Sprintf("%.4e", meanCosDist),
	}
	maxes := []string{
		fmt.Sprintf("%.4e", d.MaxAbsDiff),
		fmt.Sprintf("%.4e", math.Sqrt(d.MaxSqDiff)),
		fmt.Sprintf("%.4e", d.MaxL2Dist),
		fmt.Sprintf("%.4e", d.MaxRelL2Dist),
		fmt.Sprintf("%.4e", d.MaxCosineDist),
	}

	baseStyle := lipgloss.NewStyle().Padding(0, 1)
	t := table.New().
		Border(lipgloss.NormalBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(lipgloss.Color("238"))).
		Headers("Metric", "Mean", "Max").
		StyleFunc(func(row, col int) lipgloss.Style {
			st := baseStyle
			if col > 0 && row != table.HeaderRow {
				st = st.Align(lipgloss.Right)
			}
			if row == table.HeaderRow {
				headerStyle := st.Foreground(lipgloss.Color("252")).Bold(true)
				if col > 0 {
					headerStyle = headerStyle.Align(lipgloss.Center)
				}
				return headerStyle
			}
			return st
		})
	for i := range metrics {
		t.Row(metrics[i], means[i], maxes[i])
	}
	fmt.Println(t)
}

func cpuTime() time.Duration {
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}

func maxRSSBytes() int64 {
	var ru syscall.Rusage
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	if runtime.GOOS == "darwin" {
		return int64(ru.Maxrss) // bytes on macOS
	}
	return int64(ru.Maxrss) * 1024 // KiB on Linux/BSD
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

func mapLen(m *sync.Map) int {
	length := 0
	m.Range(func(_, _ any) bool {
		length++
		return true
	})
	return length
}

func repoHasSafetensors(repo *hub.Repo) bool {
	for filename, err := range repo.IterFileNames() {
		if err != nil {
			return false
		}
		if filepath.Ext(filename) == ".safetensors" ||
			filename == "model.safetensors.index.json" ||
			filename == "pytorch_model.safetensors.index.json" ||
			filepath.Base(filename) == "model.safetensors.index.json" ||
			filepath.Base(filename) == "pytorch_model.safetensors.index.json" {
			return true
		}
	}
	return false
}

func findONNXFile(repo *hub.Repo) (string, bool) {
	for _, preferred := range []string{"model.onnx", "onnx/model.onnx"} {
		if repo.HasFile(preferred) {
			return preferred, true
		}
	}
	for filename, err := range repo.IterFileNames() {
		if err != nil {
			return "", false
		}
		if filepath.Ext(filename) == ".onnx" {
			return filename, true
		}
	}
	return "", false
}
