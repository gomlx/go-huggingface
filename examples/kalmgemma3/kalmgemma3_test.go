package kalmgemma3

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"
	"unsafe"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/internal/humanize"
	"github.com/gomlx/go-huggingface/models/safetensors"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
)

var (
	flagUseCausalMask = flag.Bool("use_causal_mask", true, "Use causal mask in the transformer: the paper suggests one shouldn't, "+
		"but for testing it makes the result closer to Python's using HF transformer library, which seems to use it.")
	flagListPrompts        = flag.Bool("prompts", false, "During initialization lists prompts from the dataset and exit immediately.")
	flagSkipLoadingWeights = flag.Bool("skip_loading_weights", false, "Skip loading weights from the model.")
)

var (
	testBackend backends.Backend
	testRepo    *hub.Repo
	testCtx     *context.Context
	testModel   *transformer.Model
	taskPrompts TaskPrompts
	testQueries []string
	testDocs    []string
	testPadID   int32
)

func must(err error) {
	if err != nil {
		klog.Errorf("Must failed: %+v", err)
		panic(err)
	}
}

func must1[T any](v T, err error) T {
	must(err)
	return v
}

func TestMain(m *testing.M) {
	klog.InitFlags(nil)
	flag.Parse() // Ensure flags are parsed before we use them
	if testing.Short() {
		fmt.Printf("🚨 Skipping tests because -short flag is set, and these tests are very expensive.\n")
		os.Exit(0)
	}

	var err error
	testBackend, err = backends.New()
	if err != nil {
		fmt.Printf("Failed to initialize backend: %v\n", err)
		os.Exit(1)
	}

	testCtx = context.New().Checked(false)
	testRepo, err = LoadRepo()
	if err != nil {
		fmt.Printf("Failed to LoadRepo: %v\n", err)
		os.Exit(1)
	}

	testModel, err = transformer.LoadModel(testRepo)
	if err != nil {
		fmt.Printf("Failed to LoadModel: %v\n", err)
		os.Exit(1)
	}
	testModel = testModel.WithCausalMask(*flagUseCausalMask)
	if *flagListPrompts {
		fmt.Printf("Prompts:\n")
		for _, taskCode := range testModel.RegisteredPromptTasks() {
			prompt := testModel.GetTaskPrompt(taskCode)
			fmt.Printf("  [%s]:\n    %q\n\n", taskCode, prompt)
		}
		os.Exit(0)
	}

	fmt.Printf("✅ Model: %s", testModel.Description())

	tokenizer := must1(testModel.GetTokenizer())
	padID, err := tokenizer.SpecialTokenID(api.TokPad)
	if err != nil {
		fmt.Printf("Padding token not defined: %+v\n", err)
		os.Exit(1)
	}
	testPadID = int32(padID)

	taskPrompts = must1(LoadTaskPrompts(testRepo))
	fmt.Printf("✅ Task prompts loaded: %d tasks\n", len(taskPrompts))

	testQueries = []string{
		taskPrompts.BuildQueryPrompt("What is the capital of China?", ""),
		taskPrompts.BuildQueryPrompt("Explain gravity", ""),
	}
	testDocs = []string{
		"The capital of China is Beijing.",
		"Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
	}

	if !*flagSkipLoadingWeights {
		fmt.Printf(" - Loading model weights ...\r")
		start := time.Now()
		must(testModel.LoadContext(testBackend, testCtx))
		for range 3 {
			runtime.GC()
		}
		fmt.Printf("\r✅ Loading model weights: done (%v)\n", time.Since(start))
	}

	// Run the tests
	code := m.Run()

	testBackend.Finalize()
	os.Exit(code)
}

// readPythonEmbeddings reads the embeddings and layers dumped by the python script.
func readPythonEmbeddings(path string, layersToRead []int) (map[int][]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	wantLayers := make(map[int]bool)
	for _, l := range layersToRead {
		wantLayers[l] = true
	}

	results := make(map[int][]float32)
	scanner := bufio.NewScanner(file)
	currentLayer := -1
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "#") {
			if strings.HasPrefix(line, "# Token Embeddings") {
				currentLayer = 0
			} else if strings.HasPrefix(line, "# Layer ") {
				var l int
				if _, err := fmt.Sscanf(line, "# Layer %d Output", &l); err == nil {
					currentLayer = l
				} else {
					currentLayer = -1
				}
			} else {
				currentLayer = -1
			}
			continue
		}
		if currentLayer == -1 || !wantLayers[currentLayer] {
			continue
		}

		val, err := strconv.ParseFloat(line, 32)
		if err != nil {
			return nil, fmt.Errorf("failed to parse float: %v", err)
		}

		results[currentLayer] = append(results[currentLayer], float32(val))
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return results, nil
}

func TestTransformerLayers(t *testing.T) {
	// Ensure the weights were loaded.
	varName := "embeddings"
	if testCtx.In("token_embed").InspectVariableInScope(varName) == nil {
		t.Fatalf("Variable token_embed/%s not loaded in context", varName)
	}

	layersToCheck := []int{0, 1, 2, 10, 20, 30, 40, testModel.Config.NumHiddenLayers}

	uniqueLayers := make(map[int]bool)
	var finalLayersToCheck []int
	for _, l := range layersToCheck {
		if l <= testModel.Config.NumHiddenLayers && !uniqueLayers[l] {
			uniqueLayers[l] = true
			finalLayersToCheck = append(finalLayersToCheck, l)
		}
	}

	exec, err := context.NewExec(testBackend, testCtx.Reuse(), func(ctx *context.Context, tokens *graph.Node) []*graph.Node {
		_, allLayers := testModel.AllLayers(ctx, tokens, nil)
		var converted []*graph.Node
		for _, o := range allLayers {
			converted = append(converted, graph.ConvertDType(o, dtypes.Float32))
		}
		return converted
	})
	if err != nil {
		t.Fatalf("Failed to create exec: %v", err)
	}

	cases := []struct {
		name     string
		prompt   string
		fileName string
	}{
		{"Query 1", testQueries[0], "layer_emb_q1.txt"},
		{"Query 2", testQueries[1], "layer_emb_q2.txt"},
		{"Doc 1", testDocs[0], "layer_emb_d1.txt"},
		{"Doc 2", testDocs[1], "layer_emb_d2.txt"},
	}

	tokenizer := must1(testModel.GetTokenizer())
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			expectedLayers, err := readPythonEmbeddings(tc.fileName, finalLayersToCheck)
			if err != nil {
				t.Skipf("Skipping test because %s is not available: %v", tc.fileName, err)
			}
			for _, l := range finalLayersToCheck {
				if len(expectedLayers[l]) == 0 {
					t.Fatalf("Failed to read expected data for layer %d from %s", l, tc.fileName)
				}
			}

			tokens := tokenizer.Encode(tc.prompt)
			testSubTitle := []string{"no-mask", "with-mask"}
			for maskIdx, withMask := range []bool{false, true} {
				t.Run(testSubTitle[maskIdx], func(t *testing.T) {
					// Create inputTensor with extra padding if mask is selected.
					sentenceLen := len(tokens)
					var inputTensor *tensors.Tensor
					if withMask {
						sentenceLen += 10 // Extra padding.
					}
					inputTensor = tensors.FromShape(shapes.Make(dtypes.Int32, 1, sentenceLen))
					tensors.MustMutableFlatData(inputTensor, func(flat []int32) {
						for i, token := range tokens {
							flat[i] = int32(token)
						}
						if withMask && testPadID != 0 {
							for i := len(tokens); i < sentenceLen; i++ {
								flat[i] = testPadID
							}
						}
					})

					// Execute the embedder.
					fmt.Printf("- Executing model for %s ...", tc.name)
					start := time.Now()
					gotAllLayers, err := exec.Exec(inputTensor)
					if err != nil {
						t.Fatalf("Failed to execute graph: %v", err)
					}
					fmt.Printf("done (%v)\n", time.Since(start))
					if len(gotAllLayers) < testModel.Config.NumHiddenLayers+1 {
						t.Fatalf("Expected at least %d outputs from graph, got %d", testModel.Config.NumHiddenLayers+1, len(gotAllLayers))
					}

					for _, l := range finalLayersToCheck {
						gotLayer := gotAllLayers[l]
						expectedFlat := expectedLayers[l]
						name := fmt.Sprintf("Layer %d Output", l)
						if l == 0 {
							name = "Token Embeddings"
						}

						// Verify shape of layer.
						gotShape := gotLayer.Shape()
						if gotShape.Rank() < 3 {
							t.Fatalf("[%s] Expected rank >= 3, got %s", name, gotShape)
						}
						if gotShape.Dimensions[0] != 1 {
							t.Fatalf("[%s] Expected batch size 1, got %d -- output shape is %s", name, gotShape.Dimensions[0], gotShape)
						}
						gotSentenceLen := gotShape.Dimensions[1]
						if gotSentenceLen != sentenceLen {
							t.Fatalf("[%s] Expected %d tokens in output, got %d -- output shape is %s", name, sentenceLen, gotSentenceLen, gotShape)
						}
						if gotShape.Dimensions[2] != testModel.Config.HiddenSize {
							t.Fatalf("[%s] Expected hidden size %d, got %d -- output shape is %s", name, testModel.Config.HiddenSize, gotShape.Dimensions[2], gotShape)
						}

						// Verify values of the first len(tokens) only, we don't need to check the padding.
						gotLayer.ConstFlatData(func(flatAny any) {
							gotFlat := flatAny.([]float32)
							validateTensor(t, gotFlat, expectedFlat, name, 1, sentenceLen, len(tokens))
						})
					}
				})
			}
		})
	}
}

// validateTensor validates the values of a tensor.
// It checks the values of the first expectedSentenceLen tokens only, ignoring the padding in
// case gotSentencenLen > expectedSentenceLen.
func validateTensor(t *testing.T, got []float32, expected []float32, name string,
	batchSize, gotSentenceLen, expectedSentenceLen int) {

	gotHiddenDim := len(got) / batchSize / gotSentenceLen
	expectedHiddenDim := len(expected) / batchSize / expectedSentenceLen
	if gotHiddenDim == 0 || gotHiddenDim != expectedHiddenDim || gotSentenceLen < expectedSentenceLen {
		t.Fatalf("[%s] Shape mismatch: expected %d flat floats ([%d, %d, %d]?), got %d ([%d, %d, %d])",
			name, len(expected), batchSize, expectedSentenceLen, expectedHiddenDim,
			len(got), batchSize, gotSentenceLen, gotHiddenDim)
	}

	// Find mapping from expected indices to got indices.
	expectedShape := shapes.Make(dtypes.Float32, batchSize, expectedSentenceLen, expectedHiddenDim)
	gotShape := shapes.Make(dtypes.Float32, batchSize, gotSentenceLen, gotHiddenDim)
	gotStrides := gotShape.Strides()
	gotFlatIdxFn := func(indices []int) int {
		flatIdx := 0
		for i, idx := range indices {
			flatIdx += idx * gotStrides[i]
		}
		return flatIdx
	}

	var sumAbsDiff, sumAbsExpected float64
	const minRelDenominator = 0.2
	for flatIdx, expectedIndices := range expectedShape.Iter() {
		expectValue := float64(expected[flatIdx])
		gotValueF64 := float64(got[gotFlatIdxFn(expectedIndices)])
		absDiff := math.Abs(gotValueF64 - expectValue)
		sumAbsDiff += absDiff
		sumAbsExpected += math.Abs(expectValue)
	}
	meanAbsDiff := sumAbsDiff / float64(len(got))
	meanAbsExpected := sumAbsExpected / float64(len(got))

	var maxRelDiff float64
	var maxRelDiffIdx int
	for flatIdx, expectedIndices := range expectedShape.Iter() {
		expectValue := float64(expected[flatIdx])
		gotValueF64 := float64(got[gotFlatIdxFn(expectedIndices)])
		absDiff := math.Abs(gotValueF64 - expectValue)
		relDenominator := math.Max(math.Abs(expectValue), math.Abs(gotValueF64))
		relDenominator = max(relDenominator, meanAbsExpected)
		relDiff := absDiff / relDenominator
		if relDiff > maxRelDiff {
			maxRelDiff = relDiff
			maxRelDiffIdx = flatIdx
		}
	}

	maxRelTolerance := 5.0
	meanTolerance := 0.1

	if maxRelDiff > maxRelTolerance || meanAbsDiff >= meanTolerance*meanAbsExpected {
		t.Errorf("[%s] Mismatch in values: Max rel diff: %.3g at idx %d (ex %f, got %f) / "+
			"Mean abs diff: %.3g (== %.1f%% of the mean absolute values %.3g)",
			name, maxRelDiff, maxRelDiffIdx, expected[maxRelDiffIdx], got[maxRelDiffIdx],
			meanAbsDiff, 100*meanAbsDiff/meanAbsExpected, meanAbsExpected)
		for flatIdx, expectedIndices := range expectedShape.Iter() {
			expectValue := float64(expected[flatIdx])
			gotValueF64 := float64(got[gotFlatIdxFn(expectedIndices)])
			if flatIdx > 10 && flatIdx < len(expected)-10 {
				continue
			}
			fmt.Printf("\t- Value #%d:\tgot %.3g,\t expected %.3g\n", flatIdx, gotValueF64, expectValue)
		}
	} else {
		t.Logf("[%s] Match! Max rel diff: %.3g, Mean abs diff: %.3g (== %.1f%% of %.3g is the mean absolute)",
			name, maxRelDiff, meanAbsDiff, 100*meanAbsDiff/meanAbsExpected, meanAbsExpected)
	}
}

func readPythonEmbeddingsList(path string) ([]float32, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var results []float32
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		val, err := strconv.ParseFloat(line, 32)
		if err != nil {
			return nil, fmt.Errorf("failed to parse float: %v", err)
		}
		results = append(results, float32(val))
	}
	return results, scanner.Err()
}

func TestSentenceEmbedding(t *testing.T) {
	prompts := make([]string, 0, len(testQueries)+len(testDocs))
	prompts = append(prompts, testQueries...)
	prompts = append(prompts, testDocs...)

	pythonPath := "similarity_embeddings.txt"
	expectedFlatData, err := readPythonEmbeddingsList(pythonPath)
	if err != nil {
		t.Skipf("Skipping test because %s is not available: %v", pythonPath, err)
	}

	exec, err := context.NewExec(testBackend, testCtx.Checked(false), func(ctx *context.Context, tokens *graph.Node) *graph.Node {
		x := testModel.SentenceEmbeddingGraph(ctx, tokens, nil)
		return graph.ConvertDType(x, dtypes.Float32)
	})
	if err != nil {
		t.Fatalf("Failed to create exec: %v", err)
	}

	hiddenSize := testModel.Config.HiddenSize
	if len(expectedFlatData) != len(prompts)*hiddenSize {
		t.Fatalf("Expected %d flat floats from python embeddings, got %d", len(prompts)*hiddenSize, len(expectedFlatData))
	}

	for i, prompt := range prompts {
		tokensInt := must1(testModel.GetTokenizer()).Encode(prompt)
		tokens := make([]int32, len(tokensInt))
		for j, t := range tokensInt {
			tokens[j] = int32(t)
		}
		inputTensor := tensors.FromValue([][]int32{tokens})
		results, err := exec.Exec(inputTensor)
		if err != nil {
			t.Fatalf("Failed to execute graph for prompt %d: %v", i, err)
		}

		outTensor := results[0]
		outShape := outTensor.Shape()
		if outShape.Rank() != 2 || outShape.Dimensions[1] != hiddenSize {
			t.Fatalf("Expected shape [batch, hidden_size] ([1, %d]), got %s", hiddenSize, outShape)
		}

		outTensor.ConstFlatData(func(flatAny any) {
			flat := flatAny.([]float32)
			expectedData := expectedFlatData[i*hiddenSize : (i+1)*hiddenSize]
			validateTensor(t, flat, expectedData, fmt.Sprintf("Sentence Embedding %d", i),
				1, len(tokens), len(tokens))
		})
	}
}

func TestSentenceBatchEmbedding(t *testing.T) {
	prompts := make([]string, 0, len(testQueries)+len(testDocs))
	prompts = append(prompts, testQueries...)
	prompts = append(prompts, testDocs...)

	pythonPath := "similarity_embeddings.txt"
	expectedFlatData, err := readPythonEmbeddingsList(pythonPath)
	if err != nil {
		t.Skipf("Skipping test because %s is not available: %v", pythonPath, err)
	}

	exec, err := context.NewExec(testBackend, testCtx.Checked(false), func(ctx *context.Context, tokens *graph.Node) *graph.Node {
		mask := graph.NotEqual(tokens, graph.Const(tokens.Graph(), testPadID))
		x := testModel.SentenceEmbeddingGraph(ctx, tokens, mask)
		return graph.ConvertDType(x, dtypes.Float32)
	})
	if err != nil {
		t.Fatalf("Failed to create exec: %v", err)
	}

	// Tokenize all sentences.
	tokenizer := must1(testModel.GetTokenizer())
	var batchTokens [][]int32
	var maxLen int
	for _, prompt := range prompts {
		tokensInt := tokenizer.Encode(prompt)
		tokens := xslices.Map(tokensInt, func(t int) int32 { return int32(t) })
		batchTokens = append(batchTokens, tokens)
		if len(tokens) > maxLen {
			maxLen = len(tokens)
		}
	}

	// Find the paddedLen and create the flat batch with the tokens+padding.
	paddedLen := int(math.Pow(2, math.Ceil(math.Log2(float64(maxLen)))))
	var batchFlat []int32
	totalBatchSize := len(prompts) * paddedLen
	if testPadID == 0 {
		batchFlat = make([]int32, totalBatchSize)
	} else {
		batchFlat = xslices.SliceWithValue(totalBatchSize, testPadID)
	}
	for i, tokens := range batchTokens {
		offset := i * paddedLen
		copy(batchFlat[offset:offset+len(tokens)], tokens)
	}
	batch := tensors.FromFlatDataAndDimensions(batchFlat, len(prompts), paddedLen)
	batch.ToDevice(testBackend, 0)

	fmt.Printf("- Pre-compiling model for batch ...")
	start := time.Now()
	err = exec.PreCompile(batch)
	if err != nil {
		t.Fatalf("Failed to compile graph: %+v", err)
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	fmt.Printf("- Executing model ...")
	hiddenSize := testModel.Config.HiddenSize
	if len(expectedFlatData) != len(prompts)*hiddenSize {
		t.Fatalf("Expected %d flat floats from python embeddings, got %d", len(prompts)*hiddenSize, len(expectedFlatData))
	}
	start = time.Now()
	results, err := exec.Exec(batch)
	fmt.Printf("done (%v)\n", time.Since(start))
	if err != nil {
		t.Fatalf("Failed to execute graph for batched prompts: %v", err)
	}

	outTensor := results[0]
	outShape := outTensor.Shape()
	if outShape.Rank() != 2 || outShape.Dimensions[0] != len(prompts) || outShape.Dimensions[1] != hiddenSize {
		t.Fatalf("Expected shape [batch, hidden_size] ([%d, %d]), got %s", len(prompts), hiddenSize, outShape)
	}

	outTensor.ConstFlatData(func(flatAny any) {
		flat := flatAny.([]float32)
		for i := range prompts {
			gotData := flat[i*hiddenSize : (i+1)*hiddenSize]
			expectedData := expectedFlatData[i*hiddenSize : (i+1)*hiddenSize]
			validateTensor(t, gotData, expectedData, fmt.Sprintf("Sentence Batch Embedding %d", i),
				1, 1, 1)
		}
	})
}

func TestSimilarity(t *testing.T) {
	fmt.Printf("Queries:   %q\n", testQueries)
	fmt.Printf("Documents: %q\n", testDocs)

	prompts := make([]string, 0, len(testQueries)+len(testDocs))
	prompts = append(prompts, testQueries...)
	prompts = append(prompts, testDocs...)
	allEmbeddings := make([]*tensors.Tensor, 0, len(prompts))
	embedder := must1(testModel.SingleSentenceEmbeddingExec(testBackend, testCtx))
	for _, prompt := range prompts {
		tokens := must1(testModel.GetTokenizer()).Encode(prompt)
		allEmbeddings = append(allEmbeddings, must1(embedder.Exec1(tokens)))
	}

	allEmbeddingsAny := xslices.Map(allEmbeddings, func(t *tensors.Tensor) any { return t })
	similarities := must1(graph.ExecOnce(testBackend, func(allEmbeddings []*graph.Node) *graph.Node {
		queryEmbeddings := graph.Stack(allEmbeddings[:len(testQueries)], 0)
		docEmbeddings := graph.Stack(allEmbeddings[len(testQueries):], 0)
		return testModel.Similarity(queryEmbeddings, docEmbeddings)
	}, allEmbeddingsAny...))
	fmt.Printf("Similarities: %v\n", similarities)
	want := []float32{0.9316, 0.3984, 0.4251, 0.7317}
	fmt.Printf("- Expected: %v\n", want)
	got := tensors.MustCopyFlatData[float32](similarities)
	require.InDeltaSlicef(t, want, got, 1e-2, "Similaries don't match!")
}

// TestReadAllShards simply read all the shard files into /dev/null, used only
// to test the speed.
func TestReadAllShards(t *testing.T) {
	var buf [1 << 20]byte
	var f *os.File
	defer func() {
		if f != nil {
			f.Close()
		}
	}()

	fmt.Printf("- Reading all shards ...")
	start := time.Now()
	for filename, err := range testRepo.IterFileNames() {
		if err != nil {
			t.Fatalf("Failed to iterate over file names: %v", err)
		}
		if !strings.HasSuffix(filename, ".safetensors") {
			continue
		}
		localPath := must1(testRepo.DownloadFile(filename))
		f = must1(os.Open(localPath))
		for {
			_, err = f.Read(buf[:])
			if err != nil {
				if err == io.EOF {
					break
				}
				must(err)
			}
		}

		f.Close()
		f = nil
	}
	fmt.Printf("done (%v)\n", time.Since(start))
}

func TestUploadSafetensors(t *testing.T) {
	model := safetensors.NewEmpty(testRepo)
	maxSize := int64(0)
	uniqueSizes := sets.Make[int64]()
	var allShapes []shapes.Shape
	start := time.Now()
	for fileInfo, err := range model.IterSafetensors() {
		require.NoError(t, err)
		// Sort tensor names for deterministic output
		tensorNames := xslices.SortedKeys(fileInfo.Header.Tensors)
		for _, name := range tensorNames {
			meta := fileInfo.Header.Tensors[name]
			shape, err := meta.GoMLXShape()
			require.NoError(t, err)
			size := meta.DataOffsets[1] - meta.DataOffsets[0]
			if int64(shape.Memory()) != size {
				t.Fatalf("Tensor %s has shape %v and size %s, but offset size is %s", name, shape,
					humanize.Bytes(int64(shape.Memory())),
					humanize.Bytes(int64(size)))
			}
			klog.V(1).Infof(" - %s: shape=%v, size=%s\n", name, shape, humanize.Bytes(int64(shape.Memory())))
			maxSize = max(maxSize, size)
			uniqueSizes.Insert(size)
			allShapes = append(allShapes, shape)
		}
	}
	fmt.Printf("Max size: %s\n", humanize.Bytes(int64(maxSize)))
	fmt.Printf("Unique sizes: %s\n", xslices.Map(xslices.Keys(uniqueSizes),
		func(s int64) string {
			return humanize.Bytes(int64(s))
		}))
	byteBuf := make([]byte, maxSize)
	bytesPtr := unsafe.Pointer(&byteBuf[0])
	allBuffers := make([]backends.Buffer, 0, len(allShapes))

	for _, shape := range allShapes {
		length := shape.Size() / shape.DType.ValuesPerStorageUnit()
		flatAny := dtypes.UnsafeAnySliceFromBytes(bytesPtr, shape.DType, length)
		b, err := testBackend.BufferFromFlatData(0, flatAny, shape)
		require.NoError(t, err)
		allBuffers = append(allBuffers, b)
	}
	fmt.Printf("Buffers created in %v\n", time.Since(start))

	start = time.Now()
	for _, buf := range allBuffers {
		err := testBackend.BufferFinalize(buf)
		require.NoError(t, err)
	}
	fmt.Printf("Buffers finalized in %v\n", time.Since(start))
}

func TestIterTensorsFromRepo(t *testing.T) {
	start := time.Now()
	var allTensors []safetensors.TensorAndName
	for tan, err := range safetensors.IterTensorsFromRepo(testBackend, testRepo) {
		require.NoError(t, err)
		allTensors = append(allTensors, tan)
	}
	fmt.Printf("- %d tensors loaded in %v\n", len(allTensors), time.Since(start))

	start = time.Now()
	for _, tan := range allTensors {
		err := tan.Tensor.FinalizeAll()
		require.NoError(t, err)
	}
	fmt.Printf("- %d tensors finalized in %v\n", len(allTensors), time.Since(start))
}
