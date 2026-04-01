package kalmgemma3

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"
)

var (
	flagUseCausalMask = flag.Bool("use_causal_mask", true, "Use causal mask in the transformer: the paper suggests one shouldn't, "+
		"but for testing it makes the result closer to Python's using HF transformer library, which seems to use it.")
	flagListPrompts = flag.Bool("prompts", false, "During initialization lists prompts from the dataset and exit immediately.")
)

var (
	testBackend backends.Backend
	testCtx     *context.Context
	testModel   *transformer.Model
	taskPrompts TaskPrompts
	testQueries []string
	testDocs    []string
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
	repo, err := LoadRepo()
	if err != nil {
		fmt.Printf("Failed to LoadRepo: %v\n", err)
		os.Exit(1)
	}

	testModel, err = transformer.LoadModel(repo)
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

	fmt.Printf("✅ Model: %s\n\n", testModel.Description())

	taskPrompts = must1(LoadTaskPrompts(repo))
	fmt.Printf("✅ Task prompts loaded: %d tasks\n", len(taskPrompts))

	testQueries = []string{
		taskPrompts.BuildQueryPrompt("What is the capital of China?", ""),
		taskPrompts.BuildQueryPrompt("Explain gravity", ""),
	}
	testDocs = []string{
		"The capital of China is Beijing.",
		"Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
	}

	fmt.Printf("- Loading model weights ...")
	start := time.Now()
	must(testModel.LoadContext(testBackend, testCtx))
	for range 3 {
		runtime.GC()
	}
	fmt.Printf("\r✅ Loading model weights: done (%v)\n", time.Since(start))

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

			tokensInt := must1(testModel.GetTokenizer()).Encode(tc.prompt)
			tokens := make([]int32, len(tokensInt))
			for j, tIdx := range tokensInt {
				tokens[j] = int32(tIdx)
			}
			inputTensor := tensors.FromValue([][]int32{tokens})

			fmt.Printf("- Executing model for %s ...", tc.name)
			start := time.Now()
			results, err := exec.Exec(inputTensor)
			if err != nil {
				t.Fatalf("Failed to execute graph: %v", err)
			}
			fmt.Printf("done (%v)\n", time.Since(start))

			if len(results) < testModel.Config.NumHiddenLayers+1 {
				t.Fatalf("Expected at least %d outputs from graph, got %d", testModel.Config.NumHiddenLayers+1, len(results))
			}

			for _, l := range finalLayersToCheck {
				outTensor := results[l]
				expectedData := expectedLayers[l]
				name := fmt.Sprintf("Layer %d Output", l)
				if l == 0 {
					name = "Token Embeddings"
				}
				outTensor.ConstFlatData(func(flatAny any) {
					flat := flatAny.([]float32)
					outShape := outTensor.Shape()
					if outShape.Rank() < 3 {
						t.Fatalf("[%s] Expected rank >= 3, got %s", name, outShape)
					}
					batchSize := outShape.Dimensions[1]
					if batchSize != len(tokens) {
						t.Fatalf("[%s] Expected %d tokens in output, got %d -- output shape is %s", name, len(tokens), batchSize, outShape)
					}
					if outShape.Dimensions[2] != testModel.Config.HiddenSize {
						t.Fatalf("[%s] Expected hidden size %d, got %d -- output shape is %s", name, testModel.Config.HiddenSize, outShape.Dimensions[2], outShape)
					}
					validateTensor(t, flat, expectedData, name)
				})
			}
		})
	}
}

func validateTensor(t *testing.T, got []float32, expected []float32, name string) {
	if len(got) != len(expected) {
		t.Fatalf("[%s] Shape mismatch: expected %d flat floats (%d tokens), got %d (%d tokens)",
			name, len(expected), len(expected)/EmbeddingDim, len(got), len(got)/EmbeddingDim)
	}

	var sumAbsDiff, sumAbsExpected float64
	const minRelDenominator = 0.2
	for i, gotValue := range got {
		expectValue := float64(expected[i])
		gotValueF64 := float64(gotValue)

		absDiff := math.Abs(gotValueF64 - expectValue)
		sumAbsDiff += absDiff
		sumAbsExpected += math.Abs(expectValue)
	}
	meanAbsDiff := sumAbsDiff / float64(len(got))
	meanAbsExpected := sumAbsExpected / float64(len(got))

	var maxRelDiff float64
	var maxRelDiffIdx int
	for i, gotValue := range got {
		expectValue := float64(expected[i])
		gotValueF64 := float64(gotValue)
		absDiff := math.Abs(gotValueF64 - expectValue)
		relDenominator := math.Max(math.Abs(expectValue), math.Abs(gotValueF64))
		relDenominator = max(relDenominator, meanAbsExpected)
		relDiff := absDiff / relDenominator
		if relDiff > maxRelDiff {
			maxRelDiff = relDiff
			maxRelDiffIdx = i
		}
	}

	maxRelTolerance := 5.0
	meanTolerance := 0.1

	if maxRelDiff > maxRelTolerance || meanAbsDiff >= meanTolerance*meanAbsExpected {
		t.Errorf("[%s] Mismatch in values: Max rel diff: %.3g at idx %d (ex %f, got %f) / "+
			"Mean abs diff: %.3g (== %.1f%% of the mean absolute values %.3g)",
			name, maxRelDiff, maxRelDiffIdx, expected[maxRelDiffIdx], got[maxRelDiffIdx],
			meanAbsDiff, 100*meanAbsDiff/meanAbsExpected, meanAbsExpected)
		for i, gotValue := range got {
			expectValue := float64(expected[i])
			gotValueF64 := float64(gotValue)
			if i > 10 && i < len(got)-10 {
				continue
			}
			fmt.Printf("\t- Value #%d:\tgot %.3g,\t expected %.3g\n", i, gotValueF64, expectValue)
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

	fmt.Printf("- Pre-compiling model ...")
	start := time.Now()
	dummyTokens := tensors.FromValue([][]int32{{2}})
	_, err = exec.Exec(dummyTokens)
	if err != nil {
		t.Fatalf("Failed to compile graph: %v", err)
	}
	fmt.Printf("done (%v)\n", time.Since(start))

	fmt.Printf("- Executing model ...")
	start = time.Now()
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
			validateTensor(t, flat, expectedData, fmt.Sprintf("Sentence Embedding %d", i))
		})
	}
	fmt.Printf("done (%v)\n", time.Since(start))
}

func TestSimilarity(t *testing.T) {
	fmt.Printf("Queries: %q\n", testQueries)

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
		queryEmbeddings := graph.Concatenate(allEmbeddings[:len(testQueries)], 0)
		docEmbeddings := graph.Concatenate(allEmbeddings[len(testQueries):], 0)
		return testModel.Similarity(queryEmbeddings, docEmbeddings)
	}, allEmbeddingsAny...))
	fmt.Printf("Similarities: %v\n", similarities)
	want := []float32{0.9316, 0.3984, 0.4251, 0.7317}
	got := tensors.MustCopyFlatData[float32](similarities)
	require.InDeltaSlicef(t, want, got, 1e-2, "Similaries don't match!")
}
