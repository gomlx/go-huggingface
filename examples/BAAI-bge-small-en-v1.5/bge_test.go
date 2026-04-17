package bge

import (
	"flag"
	"fmt"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/go-huggingface/hub"
	hftesting "github.com/gomlx/go-huggingface/internal/testing"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
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
	testQueries = []string{
		QueryInstruction + "What is the capital of China?",
		QueryInstruction + "Explain gravity",
	}
	testDocs = []string{
		"The capital of China is Beijing.",
		"Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
	}
	testPadID int32
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
	testRepo = hub.New(Repository)
	if err := testRepo.DownloadInfo(false); err != nil {
		fmt.Printf("Failed to LoadRepo: %v\n", err)
		os.Exit(1)
	}

	testModel, err = transformer.LoadModel(testRepo)
	if err != nil {
		fmt.Printf("Failed to LoadModel: %v\n", err)
		os.Exit(1)
	}
	if testModel.Config.HiddenSize != EmbeddingDim {
		fmt.Printf("Model configuration 'HiddenSize=%d', expected %d\n", testModel.Config.HiddenSize, EmbeddingDim)
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
	fmt.Printf("  - Config: %+v\n", testModel.Config)

	tokenizer := must1(testModel.GetTokenizer())
	padID, err := tokenizer.SpecialTokenID(api.TokPad)
	if err != nil {
		fmt.Printf("Padding token not defined: %+v\n", err)
		os.Exit(1)
	}
	testPadID = int32(padID)

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

func TestTokenization(t *testing.T) {
	tokenizer := must1(testModel.GetTokenizer())
	expectedQueries := [][]int{
		{101, 100, 100, 100, 100, 1816, 1910, 1854, 100, 1923, 100, 100, 100, 100, 100, 1919, 100, 1861, 1932, 1993, 2054, 2003, 1996, 3007, 1997, 2859, 1029, 102},
		{101, 100, 100, 100, 100, 1816, 1910, 1854, 100, 1923, 100, 100, 100, 100, 100, 1919, 100, 1861, 1932, 1993, 4863, 8992, 102},
	}

	for i, q := range testQueries {
		got := tokenizer.Encode(q)
		expected := expectedQueries[i]
		if len(got) != len(expected) {
			t.Errorf("Query %d: expected length %d, got %d", i, len(expected), len(got))
		}
		for j := 0; j < len(got) && j < len(expected); j++ {
			if got[j] != expected[j] {
				t.Errorf("Query %d at index %d: expected %d, got %d", i, j, expected[j], got[j])
			}
		}
	}
}

func TestSentenceEmbedding(t *testing.T) {
	prompts := make([]string, 0, len(testQueries)+len(testDocs))
	prompts = append(prompts, testQueries...)
	prompts = append(prompts, testDocs...)

	pythonPath := "similarity_embeddings.txt"
	expectedFlatData, err := hftesting.ReadPythonEmbeddingsList(pythonPath)
	if err != nil {
		t.Fatalf("Skipping test because %s is not available: %v", pythonPath, err)
	}
	if len(expectedFlatData) != len(prompts)*EmbeddingDim {
		t.Fatalf("The ground truth file %q has %d values, but expected %d x %d = %d instead",
			pythonPath, len(expectedFlatData), len(prompts), EmbeddingDim, len(prompts)*EmbeddingDim)
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
			hftesting.ValidateEmbeddingTensor(t, flat, expectedData, fmt.Sprintf("Sentence Embedding %d", i),
				1, len(tokens), len(tokens))
		})
	}
}
