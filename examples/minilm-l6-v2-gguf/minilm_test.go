package minilmgguf

import (
	"flag"
	"fmt"
	"math"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/hub"
	hftesting "github.com/gomlx/go-huggingface/internal/testing"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagSkipLoadingWeights = flag.Bool("skip_loading_weights", false, "Skip loading weights from the model.")
)

var (
	testBackend compute.Backend
	testRepo    *hub.Repo
	testStore   *model.Store
	testModel   *transformer.Model
	testQueries = []string{
		"What is the capital of China?",
		"Explain gravity",
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
	flag.Parse()
	if testing.Short() {
		fmt.Printf("🚨 Skipping tests because -short flag is set.\n")
		os.Exit(0)
	}

	var err error
	testBackend, err = compute.New()
	if err != nil {
		fmt.Printf("Failed to initialize backend: %v\n", err)
		os.Exit(1)
	}

	testStore = model.NewStore()
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

	tokenizer := must1(testModel.GetTokenizer())
	padID, err := tokenizer.SpecialTokenID(api.TokPad)
	if err != nil {
		fmt.Printf("Padding token not defined: %+v\n", err)
		os.Exit(1)
	}
	testPadID = int32(padID)

	if !*flagSkipLoadingWeights {
		fmt.Printf(" - Loading GGUF model weights (%s) ...\r", testModel.GGUFFile())
		start := time.Now()
		must(testModel.LoadStore(testBackend, testStore))
		for range 3 {
			runtime.GC()
		}
		fmt.Printf("\r✅ Loading GGUF model weights (%s): done (%s)\n", testModel.GGUFFile(), humanize.Duration(time.Since(start)))
	}

	code := m.Run()
	testBackend.Finalize()
	os.Exit(code)
}

func TestModelSelection(t *testing.T) {
	assert.True(t, testModel.IsGGUF())
	assert.Equal(t, DefaultGGUFFile, testModel.GGUFFile())
	assert.False(t, testModel.UseCausalMask())
}

func TestTokenization(t *testing.T) {
	tokenizer := must1(testModel.GetTokenizer())
	prompts := append(append([]string(nil), testQueries...), testDocs...)
	// Ground-truth token IDs produced by llama.cpp (llama.Tokenize with addSpecial=true).
	expectedTokens := [][]int{
		{101, 2054, 2003, 1996, 3007, 1997, 2859, 1029, 102},
		{101, 4863, 8992, 102},
		{101, 1996, 3007, 1997, 2859, 2003, 7211, 1012, 102},
		{101, 8992, 2003, 1037, 2486, 2008, 17771, 2048, 4230, 2875, 2169, 2060, 1012, 2009, 3957, 3635, 2000, 3558, 5200, 1998, 2003, 3625, 2005, 1996, 2929, 1997, 11358, 2105, 1996, 3103, 1012, 102},
	}

	for i, p := range prompts {
		got := tokenizer.Encode(p)
		assert.Equalf(t, expectedTokens[i], got, "Prompt %d token IDs mismatch vs llama.cpp", i)
	}
}

func TestSentenceEmbedding(t *testing.T) {
	prompts := make([]string, 0, len(testQueries)+len(testDocs))
	prompts = append(prompts, testQueries...)
	prompts = append(prompts, testDocs...)

	refPath := "similarity_embeddings.txt"
	expectedFlatData, err := hftesting.ReadPythonEmbeddingsList(refPath)
	require.NoError(t, err)
	require.Equal(t, len(prompts)*EmbeddingDim, len(expectedFlatData))

	exec, err := model.NewExec(testBackend, testStore, func(scope *model.Scope, tokens *graph.Node) *graph.Node {
		constPad := graph.Scalar(tokens.Graph(), tokens.DType(), testPadID)
		seqLen := graph.ReduceSum(graph.ConvertDType(graph.NotEqual(tokens, constPad), dtypes.Int32), 1)
		x := testModel.SentenceEmbeddingGraph(scope, tokens, seqLen)
		return graph.ConvertDType(x, dtypes.Float32)
	})
	require.NoError(t, err)

	tokenizer := must1(testModel.GetTokenizer())
	for i, prompt := range prompts {
		tokensInt := tokenizer.Encode(prompt)
		tokens := make([]int32, len(tokensInt))
		for j, tok := range tokensInt {
			tokens[j] = int32(tok)
		}
		results, err := exec.Exec(tensors.FromValue([][]int32{tokens}))
		require.NoError(t, err)

		outTensor := results[0]
		require.Equal(t, []int{1, EmbeddingDim}, outTensor.Shape().Dimensions)

		outTensor.ConstFlatData(func(flatAny any) {
			flat := flatAny.([]float32)
			expectedData := expectedFlatData[i*EmbeddingDim : (i+1)*EmbeddingDim]
			hftesting.ValidateEmbeddingTensor(t, flat, expectedData, fmt.Sprintf("Sentence Embedding %d", i),
				1, 1, 1)

			// Compute cosine similarity and max absolute difference against llama.cpp.
			var dot, normGotSq, normExpSq, maxAbs float64
			for k := range EmbeddingDim {
				g := float64(flat[k])
				e := float64(expectedData[k])
				dot += g * e
				normGotSq += g * g
				normExpSq += e * e
				if d := math.Abs(g - e); d > maxAbs {
					maxAbs = d
				}
			}
			cosSim := dot / (math.Sqrt(normGotSq) * math.Sqrt(normExpSq))
			t.Logf("Prompt %d vs llama.cpp: cosine_similarity=%.8f, max_abs_diff=%.6g", i, cosSim, maxAbs)
			assert.Greaterf(t, cosSim, 0.9995, "Prompt %d cosine similarity vs llama.cpp", i)
			assert.Lessf(t, maxAbs, 5e-3, "Prompt %d max abs diff vs llama.cpp", i)
		})
	}
}

func TestCheckUnloadedVariables(t *testing.T) {
	loadedPaths := make(map[string]bool)
	for v := range testStore.IterVariables() {
		loadedPaths[v.Path()] = true
	}

	exec, err := model.NewExec(testBackend, testStore, func(scope *model.Scope, tokens *graph.Node) *graph.Node {
		x := testModel.SentenceEmbeddingGraph(scope, tokens, nil)
		return graph.ConvertDType(x, dtypes.Float32)
	})
	require.NoError(t, err)
	_ = exec

	var missing []string
	for v := range testStore.IterVariables() {
		if !loadedPaths[v.Path()] {
			missing = append(missing, v.Path())
		}
	}
	assert.Empty(t, missing, "expected zero unloaded variables during SentenceEmbeddingGraph construction")
}
