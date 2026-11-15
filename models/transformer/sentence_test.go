package transformer_test

import (
	"fmt"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/go-huggingface/models/transformer"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
)

func TestApplySentencePooling(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		const (
			MeanPool = 1
			LastPool = 2
		)
		mask := [][]bool{
			{true, true, true, true},
			{true, true, true, false},
		}
		testCases := []struct {
			name     string
			mask     [][]bool
			poolType int
			expected [][]float32
		}{
			{"mean_pool-no_mask", nil, MeanPool, [][]float32{{2.5, 2.5, 2.5}, {6.5, 6.5, 6.5}}},
			{"mean_pool-with_mask", mask, MeanPool, [][]float32{{2.5, 2.5, 2.5}, {6, 6, 6}}},
			{"last_pool-no_mask", nil, LastPool, [][]float32{{4, 4, 4}, {8, 8, 8}}},
			{"last_pool-with_mask", mask, LastPool, [][]float32{{4, 4, 4}, {7, 7, 7}}},
		}
		for _, tc := range testCases {
			graphtest.RunTestGraphFnWithBackend(t, tc.name, backend,
				func(g *graph.Graph) (inputs, outputs []*graph.Node) {
					// [batchSize=2, seqLen=4, hiddenDim=3]
					hiddenStates := graph.OnePlus(graph.Iota(g, shapes.Make(dtypes.Float32, 2*4, 3), 0))
					hiddenStates = graph.Reshape(hiddenStates, 2, 4, 3)
					var mask *graph.Node
					var seqLen *graph.Node
					if tc.mask != nil {
						mask = graph.Const(g, tc.mask)
						seqLen = graph.ReduceSum(graph.ConvertDType(mask, dtypes.Int32), 1)
					}
					m := &transformer.Model{PoolingConfig: &transformer.PoolingConfig{}}
					switch tc.poolType {
					case MeanPool:
						m.PoolingConfig.PoolingModeMeanTokens = true
					case LastPool:
						m.PoolingConfig.PoolingModeLastToken = true
					default:
						t.Fatalf("unknown pool type %d", tc.poolType)
					}
					out := m.ApplySentencePooling(hiddenStates, seqLen)
					fmt.Printf("out.shape=%s\n", out.Shape())
					if mask != nil {
						return []*graph.Node{hiddenStates, mask}, []*graph.Node{out}
					}
					return []*graph.Node{hiddenStates}, []*graph.Node{out}
				}, []any{tc.expected}, 1e-3)
		}

		// Default pooling when PoolingConfig == nil:
		// - Causal models default to last-token pooling (and handle seqLen == 0 padded rows safely).
		// - Non-causal models default to mean pooling.
		graphtest.RunTestGraphFnWithBackend(t, "default_causal_last_token_and_zero_seqlen", backend,
			func(g *graph.Graph) (inputs, outputs []*graph.Node) {
				hiddenStates := graph.OnePlus(graph.Iota(g, shapes.Make(dtypes.Float32, 3*4, 2), 0))
				hiddenStates = graph.Reshape(hiddenStates, 3, 4, 2)
				seqLen := graph.Const(g, []int32{4, 2, 0})
				m := (&transformer.Model{}).WithCausalMask(true)
				out := m.SentenceEmbeddingFromLastHiddenState(hiddenStates, seqLen)
				return []*graph.Node{hiddenStates, seqLen}, []*graph.Node{out}
			}, []any{[][]float32{{4, 4}, {6, 6}, {9, 9}}}, 1e-3)

		graphtest.RunTestGraphFnWithBackend(t, "default_bidirectional_mean_pool", backend,
			func(g *graph.Graph) (inputs, outputs []*graph.Node) {
				hiddenStates := graph.OnePlus(graph.Iota(g, shapes.Make(dtypes.Float32, 2*4, 2), 0))
				hiddenStates = graph.Reshape(hiddenStates, 2, 4, 2)
				seqLen := graph.Const(g, []int32{4, 3})
				m := (&transformer.Model{}).WithCausalMask(false)
				out := m.SentenceEmbeddingFromLastHiddenState(hiddenStates, seqLen)
				return []*graph.Node{hiddenStates, seqLen}, []*graph.Node{out}
			}, []any{[][]float32{{2.5, 2.5}, {6, 6}}}, 1e-3)
	})
}

func TestSentenceHalfPrecisionPoolingAndNormalize(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		for _, dt := range []dtypes.DType{dtypes.Float16, dtypes.BFloat16} {
			t.Run(dt.String(), func(t *testing.T) {
				// Large values (300.0 -> 300^2 = 90000 > Float16 max 65504) and zero vectors
				// verify Float32 promotion in mean pooling and L2 normalization.
				m := &transformer.Model{
					PoolingConfig: &transformer.PoolingConfig{PoolingModeMeanTokens: true},
					Modules: []transformer.ModuleConfig{
						{Type: "sentence_transformers.models.Pooling"},
						{Type: "sentence_transformers.models.Normalize"},
					},
				}
				m.WithCausalMask(false)

				exec := graph.MustNewExec(backend, func(hidden, seqLen *graph.Node) *graph.Node {
					h := graph.ConvertDType(hidden, dt)
					out := m.SentenceEmbeddingFromLastHiddenState(h, seqLen)
					assert.Equal(t, dt, out.DType())
					return graph.ConvertDType(out, dtypes.Float32)
				})

				// Row 0: valid tokens have [300, 400] -> norm=500 -> normalized [0.6, 0.8]
				// Row 1: zero tokens -> eps=1e-12 prevents 0/0=NaN -> normalized [0, 0]
				hiddenIn := tensors.FromValue([][][]float32{
					{{300, 400}, {300, 400}, {0, 0}},
					{{0, 0}, {0, 0}, {0, 0}},
				})
				seqLenIn := tensors.FromValue([]int32{2, 2})
				got := exec.MustExec(hiddenIn, seqLenIn)[0].Value().([][]float32)

				assert.InDeltaSlice(t, []float32{0.6, 0.8}, got[0], 1e-2)
				assert.InDeltaSlice(t, []float32{0, 0}, got[1], 1e-6)
			})
		}
	})
}
