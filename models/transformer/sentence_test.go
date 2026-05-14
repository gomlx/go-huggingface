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
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/support/testutil"
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
					if tc.mask != nil {
						mask = graph.Const(g, tc.mask)
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
					out := m.ApplySentencePooling(hiddenStates, mask)
					fmt.Printf("out.shape=%s\n", out.Shape())
					if mask != nil {
						return []*graph.Node{hiddenStates, mask}, []*graph.Node{out}
					}
					return []*graph.Node{hiddenStates}, []*graph.Node{out}
				}, []any{tc.expected}, 1e-3)
		}
	})
}
