package timestral

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/tensors"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func tinyForecaster(t *testing.T, blockSize, maxContext int) (*Forecaster, compute.Backend) {
	t.Helper()
	backend := compute.MustNew()

	cfg := DefaultConfig()
	cfg.PatchSize = 8
	cfg.DModel = 32
	cfg.NumHeads = 4
	cfg.MLPHiddenDim = 64
	cfg.BlockSchedule = []LayerKind{
		LayerTime,
		LayerTimeVariate,
		LayerTime,
		LayerTimeVariate,
	}
	cfg.BlockSize = blockSize
	cfg.ContextLength = maxContext

	tmModel, err := NewModel(cfg)
	require.NoError(t, err)

	store := model.NewStore()
	require.NoError(t, store.SetRNGStateFromSeed(42))

	// Build a dummy forward pass once so all model variables are registered in store,
	// using RandomNormalFn(scope, 0.05) as the default initializer for Dense weights.
	initExec, err := model.NewExec(backend, store, func(scope *model.Scope, x *Node) *Node {
		scope = scope.WithInitializer(initializer.RandomNormalFn(scope, 0.05))
		return tmModel.BuildGraph(scope, x, 8).SymmetricAveraging(false).Done()
	})
	require.NoError(t, err)
	dummy := tensors.FromFlatDataAndDimensions(make([]float32, 16), 1, 1, 16)
	out, err := initExec.Exec1(dummy)
	require.NoError(t, err)
	out.FinalizeAll()
	dummy.FinalizeAll()

	// Scale forecast_query and RMSNorm scales with slight random normal perturbation too,
	// matching Python's `for p in model.parameters(): torch.nn.init.normal_(p, std=0.05)`.
	rng := rand.New(rand.NewPCG(1, 2))
	for v := range store.IterVariables() {
		tVal, err := v.Value()
		require.NoError(t, err)
		tensors.MutableFlatData(tVal, func(flat []float32) {
			for i := range flat {
				flat[i] = float32(rng.NormFloat64() * 0.05)
			}
		})
	}
	store.WithVariableAsConst(true)

	fc := NewForecasterWithStore(backend, tmModel, store)
	return fc, backend
}

func makeSeries(batch, variates, length int, seed uint64, positive bool) [][][]float32 {
	rng := rand.New(rand.NewPCG(seed, seed+1))
	out := make([][][]float32, batch)
	for b := range batch {
		out[b] = make([][]float32, variates)
		for v := range variates {
			row := make([]float32, length)
			var cum float64
			for t := range length {
				cum += rng.NormFloat64()
				val := 10*math.Sin(2*math.Pi*float64(t)/24.0) + cum
				if positive {
					val = math.Abs(val) + 1.0
				}
				row[t] = float32(val)
			}
			out[b][v] = row
		}
	}
	return out
}

func negateSeries3D(x [][][]float32) [][][]float32 {
	out := make([][][]float32, len(x))
	for b := range x {
		out[b] = make([][]float32, len(x[b]))
		for v := range x[b] {
			out[b][v] = make([]float32, len(x[b][v]))
			for t, val := range x[b][v] {
				out[b][v][t] = -val
			}
		}
	}
	return out
}

func TestConfigValidation(t *testing.T) {
	cfg := DefaultConfig()
	require.NoError(t, cfg.Validate())
	assert.Equal(t, 4, cfg.MedianIndex())
	assert.Equal(t, 9, cfg.NumQuantiles())
	assert.True(t, cfg.AreQuantilesSymmetric())
	assert.True(t, cfg.SupportsKnownCovariates())
	assert.Equal(t, 64, cfg.HeadDim())
	assert.Equal(t, 32, cfg.RoPEDim())

	bad := *cfg
	bad.QuantileKnots = []float64{0.1, 0.9} // missing 0.5
	require.Error(t, bad.Validate())

	bad2 := *cfg
	bad2.BlockSchedule = []LayerKind{"invalid_kind"}
	require.Error(t, bad2.Validate())

	noKnown := *cfg
	noKnown.CovariateEnabled = false
	assert.False(t, noKnown.SupportsKnownCovariates())

	timeOnly := *cfg
	timeOnly.BlockSchedule = []LayerKind{LayerTime, LayerTime}
	assert.False(t, timeOnly.SupportsKnownCovariates())
}

func TestForecastShapesAndFinite(t *testing.T) {
	fc, _ := tinyForecaster(t, 4, 256)

	cases := []struct {
		name     string
		variates int
		length   int
		horizon  int
	}{
		{"OneBlock", 1, 100, 12},
		{"ContextPastCap", 3, 300, 50},
		{"ThreeBlocksShortContext", 2, 37, 70},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x := makeSeries(2, tc.variates, tc.length, 0, false)
			for i := 5; i < 9 && i < tc.length; i++ {
				x[0][0][i] = float32(math.NaN())
			}

			res, err := fc.Forecast(x, tc.horizon).Done()
			require.NoError(t, err)
			defer res.Close()

			qShape := res.Quantiles.Shape()
			assert.Equal(t, dtypes.Float32, qShape.DType)
			assert.Equal(t, []int{2, tc.variates, tc.horizon, 9}, qShape.Dimensions)

			numQ := 9
			tensors.ConstFlatData(res.Quantiles, func(flat []float32) {
				for idx, val := range flat {
					require.Falsef(t, math.IsNaN(float64(val)) || math.IsInf(float64(val), 0),
						"non-finite quantile at index %d: %v", idx, val)
				}
				// Verify quantiles are sorted (monotone non-decreasing along last axis).
				numSteps := len(flat) / numQ
				for s := range numSteps {
					base := s * numQ
					for q := range numQ - 1 {
						require.GreaterOrEqualf(t, flat[base+q+1], flat[base+q]-1e-6,
							"quantiles not sorted at step %d: q[%d]=%v > q[%d]=%v",
							s, q, flat[base+q], q+1, flat[base+q+1])
					}
				}
			})

			med, err := res.Median()
			require.NoError(t, err)
			assert.Equal(t, []int{2, tc.variates, tc.horizon}, med.Shape().Dimensions)
			med.FinalizeAll()

			q90, err := res.Quantile(0.9)
			require.NoError(t, err)
			assert.Equal(t, []int{2, tc.variates, tc.horizon}, q90.Shape().Dimensions)
			q90.FinalizeAll()
		})
	}
}

func TestSymmetricAveragingIsOddWithCovariates(t *testing.T) {
	fc, _ := tinyForecaster(t, 4, 256)

	x := makeSeries(2, 1, 120, 3, false)
	past := makeSeries(2, 2, 120, 1, false)
	known := makeSeries(2, 1, 120+24, 2, false)

	posRes, err := fc.Forecast(x, 24).
		WithPastCovariates(past).
		WithKnownCovariates(known).
		ClipNonNegative(false).
		Done()
	require.NoError(t, err)
	defer posRes.Close()

	negRes, err := fc.Forecast(negateSeries3D(x), 24).
		WithPastCovariates(negateSeries3D(past)).
		WithKnownCovariates(negateSeries3D(known)).
		ClipNonNegative(false).
		Done()
	require.NoError(t, err)
	defer negRes.Close()

	numQ := len(posRes.QuantileLevels)
	tensors.ConstFlatData(posRes.Quantiles, func(posFlat []float32) {
		tensors.ConstFlatData(negRes.Quantiles, func(negFlat []float32) {
			require.Equal(t, len(posFlat), len(negFlat))
			numSteps := len(posFlat) / numQ
			for s := range numSteps {
				base := s * numQ
				for q := range numQ {
					expected := -posFlat[base+(numQ-1-q)]
					actual := negFlat[base+q]
					require.InDeltaf(t, expected, actual, 1e-4,
						"mismatch at step %d quantile %d: expected %v, got %v", s, q, expected, actual)
				}
			}
		})
	})
}

func TestCategoricalCovariatesAreTargetEncoded(t *testing.T) {
	fc, _ := tinyForecaster(t, 4, 256)

	x := makeSeries(2, 1, 120, 0, true)
	rng := rand.New(rand.NewPCG(7, 8))
	catKnown := make([][][]float32, 2)
	for b := range 2 {
		catKnown[b] = make([][]float32, 1)
		catKnown[b][0] = make([]float32, 120+24)
		for idx := range 120 + 24 {
			catKnown[b][0][idx] = float32(rng.IntN(4))
		}
	}
	for idx := 10; idx < 20; idx++ {
		catKnown[0][0][idx] = float32(math.NaN())
	}

	outCat, err := fc.Forecast(x, 24).
		WithCategoricalKnownCovariates(catKnown).
		SymmetricAveraging(false).
		Done()
	require.NoError(t, err)
	defer outCat.Close()

	// Manually TargetEncode and pass via WithKnownCovariates.
	encoded := make([][][]float32, 2)
	for b := range 2 {
		tgt64 := f32SliceToF64(x[b][0])
		cat64 := f32SliceToF64(catKnown[b][0])
		encCtx, encFut := TargetEncode(tgt64, cat64[:120], cat64[120:], DefaultTargetEncodingSmooth)
		row := make([]float32, 0, 120+24)
		row = append(row, encCtx...)
		row = append(row, encFut...)
		encoded[b] = [][]float32{row}
	}

	outManual, err := fc.Forecast(x, 24).
		WithKnownCovariates(encoded).
		SymmetricAveraging(false).
		Done()
	require.NoError(t, err)
	defer outManual.Close()

	tensors.ConstFlatData(outCat.Quantiles, func(catFlat []float32) {
		tensors.ConstFlatData(outManual.Quantiles, func(manFlat []float32) {
			require.InDeltaSlice(t, manFlat, catFlat, 1e-6)
		})
	})
}

func TestLoadConfigFromLocalRepo(t *testing.T) {
	tmpDir := t.TempDir()
	cfg := DefaultConfig()
	cfg.DModel = 64
	cfg.NumHeads = 4
	data, err := json.MarshalIndent(cfg, "", "  ")
	require.NoError(t, err)
	require.NoError(t, os.WriteFile(filepath.Join(tmpDir, "config.json"), data, 0o644))

	repo := hub.NewLocal(tmpDir)

	loadedModel, err := LoadModel(repo)
	require.NoError(t, err)
	assert.Equal(t, 64, loadedModel.Config.DModel)
	assert.Equal(t, 4, loadedModel.Config.NumHeads)
}

func f32SliceToF64(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}

func f64SliceToF32(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}
