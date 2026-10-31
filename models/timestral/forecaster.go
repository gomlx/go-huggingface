// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package timestral

import (
	"fmt"
	"math"
	"slices"
	"sync"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/pkg/errors"
)

// Forecaster wraps a Timestral Model, its GoMLX Store, and cached compiled executors.
type Forecaster struct {
	backend               compute.Backend
	model                 *Model
	store                 *model.Store
	maxContext            int
	perShotHorizonPatches int
	quantileRealCapK      float64
	extremeValueThreshold float64

	mu        sync.Mutex
	execCache map[execCacheKey]*model.Exec
}

type execCacheKey struct {
	horizonSteps          int
	maxContext            int
	perShotHorizonPatches int
	quantileRealCapK      float64
	extremeValueThreshold float64
	symmetricAveraging    bool
	clipNonNegative       bool
	hasObsMask            bool
	hasPastCov            bool
	hasKnownCov           bool
}

// NewForecaster creates a Forecaster for the given backend and loaded Model, and loads
// the model weights from modelObj.Repo into an internal GoMLX model.Store.
func NewForecaster(backend compute.Backend, modelObj *Model) (*Forecaster, error) {
	if modelObj == nil {
		return nil, errors.New("timestral.NewForecaster: modelObj cannot be nil")
	}
	store := model.NewStore().WithVariableAsConst(true)
	if modelObj.Repo != nil {
		if err := modelObj.LoadStore(backend, store); err != nil {
			return nil, err
		}
	}
	return NewForecasterWithStore(backend, modelObj, store), nil
}

// NewForecasterWithStore creates a Forecaster using an existing GoMLX model.Store.
// Useful when variables are already loaded or initialized in store.
func NewForecasterWithStore(backend compute.Backend, modelObj *Model, store *model.Store) *Forecaster {
	perShot := min(32, modelObj.Config.BlockSize)
	return &Forecaster{
		backend:               backend,
		model:                 modelObj,
		store:                 store,
		maxContext:            modelObj.Config.ContextLength,
		perShotHorizonPatches: perShot,
		quantileRealCapK:      1e4,
		extremeValueThreshold: 1e30,
		execCache:             make(map[execCacheKey]*model.Exec),
	}
}

// MaxContext sets the context cap in time steps (most recent kept); defaults to Config.ContextLength.
func (f *Forecaster) MaxContext(steps int) *Forecaster {
	f.maxContext = steps
	return f
}

// PerShotHorizonPatches sets the number of patches decoded per block, capped at Config.BlockSize.
func (f *Forecaster) PerShotHorizonPatches(patches int) *Forecaster {
	f.perShotHorizonPatches = min(patches, f.model.Config.BlockSize)
	return f
}

// QuantileRealCapK sets the real-space quantile clamp factor k (defaults to 1e4; 0 disables).
func (f *Forecaster) QuantileRealCapK(k float64) *Forecaster {
	f.quantileRealCapK = k
	return f
}

// ExtremeValueThreshold sets the threshold at or above which fed-back medians become 0 (defaults to 1e30).
func (f *Forecaster) ExtremeValueThreshold(t float64) *Forecaster {
	f.extremeValueThreshold = t
	return f
}

// Model returns the underlying Timestral Model.
func (f *Forecaster) Model() *Model {
	return f.model
}

// Store returns the underlying GoMLX model.Store.
func (f *Forecaster) Store() *model.Store {
	return f.store
}

// Close finalizes all cached executors and the underlying model.Store.
func (f *Forecaster) Close() {
	f.mu.Lock()
	defer f.mu.Unlock()
	for k, exec := range f.execCache {
		exec.Finalize()
		delete(f.execCache, k)
	}
	if f.store != nil {
		f.store.Finalize()
	}
}

// ForecastCall is a fluent builder for a single forecasting request, created via Forecaster.Forecast.
type ForecastCall struct {
	forecaster                  *Forecaster
	context                     any
	predictionLength            int
	observationMask             any
	pastCovariates              any
	knownCovariates             any
	categoricalPastCovariates   any
	categoricalKnownCovariates  any
	symmetricAveraging          bool
	clipNonNegative             bool
}

// Forecast starts building a forecast call for predictionLength future steps after context.
//
// context can be a *tensors.Tensor, []float32 (shape [T]), [][]float32 (shape [B, T]),
// or [][][]float32 (shape [B, V, T]). NaN marks missing values.
//
// By default, SymmetricAveraging(true) and ClipNonNegative(true) are enabled.
func (f *Forecaster) Forecast(context any, predictionLength int) *ForecastCall {
	return &ForecastCall{
		forecaster:         f,
		context:            context,
		predictionLength:   predictionLength,
		symmetricAveraging: true,
		clipNonNegative:    true,
	}
}

// WithObservationMask sets an optional boolean observation mask shaped like context (false = missing).
func (c *ForecastCall) WithObservationMask(mask any) *ForecastCall {
	c.observationMask = mask
	return c
}

// WithPastCovariates sets optional past-only numeric covariates shaped [B, C_past, T] or [B, T].
func (c *ForecastCall) WithPastCovariates(past any) *ForecastCall {
	c.pastCovariates = past
	return c
}

// WithKnownCovariates sets optional known-future numeric covariates shaped [B, C_known, T + predictionLength] or [B, T + predictionLength].
func (c *ForecastCall) WithKnownCovariates(known any) *ForecastCall {
	c.knownCovariates = known
	return c
}

// WithCategoricalPastCovariates sets optional past-only categorical covariates shaped [B, C, T] or [B, T].
func (c *ForecastCall) WithCategoricalPastCovariates(codes any) *ForecastCall {
	c.categoricalPastCovariates = codes
	return c
}

// WithCategoricalKnownCovariates sets optional known-future categorical covariates shaped [B, C, T + predictionLength] or [B, T + predictionLength].
func (c *ForecastCall) WithCategoricalKnownCovariates(codes any) *ForecastCall {
	c.categoricalKnownCovariates = codes
	return c
}

// SymmetricAveraging enables/disables averaging with the reflected forecast of -context (default: true).
func (c *ForecastCall) SymmetricAveraging(enabled bool) *ForecastCall {
	c.symmetricAveraging = enabled
	return c
}

// ClipNonNegative enables/disables clamping quantiles at 0 for series whose observed context is non-negative (default: true).
func (c *ForecastCall) ClipNonNegative(enabled bool) *ForecastCall {
	c.clipNonNegative = enabled
	return c
}

// Done executes the forecast and returns the *ForecastResult with Quantiles shaped [B, V, predictionLength, Q].
func (c *ForecastCall) Done() (*ForecastResult, error) {
	f := c.forecaster
	if c.predictionLength < 1 {
		return nil, errors.Errorf("predictionLength must be >= 1, got %d", c.predictionLength)
	}
	if f.perShotHorizonPatches < 1 {
		return nil, errors.Errorf("perShotHorizonPatches must be >= 1, got %d", f.perShotHorizonPatches)
	}

	ctxTensor, maskTensor, b, v, t, err := toContextAndMaskTensors(c.context, c.observationMask)
	if err != nil {
		return nil, err
	}
	defer ctxTensor.FinalizeAll()
	if maskTensor != nil {
		defer maskTensor.FinalizeAll()
	}

	pastTensor, err := toCovariateTensorF32(c.pastCovariates, b, t, "pastCovariates", f.model.Config.CovariateEnabled)
	if err != nil {
		return nil, err
	}
	if pastTensor != nil {
		defer pastTensor.FinalizeAll()
	}

	knownTensor, err := toCovariateTensorF32(c.knownCovariates, b, t+c.predictionLength, "knownCovariates", f.model.Config.CovariateEnabled)
	if err != nil {
		return nil, err
	}
	if knownTensor != nil {
		defer knownTensor.FinalizeAll()
	}

	if c.categoricalPastCovariates != nil || c.categoricalKnownCovariates != nil {
		ctxFlat, err := tensors.CopyFlatData[float32](ctxTensor)
		if err != nil {
			return nil, err
		}
		var maskFlat []bool
		if maskTensor != nil {
			maskFlat, err = tensors.CopyFlatData[bool](maskTensor)
			if err != nil {
				return nil, err
			}
		} else {
			maskFlat = make([]bool, len(ctxFlat))
			for i, val := range ctxFlat {
				maskFlat[i] = !math.IsNaN(float64(val)) && !math.IsInf(float64(val), 0)
			}
		}

		if c.categoricalPastCovariates != nil {
			catPastFlat, cPast, err := toCategoricalFlatF64(c.categoricalPastCovariates, b, t, "categoricalPastCovariates", f.model.Config.CovariateEnabled)
			if err != nil {
				return nil, err
			}
			encPast := encodeCategoricalSeries(catPastFlat, b, cPast, t, ctxFlat, maskFlat, v, t, f.maxContext)
			encPastTensor := tensors.FromFlatDataAndDimensions(encPast, b, cPast, t)
			combined, err := concatCovariateTensors(pastTensor, encPastTensor)
			if err != nil {
				encPastTensor.FinalizeAll()
				return nil, err
			}
			if pastTensor != nil && combined != pastTensor {
				pastTensor.FinalizeAll()
			}
			if combined != encPastTensor {
				encPastTensor.FinalizeAll()
			}
			pastTensor = combined
			defer pastTensor.FinalizeAll()
		}

		if c.categoricalKnownCovariates != nil {
			totalLen := t + c.predictionLength
			catKnownFlat, cKnown, err := toCategoricalFlatF64(c.categoricalKnownCovariates, b, totalLen, "categoricalKnownCovariates", f.model.Config.CovariateEnabled)
			if err != nil {
				return nil, err
			}
			encKnown := encodeCategoricalSeries(catKnownFlat, b, cKnown, totalLen, ctxFlat, maskFlat, v, t, f.maxContext)
			encKnownTensor := tensors.FromFlatDataAndDimensions(encKnown, b, cKnown, totalLen)
			combined, err := concatCovariateTensors(knownTensor, encKnownTensor)
			if err != nil {
				encKnownTensor.FinalizeAll()
				return nil, err
			}
			if knownTensor != nil && combined != knownTensor {
				knownTensor.FinalizeAll()
			}
			if combined != encKnownTensor {
				encKnownTensor.FinalizeAll()
			}
			knownTensor = combined
			defer knownTensor.FinalizeAll()
		}
	}

	if knownTensor != nil && !f.model.Config.SupportsKnownCovariates() {
		return nil, errors.New("this checkpoint cannot condition on known-future covariates")
	}
	if c.symmetricAveraging && !f.model.Config.AreQuantilesSymmetric() {
		return nil, errors.Errorf("symmetric averaging needs quantile levels symmetric about 0.5, got %v", f.model.Config.QuantileKnots)
	}

	key := execCacheKey{
		horizonSteps:          c.predictionLength,
		maxContext:            f.maxContext,
		perShotHorizonPatches: f.perShotHorizonPatches,
		quantileRealCapK:      f.quantileRealCapK,
		extremeValueThreshold: f.extremeValueThreshold,
		symmetricAveraging:    c.symmetricAveraging,
		clipNonNegative:       c.clipNonNegative,
		hasObsMask:            maskTensor != nil,
		hasPastCov:            pastTensor != nil,
		hasKnownCov:           knownTensor != nil,
	}

	exec, err := f.getOrBuildExec(key)
	if err != nil {
		return nil, err
	}

	var args []any
	args = append(args, ctxTensor)
	if maskTensor != nil {
		args = append(args, maskTensor)
	}
	if pastTensor != nil {
		args = append(args, pastTensor)
	}
	if knownTensor != nil {
		args = append(args, knownTensor)
	}

	outputs, err := exec.Exec(args...)
	if err != nil {
		return nil, err
	}
	quantiles := outputs[0]

	// Verify no NaNs in output (matching Python's forecast NaN check).
	flatQ, err := tensors.CopyFlatData[float32](quantiles)
	if err != nil {
		quantiles.FinalizeAll()
		return nil, err
	}
	nanCount := 0
	for _, val := range flatQ {
		if math.IsNaN(float64(val)) {
			nanCount++
		}
	}
	if nanCount > 0 {
		quantiles.FinalizeAll()
		return nil, errors.Errorf("forecast produced %d NaN values; the input is degenerate", nanCount)
	}

	return &ForecastResult{
		Quantiles:      quantiles,
		QuantileLevels: slices.Clone(f.model.Config.QuantileKnots),
		medianIndex:    f.model.Config.MedianIndex(),
		backend:        f.backend,
	}, nil
}

func (f *Forecaster) getOrBuildExec(key execCacheKey) (*model.Exec, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if exec, ok := f.execCache[key]; ok {
		return exec, nil
	}

	exec, err := model.NewExec(f.backend, f.store, func(scope *model.Scope, nodes []*Node) *Node {
		idx := 0
		ctxNode := nodes[idx]
		idx++

		gb := f.model.BuildGraph(scope, ctxNode, key.horizonSteps).
			MaxContext(key.maxContext).
			PerShotHorizonPatches(key.perShotHorizonPatches).
			QuantileRealCapK(key.quantileRealCapK).
			ExtremeValueThreshold(key.extremeValueThreshold).
			SymmetricAveraging(key.symmetricAveraging).
			ClipNonNegative(key.clipNonNegative)

		if key.hasObsMask {
			gb.WithObservationMask(nodes[idx])
			idx++
		}
		if key.hasPastCov {
			gb.WithPastCovariates(nodes[idx])
			idx++
		}
		if key.hasKnownCov {
			gb.WithKnownCovariates(nodes[idx])
			idx++
		}
		return gb.Done()
	})
	if err != nil {
		return nil, err
	}
	f.execCache[key] = exec
	return exec, nil
}

// ForecastResult holds the output of a Forecast call, mirroring Python's timestral.Forecast.
type ForecastResult struct {
	// Quantiles has shape [B, V, PredictionLength, Q] (float32).
	Quantiles      *tensors.Tensor
	QuantileLevels []float64
	medianIndex    int
	backend        compute.Backend
}

// Median extracts the median (0.5 quantile) forecast as a tensor of shape [B, V, PredictionLength].
func (r *ForecastResult) Median() (*tensors.Tensor, error) {
	return r.extractQuantileIndex(r.medianIndex)
}

// Quantile extracts the forecast at the given probability level (e.g. 0.1, 0.9) as [B, V, PredictionLength].
func (r *ForecastResult) Quantile(level float64) (*tensors.Tensor, error) {
	for i, q := range r.QuantileLevels {
		if math.Abs(q-level) < 1e-9 {
			return r.extractQuantileIndex(i)
		}
	}
	return nil, fmt.Errorf("level %v is not one of %v", level, r.QuantileLevels)
}

func (r *ForecastResult) extractQuantileIndex(qIdx int) (*tensors.Tensor, error) {
	dims := r.Quantiles.Shape().Dimensions
	b, v, h := dims[0], dims[1], dims[2]
	return transformTensor(r.backend, r.Quantiles, func(x *Node) *Node {
		return Reshape(SliceAxis(x, -1, AxisRange(qIdx, qIdx+1)), b, v, h)
	})
}

// Close finalizes the underlying Quantiles tensor.
func (r *ForecastResult) Close() {
	if r != nil && r.Quantiles != nil {
		r.Quantiles.FinalizeAll()
	}
}

func toAnyTensor(val any) (*tensors.Tensor, error) {
	if t, ok := val.(*tensors.Tensor); ok {
		return t.Clone()
	}
	return tensors.FromAnyValue(val)
}

func toContextAndMaskTensors(contextAny, maskAny any) (ctxTensor, maskTensor *tensors.Tensor, b, v, t int, err error) {
	rawCtx, err := toAnyTensor(contextAny)
	if err != nil {
		return nil, nil, 0, 0, 0, errors.Wrap(err, "invalid context")
	}
	defer rawCtx.FinalizeAll()

	dims := rawCtx.Shape().Dimensions
	switch len(dims) {
	case 1:
		b, v, t = 1, 1, dims[0]
	case 2:
		b, v, t = dims[0], 1, dims[1]
	case 3:
		b, v, t = dims[0], dims[1], dims[2]
	default:
		return nil, nil, 0, 0, 0, errors.Errorf("context must be [T], [B, T], or [B, V, T], got %s", rawCtx.Shape())
	}

	flatF32, err := copyAsFloat32Slice(rawCtx)
	if err != nil {
		return nil, nil, 0, 0, 0, err
	}
	ctxTensor = tensors.FromFlatDataAndDimensions(flatF32, b, v, t)

	if maskAny != nil {
		rawMask, err := toAnyTensor(maskAny)
		if err != nil {
			ctxTensor.FinalizeAll()
			return nil, nil, 0, 0, 0, errors.Wrap(err, "invalid observationMask")
		}
		defer rawMask.FinalizeAll()
		if rawMask.Shape().Size() != b*v*t {
			ctxTensor.FinalizeAll()
			return nil, nil, 0, 0, 0, errors.Errorf("observationMask size %d does not match context size %d", rawMask.Shape().Size(), b*v*t)
		}
		maskBool, err := copyAsBoolSlice(rawMask)
		if err != nil {
			ctxTensor.FinalizeAll()
			return nil, nil, 0, 0, 0, err
		}
		maskTensor = tensors.FromFlatDataAndDimensions(maskBool, b, v, t)
	}
	return ctxTensor, maskTensor, b, v, t, nil
}

func toCovariateTensorF32(covAny any, batch, length int, name string, covEnabled bool) (*tensors.Tensor, error) {
	if covAny == nil {
		return nil, nil
	}
	if !covEnabled {
		return nil, errors.Errorf("this checkpoint was not trained with covariates; got %s", name)
	}
	raw, err := toAnyTensor(covAny)
	if err != nil {
		return nil, errors.Wrapf(err, "invalid %s", name)
	}
	defer raw.FinalizeAll()

	dims := raw.Shape().Dimensions
	var c int
	switch len(dims) {
	case 2:
		if dims[0] != batch || dims[1] != length {
			return nil, errors.Errorf("%s must be [B=%d, C, %d], got %s", name, batch, length, raw.Shape())
		}
		c = 1
	case 3:
		if dims[0] != batch || dims[2] != length {
			return nil, errors.Errorf("%s must be [B=%d, C, %d], got %s", name, batch, length, raw.Shape())
		}
		c = dims[1]
	default:
		return nil, errors.Errorf("%s must be [B=%d, C, %d], got %s", name, batch, length, raw.Shape())
	}
	if c == 0 {
		return nil, nil
	}
	flatF32, err := copyAsFloat32Slice(raw)
	if err != nil {
		return nil, err
	}
	return tensors.FromFlatDataAndDimensions(flatF32, batch, c, length), nil
}

func toCategoricalFlatF64(catAny any, batch, length int, name string, covEnabled bool) ([]float64, int, error) {
	if !covEnabled {
		return nil, 0, errors.Errorf("this checkpoint was not trained with covariates; got %s", name)
	}
	raw, err := toAnyTensor(catAny)
	if err != nil {
		return nil, 0, errors.Wrapf(err, "invalid %s", name)
	}
	defer raw.FinalizeAll()

	dims := raw.Shape().Dimensions
	var c int
	switch len(dims) {
	case 2:
		if dims[0] != batch || dims[1] != length {
			return nil, 0, errors.Errorf("%s must be [B=%d, C, %d], got %s", name, batch, length, raw.Shape())
		}
		c = 1
	case 3:
		if dims[0] != batch || dims[2] != length {
			return nil, 0, errors.Errorf("%s must be [B=%d, C, %d], got %s", name, batch, length, raw.Shape())
		}
		c = dims[1]
	default:
		return nil, 0, errors.Errorf("%s must be [B=%d, C, %d], got %s", name, batch, length, raw.Shape())
	}
	flatF64, err := copyAsFloat64Slice(raw)
	if err != nil {
		return nil, 0, err
	}
	for _, v := range flatF64 {
		if math.IsInf(v, 0) {
			return nil, 0, errors.Errorf("%s must hold finite codes or NaN; got an infinite value", name)
		}
	}
	return flatF64, c, nil
}

func concatCovariateTensors(a, b *tensors.Tensor) (*tensors.Tensor, error) {
	if a == nil {
		return b, nil
	}
	if b == nil {
		return a, nil
	}
	dimsA := a.Shape().Dimensions
	dimsB := b.Shape().Dimensions
	batch, cA, length := dimsA[0], dimsA[1], dimsA[2]
	cB := dimsB[1]
	cTotal := cA + cB

	flatA, err := tensors.CopyFlatData[float32](a)
	if err != nil {
		return nil, err
	}
	flatB, err := tensors.CopyFlatData[float32](b)
	if err != nil {
		return nil, err
	}
	out := make([]float32, batch*cTotal*length)
	for i := range batch {
		copy(out[i*cTotal*length:(i*cTotal+cA)*length], flatA[i*cA*length:(i+1)*cA*length])
		copy(out[(i*cTotal+cA)*length:(i+1)*cTotal*length], flatB[i*cB*length:(i+1)*cB*length])
	}
	return tensors.FromFlatDataAndDimensions(out, batch, cTotal, length), nil
}

func copyAsFloat32Slice(t *tensors.Tensor) ([]float32, error) {
	switch t.DType() {
	case dtypes.Float32:
		return tensors.CopyFlatData[float32](t)
	case dtypes.Float64:
		f64, err := tensors.CopyFlatData[float64](t)
		if err != nil {
			return nil, err
		}
		out := make([]float32, len(f64))
		for i, v := range f64 {
			out[i] = float32(v)
		}
		return out, nil
	case dtypes.Int32:
		i32, err := tensors.CopyFlatData[int32](t)
		if err != nil {
			return nil, err
		}
		out := make([]float32, len(i32))
		for i, v := range i32 {
			out[i] = float32(v)
		}
		return out, nil
	case dtypes.Int64:
		i64, err := tensors.CopyFlatData[int64](t)
		if err != nil {
			return nil, err
		}
		out := make([]float32, len(i64))
		for i, v := range i64 {
			out[i] = float32(v)
		}
		return out, nil
	default:
		return nil, errors.Errorf("unsupported tensor dtype %s for float32 conversion", t.DType())
	}
}

func copyAsFloat64Slice(t *tensors.Tensor) ([]float64, error) {
	switch t.DType() {
	case dtypes.Float64:
		return tensors.CopyFlatData[float64](t)
	case dtypes.Float32:
		f32, err := tensors.CopyFlatData[float32](t)
		if err != nil {
			return nil, err
		}
		out := make([]float64, len(f32))
		for i, v := range f32 {
			out[i] = float64(v)
		}
		return out, nil
	case dtypes.Int32:
		i32, err := tensors.CopyFlatData[int32](t)
		if err != nil {
			return nil, err
		}
		out := make([]float64, len(i32))
		for i, v := range i32 {
			out[i] = float64(v)
		}
		return out, nil
	case dtypes.Int64:
		i64, err := tensors.CopyFlatData[int64](t)
		if err != nil {
			return nil, err
		}
		out := make([]float64, len(i64))
		for i, v := range i64 {
			out[i] = float64(v)
		}
		return out, nil
	default:
		return nil, errors.Errorf("unsupported tensor dtype %s for float64 conversion", t.DType())
	}
}

func copyAsBoolSlice(t *tensors.Tensor) ([]bool, error) {
	if t.DType() == dtypes.Bool {
		return tensors.CopyFlatData[bool](t)
	}
	return nil, errors.Errorf("observationMask must have Bool dtype, got %s", t.DType())
}

// Compile check that shapes is used if needed.
var _ = shapes.Make
