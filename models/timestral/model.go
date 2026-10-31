// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package timestral

import (
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/safetensors"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Model holds the loaded configuration and reference to the Hugging Face repository.
type Model struct {
	Repo   *hub.Repo
	Config *Config

	totalParameters *int64
	totalBytes      *int64
}

// LoadModel reads and validates config.json from repo (supports remote, local, and embedded repos).
func LoadModel(repo *hub.Repo) (*Model, error) {
	if repo == nil {
		return nil, errors.New("timestral.LoadModel: repo cannot be nil")
	}
	data, err := repo.ReadFile("config.json")
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read config.json from repo %q", repo.ID)
	}
	cfg, err := ParseConfig(data)
	if err != nil {
		return nil, err
	}
	return &Model{
		Repo:   repo,
		Config: cfg,
	}, nil
}

// NewModel creates a Model from an in-memory Config (useful for tests or custom models).
func NewModel(cfg *Config) (*Model, error) {
	if cfg == nil {
		return nil, errors.New("timestral.NewModel: cfg cannot be nil")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &Model{Config: cfg}, nil
}

// Description returns a human-readable summary of the model architecture and parameter count.
func (m *Model) Description() string {
	var sb strings.Builder
	repoID := "<in-memory>"
	if m.Repo != nil {
		repoID = m.Repo.ID
	}
	c := m.Config
	fmt.Fprintf(&sb, "Timestral Model (%s): d_model=%d, heads=%d, mlp_hidden=%d, layers=%d, patch_size=%d, block_size=%d, context_length=%d, quantiles=%d",
		repoID, c.DModel, c.NumHeads, c.MLPHiddenDim, len(c.BlockSchedule), c.PatchSize, c.BlockSize, c.ContextLength, c.NumQuantiles())
	if m.totalParameters != nil && m.totalBytes != nil {
		fmt.Fprintf(&sb, " [%s params, %s]", humanize.Count(*m.totalParameters), humanize.Bytes(*m.totalBytes))
	}
	return sb.String()
}

// LoadStore streams model.safetensors from m.Repo into store, transposing 2D Linear weights
// to GoMLX's [in_features, out_features] convention, permuting wQKV weights to [in, 3, numHeads, headDim],
// and organizing variables into scopes matching layers.go.
func (m *Model) LoadStore(backend compute.Backend, store *model.Store) error {
	if m.Repo == nil {
		return errors.New("timestral.Model.LoadStore: Model has no Repo")
	}
	var totalParams int64
	var totalBytes int64

	for tensorAndName, err := range safetensors.IterTensorsFromRepo(backend, m.Repo) {
		if err != nil {
			return errors.WithMessagef(err, "failed loading variables of Timestral model %q", m.Repo.ID)
		}
		shape := tensorAndName.Tensor.Shape()
		totalParams += int64(shape.Size())
		totalBytes += int64(shape.ByteSize())

		if err := m.loadSingleTensor(backend, store, tensorAndName.Name, tensorAndName.Tensor); err != nil {
			tensorAndName.Tensor.FinalizeAll()
			return err
		}
	}

	klog.V(1).Infof("Loaded %s parameters (%s bytes) for Timestral model %q",
		humanize.Count(totalParams), humanize.Bytes(totalBytes), m.Repo.ID)
	m.totalParameters = &totalParams
	m.totalBytes = &totalBytes
	return nil
}

// loadSingleTensor maps a single PyTorch safetensors parameter into the GoMLX store.
func (m *Model) loadSingleTensor(backend compute.Backend, store *model.Store, name string, t *tensors.Tensor) error {
	scopePath, varName, isWQKV, ok := mapTimestralTensorName(name)
	if !ok {
		klog.V(1).Infof("Skipping unmapped Timestral tensor: %s", name)
		t.FinalizeAll()
		return nil
	}

	subScope := store.RootScope()
	for _, part := range scopePath {
		subScope = subScope.In("%s", part)
	}

	numHeads := m.Config.NumHeads
	headDim := m.Config.HeadDim()

	var finalTensor *tensors.Tensor
	var err error
	if isWQKV && varName == "weights" {
		// PyTorch wQKV.weight is [3 * headDim * numHeads, D].
		// Reshape to [3, headDim, numHeads, D], transpose (3, 0, 2, 1) -> [D, 3, numHeads, headDim],
		// and reshape to [D, 3 * numHeads * headDim].
		finalTensor, err = transformTensor(backend, t, func(x *Node) *Node {
			d := x.Shape().Dimensions[1]
			x4D := Reshape(x, 3, headDim, numHeads, d)
			xPerm := TransposeAllAxes(x4D, 3, 0, 2, 1)
			return Reshape(xPerm, d, 3*numHeads*headDim)
		})
		t.FinalizeAll()
		if err != nil {
			return errors.Wrapf(err, "failed to permute wQKV weight %q", name)
		}
	} else if isWQKV && varName == "biases" {
		// PyTorch wQKV.bias is [3 * headDim * numHeads].
		// Reshape to [3, headDim, numHeads], transpose (0, 2, 1) -> [3, numHeads, headDim],
		// and reshape to [3 * numHeads * headDim].
		finalTensor, err = transformTensor(backend, t, func(x *Node) *Node {
			x3D := Reshape(x, 3, headDim, numHeads)
			xPerm := TransposeAllAxes(x3D, 0, 2, 1)
			return Reshape(xPerm, 3*numHeads*headDim)
		})
		t.FinalizeAll()
		if err != nil {
			return errors.Wrapf(err, "failed to permute wQKV bias %q", name)
		}
	} else if varName == "weights" && t.Rank() == 2 {
		// Standard 2D Linear weight: transpose [out, in] -> [in, out].
		finalTensor, err = transformTensor(backend, t, func(x *Node) *Node {
			return TransposeAllAxes(x, 1, 0)
		})
		t.FinalizeAll()
		if err != nil {
			return errors.Wrapf(err, "failed to transpose 2D weight %q", name)
		}
	} else {
		finalTensor = t
	}

	subScope.VariableWithValue(varName, finalTensor)
	return nil
}

func transformTensor(backend compute.Backend, t *tensors.Tensor, fn func(x *Node) *Node) (*tensors.Tensor, error) {
	exec, err := NewExec(backend, fn)
	if err != nil {
		return nil, err
	}
	defer exec.Finalize()
	res, err := exec.Call(t)
	if err != nil {
		return nil, err
	}
	return res[0], nil
}

// mapTimestralTensorName converts a safetensors parameter name to a GoMLX scope path and variable name.
func mapTimestralTensorName(safetensorsName string) (scopePath []string, varName string, isWQKV bool, ok bool) {
	parts := strings.Split(safetensorsName, ".")
	if len(parts) < 2 {
		return nil, "", false, false
	}

	last := parts[len(parts)-1]
	parent := parts[len(parts)-2]
	prefix := slices.Clone(parts[:len(parts)-1])

	if safetensorsName == "representation.forecast_query" {
		return []string{"representation"}, "forecast_query", false, true
	}

	if last == "scale" && (strings.Contains(parent, "norm") || (len(parts) >= 3 && strings.Contains(parts[len(parts)-3], "norm"))) {
		return append(prefix, "rms_norm"), "scale", false, true
	}

	if last == "weight" {
		return append(prefix, "dense"), "weights", parent == "wQKV", true
	}
	if last == "bias" {
		return append(prefix, "dense"), "biases", parent == "wQKV", true
	}

	return prefix, last, false, true
}

// Scale performs causal per-patch normalization of packed [R, T] rows.
// Returns (scaled, loc, scale), each shaped [R, T].
func (m *Model) Scale(inputs, observed *Node) (scaled, loc, scale *Node) {
	c := m.Config
	loc, scale = CausalPatchLocScale(inputs, observed, c.PatchSize, c.MinimumScale, c.StabilizeWithGlobal)
	scaled = ApplyAffineAsinhMasked(inputs, loc, scale, observed, c.UseAsinh)
	return scaled, loc, scale
}

// knownCovScaling normalizes known-future covariate rows with non-causal whole-row statistics.
// Indicator rows (values in {0, 1}) skip the asinh squash.
func (m *Model) knownCovScaling(inputs, observed, scaled, loc, scale, knownRows *Node) (*Node, *Node, *Node) {
	if knownRows == nil {
		return scaled, loc, scale
	}
	c := m.Config
	locC, scaleC, isBinary := InstanceNormStats(inputs, observed, c.MinimumScale)
	locCBroad := BroadcastLike(locC, inputs)
	scaleCBroad := BroadcastLike(scaleC, inputs)
	knownScaled := ApplyAffineAsinhMasked(inputs, locCBroad, scaleCBroad, observed, false)
	if c.UseAsinh {
		squashed := Asinh(knownScaled)
		useSquash := LogicalAnd(observed, BroadcastLike(LogicalNot(isBinary), observed))
		knownScaled = Where(useSquash, squashed, knownScaled)
	}
	sel := BroadcastLike(InsertAxes(knownRows, -1), inputs)
	return Where(sel, knownScaled, scaled), Where(sel, locCBroad, loc), Where(sel, scaleCBroad, scale)
}

func (m *Model) runTwoStream(scope *model.Scope, xEnc, xDec, timeMask, variateMask, positions *Node) *Node {
	x := Concatenate([]*Node{xEnc, xDec}, 1)
	encLen := xEnc.Shape().Dimensions[1]
	totalLen := x.Shape().Dimensions[1]

	var tsMask *TwoStreamMask
	var plainTimeMask *Node
	if m.Config.SplitContextHorizonAttn {
		tsMask = &TwoStreamMask{Mask: timeMask, EncLen: encLen}
	} else {
		plainTimeMask = timeMask
	}

	h := Transformer(scope.In("transformer"), x, m.Config, plainTimeMask, tsMask, variateMask, positions)
	decOut := SliceAxis(h, 1, AxisRange(encLen, totalLen))
	return QuantileHead(scope.In("head"), decOut, m.Config)
}

// BlockBuilder configures a single-block DHF forward pass over packed [R, T] rows.
type BlockBuilder struct {
	model                *Model
	scope                *model.Scope
	inputs               *Node
	observed             *Node
	horizonPatches       int
	groupIDs             *Node
	roleIDs              *Node
	knownFuture          *Node
	knownFutureMask      *Node
	refineAdvancingFrame bool
}

// NewBlockBuilder starts configuring a single-block DHF prediction over packed [R, T] rows.
//   - inputs: float32 [R, T] where T is a multiple of PatchSize.
//   - observed: bool [R, T] observation mask.
//   - horizonPatches: number of patches h to decode in this block (h >= 1).
func (m *Model) NewBlockBuilder(scope *model.Scope, inputs, observed *Node, horizonPatches int) *BlockBuilder {
	return &BlockBuilder{
		model:          m,
		scope:          scope,
		inputs:         inputs,
		observed:       observed,
		horizonPatches: horizonPatches,
	}
}

// WithGroupIDs sets the [R] int32 series ID per row (rows with the same group ID attend each other).
// If not set, defaults to 0..R-1 (each row in its own group).
func (b *BlockBuilder) WithGroupIDs(groupIDs *Node) *BlockBuilder {
	b.groupIDs = groupIDs
	return b
}

// WithCovariates sets the row role IDs [R] and optional known-future covariate horizon values/mask [R, h*PatchSize].
func (b *BlockBuilder) WithCovariates(roleIDs, knownFuture, knownFutureMask *Node) *BlockBuilder {
	b.roleIDs = roleIDs
	b.knownFuture = knownFuture
	b.knownFutureMask = knownFutureMask
	return b
}

// RefineAdvancingFrame enables/disables re-denormalizing patches 1..h-1 in the advancing Welford frame.
func (b *BlockBuilder) RefineAdvancingFrame(enabled bool) *BlockBuilder {
	b.refineAdvancingFrame = enabled
	return b
}

// Done constructs the single-block DHF computation graph and returns quantiles of shape [R, h * PatchSize, Q].
func (b *BlockBuilder) Done() *Node {
	m := b.model
	c := m.Config
	g := b.inputs.Graph()
	rows := b.inputs.Shape().Dimensions[0]
	t := b.inputs.Shape().Dimensions[1]
	ps := c.PatchSize
	h := b.horizonPatches

	if t%ps != 0 {
		exceptions.Panicf("timestral: context length %d must be a multiple of patch_size %d", t, ps)
	}
	if h < 1 {
		exceptions.Panicf("timestral: horizonPatches must be >= 1, got %d", h)
	}

	groupIDs := b.groupIDs
	if groupIDs == nil {
		groupIDs = Iota(g, shapes.Make(dtypes.Int32, rows), 0)
	}

	var quantiles, ctxLoc, ctxScale, targetRows *Node
	if b.roleIDs == nil || !c.CovariateEnabled {
		quantiles, ctxLoc, ctxScale = m.predictContextOnly(b.scope, b.inputs, b.observed, groupIDs, h)
	} else {
		horizonLen := h * ps
		kf := b.knownFuture
		kfm := b.knownFutureMask
		if kf == nil {
			kf = Zeros(g, shapes.Make(b.inputs.DType(), rows, horizonLen))
			kfm = BroadcastToDims(Const(g, false), rows, horizonLen)
		}
		fullInputs := Concatenate([]*Node{b.inputs, kf}, -1)
		fullObserved := Concatenate([]*Node{b.observed, kfm}, -1)
		quantiles = m.predictWindow(b.scope, fullInputs, fullObserved, groupIDs, h, b.roleIDs)
		targetRows = Equal(b.roleIDs, Scalar(g, b.roleIDs.DType(), RoleTarget))
	}

	if b.refineAdvancingFrame && h > 1 {
		if ctxLoc == nil || ctxScale == nil {
			_, ctxLoc, ctxScale = m.Scale(b.inputs, b.observed)
		}
		quantiles = m.refineBlock(quantiles, b.inputs, b.observed, ctxLoc, ctxScale, targetRows)
	}
	return quantiles
}

func (m *Model) predictContextOnly(scope *model.Scope, inputs, observed, groupIDs *Node, horizonPatches int) (quantiles, loc, scale *Node) {
	c := m.Config
	g := inputs.Graph()
	ps := c.PatchSize
	rows := inputs.Shape().Dimensions[0]
	t := inputs.Shape().Dimensions[1]
	ll := t / ps
	h := horizonPatches

	scaled, loc, scale := m.Scale(inputs, observed)
	repScope := scope.In("representation")
	xEnc := PatchRepresentation(repScope, scaled, observed, ps, c.DModel)
	xDec := ForecastTokens(repScope, g, scaled.DType(), rows, h, c.DModel)

	obsPerPatch := ReduceLogicalOr(Reshape(observed, rows, ll, ps), -1)
	keyValid := ComputeEncKeyValidity(obsPerPatch, c.AttendMissingPatches)
	timeMask := BuildForecastTimeMask(ll, h, keyValid)
	variateMask := BuildTwoStreamVariateMask(groupIDs, keyValid, h, nil, nil)
	firstObs := RowFirstObservedPatch(obsPerPatch, groupIDs)
	positions := BuildForecastPositions(ll, h, firstObs)

	preds := m.runTwoStream(scope, xEnc, xDec, timeMask, variateMask, positions)
	anchorLoc := Slice(Reshape(loc, rows, ll, ps), AxisRange(), AxisRange(ll-1, ll), AxisRange(0, 1))
	anchorScale := Slice(Reshape(scale, rows, ll, ps), AxisRange(), AxisRange(ll-1, ll), AxisRange(0, 1))
	quantiles = DenormalizeQuantiles(preds, anchorLoc, anchorScale, c.UseAsinh)
	return quantiles, loc, scale
}

func (m *Model) predictWindow(scope *model.Scope, fullInputs, fullObserved, groupIDs *Node, horizonPatches int, roleIDs *Node) *Node {
	c := m.Config
	g := fullInputs.Graph()
	ps := c.PatchSize
	rows := fullInputs.Shape().Dimensions[0]
	w := fullInputs.Shape().Dimensions[1]
	h := horizonPatches
	n := w / ps
	ll := n - h

	targetRows := Equal(roleIDs, Scalar(g, roleIDs.DType(), RoleTarget))
	knownRows := Equal(roleIDs, Scalar(g, roleIDs.DType(), RoleKnownCov))

	scaled, loc, scale := m.Scale(fullInputs, fullObserved)
	scaled, loc, scale = m.knownCovScaling(fullInputs, fullObserved, scaled, loc, scale, knownRows)
	repScope := scope.In("representation")
	xEnc := PatchRepresentation(repScope, scaled, fullObserved, ps, c.DModel)
	xDec := ForecastTokens(repScope, g, scaled.DType(), rows, n, c.DModel)

	obsPerPatch := ReduceLogicalOr(Reshape(fullObserved, rows, n, ps), -1)
	keyValid := ComputeEncKeyValidity(obsPerPatch, c.AttendMissingPatches)
	sRows := BroadcastToDims(Scalar(g, dtypes.Int32, int32(ll)), rows)
	timeMask := BuildTwoStreamTimeMask(sRows, n, c.BlockSize, keyValid, knownRows)
	variateMask := BuildTwoStreamVariateMask(groupIDs, keyValid, n, targetRows, knownRows)
	firstObs := RowFirstObservedPatch(obsPerPatch, groupIDs)
	positions := BuildTwoStreamPositions(n, firstObs)

	predsAll := m.runTwoStream(scope, xEnc, xDec, timeMask, variateMask, positions)
	numQ := c.NumQuantiles()
	predsUnflat := Reshape(predsAll, rows, n, ps, numQ)
	horizonPreds := Reshape(SliceAxis(predsUnflat, 1, AxisRange(ll, n)), rows, h*ps, numQ)

	anchorLoc := Slice(Reshape(loc, rows, n, ps), AxisRange(), AxisRange(ll-1, ll), AxisRange(0, 1))
	anchorScale := Slice(Reshape(scale, rows, n, ps), AxisRange(), AxisRange(ll-1, ll), AxisRange(0, 1))
	return DenormalizeQuantiles(horizonPreds, anchorLoc, anchorScale, c.UseAsinh)
}

// refineBlock moves one decoded block from the anchor frame of the last context patch
// to the advancing refine frame it was trained in.
func (m *Model) refineBlock(quantiles, context, mask, loc, scale, targetRows *Node) *Node {
	c := m.Config
	g := quantiles.Graph()
	ps := c.PatchSize
	rows := quantiles.Shape().Dimensions[0]
	nPred := quantiles.Shape().Dimensions[1] / ps
	if nPred <= 1 {
		return quantiles
	}

	ctxLen := loc.Shape().Dimensions[1]
	loc0 := SliceAxis(loc, 1, AxisRange(ctxLen-1, ctxLen))     // [R, 1]
	scale0 := SliceAxis(scale, 1, AxisRange(ctxLen-1, ctxLen)) // [R, 1]

	medIdx := c.MedianIndex()
	median := Reshape(SliceAxis(quantiles, -1, AxisRange(medIdx, medIdx+1)), rows, nPred, ps)
	median64 := ConvertDType(median, dtypes.Float64)
	loc064 := ConvertDType(InsertAxes(loc0, -1), dtypes.Float64)     // [R, 1, 1]
	scale064 := ConvertDType(InsertAxes(scale0, -1), dtypes.Float64) // [R, 1, 1]
	sinhMedian := Div(Sub(median64, loc064), scale064)

	locJ64, scaleJ64 := AdvancingBlockFrames(context, mask, sinhMedian, c.MinimumScale)
	locJ := ConvertDType(locJ64, quantiles.DType())     // [R, nPred]
	scaleJ := ConvertDType(scaleJ64, quantiles.DType()) // [R, nPred]

	// Replace patch 0 with (loc0, scale0) and repeat_interleave(ps, dim=-1).
	firstMask := Equal(BroadcastToDims(Iota(g, shapes.Make(dtypes.Int32, 1, nPred), 1), rows, nPred), ScalarZero(g, dtypes.Int32))
	loc0Broad := BroadcastToDims(loc0, rows, nPred)
	scale0Broad := BroadcastToDims(scale0, rows, nPred)
	locJ = Where(firstMask, loc0Broad, locJ)
	scaleJ = Where(firstMask, scale0Broad, scaleJ)

	locJSteps := Reshape(BroadcastToDims(InsertAxes(locJ, -1), rows, nPred, ps), rows, nPred*ps, 1)
	scaleJSteps := Reshape(BroadcastToDims(InsertAxes(scaleJ, -1), rows, nPred, ps), rows, nPred*ps, 1)

	loc03D := InsertAxes(loc0, -1)     // [R, 1, 1]
	scale03D := InsertAxes(scale0, -1) // [R, 1, 1]
	sinhNorm := Div(Sub(quantiles, loc03D), scale03D)
	refined := ClampNonFinite(Add(Mul(sinhNorm, scaleJSteps), locJSteps))
	if targetRows != nil {
		sel := BroadcastLike(Reshape(targetRows, rows, 1, 1), refined)
		refined = Where(sel, refined, quantiles)
	}
	return refined
}

// FFillWithMask forward-fills unobserved positions along the last axis; leading unobserved positions become 0.
func FFillWithMask(inputs, mask *Node) *Node {
	g := inputs.Graph()
	origShape := inputs.Shape()
	rank := origShape.Rank()
	t := origShape.Dimensions[rank-1]
	r := origShape.Size() / t

	inputsFlat := Reshape(inputs, r*t)
	mask2D := Reshape(mask, r, t)

	idx := Iota(g, shapes.Make(dtypes.Int32, r, t), 1)
	idx = Where(mask2D, idx, ZerosLike(idx))
	idx = CumMax(idx, -1)

	rowOffsets := MulScalar(Iota(g, shapes.Make(dtypes.Int32, r, t), 0), t)
	flatIdx := InsertAxes(Add(rowOffsets, idx), -1) // [R, T, 1]
	gathered := Gather(inputsFlat, flatIdx, true)
	gathered = Reshape(gathered, origShape.Dimensions...)
	return Where(IsFinite(gathered), gathered, ZerosLike(gathered))
}

// PadToPatch left-pads inputs and mask along the last axis to a multiple of patchSize.
// Padding is 0 for inputs and false for mask.
func PadToPatch(inputs, mask *Node, patchSize int) (*Node, *Node) {
	g := inputs.Graph()
	rank := inputs.Rank()
	ctxLen := inputs.Shape().Dimensions[rank-1]
	if ctxLen%patchSize == 0 {
		return inputs, mask
	}
	padLen := patchSize - (ctxLen % patchSize)
	padAxes := make([]PadAxis, rank)
	padAxes[rank-1] = PadAxis{Start: padLen}
	inputs = Pad(inputs, ScalarZero(g, inputs.DType()), padAxes...)
	mask = Pad(mask, Const(g, false), padAxes...)
	return inputs, mask
}

// ReplaceExtremeValues replaces non-finite values or values with |x| >= threshold with replacement.
func ReplaceExtremeValues(x *Node, threshold, replacement float64) *Node {
	g := x.Graph()
	valid := LogicalAnd(IsFinite(x), LessThan(Abs(x), Scalar(g, x.DType(), threshold)))
	return Where(valid, x, BroadcastLike(Scalar(g, x.DType(), replacement), x))
}

// NonNegativeContext returns a [B, V] boolean mask indicating whether every observed context
// value is non-negative (and at least one observed value exists).
func NonNegativeContext(inputs, observationMask *Node) *Node {
	g := inputs.Graph()
	observed := LogicalAnd(IsFinite(inputs), observationMask)
	anyObs := ReduceLogicalOr(observed, -1)
	nonNegOrUnobs := LogicalOr(GreaterOrEqual(inputs, ScalarZero(g, inputs.DType())), LogicalNot(observed))
	allNonNeg := ReduceLogicalAnd(nonNegOrUnobs, -1)
	return LogicalAnd(anyObs, allNonNeg)
}

// CapQuantiles clamps quantiles [B, V, H, Q] to [ctxMin - k*scaleLast, ctxMax + k*scaleLast]
// of the observed context [B, V, T].
func (m *Model) CapQuantiles(quantiles, inputs, observationMask *Node, quantileRealCapK float64) *Node {
	if quantileRealCapK <= 0 {
		return quantiles
	}
	g := quantiles.Graph()
	dtype := quantiles.DType()
	posInf := BroadcastLike(Scalar(g, dtype, math.Inf(1)), inputs)
	negInf := BroadcastLike(Scalar(g, dtype, math.Inf(-1)), inputs)

	ctxMax := ReduceAndKeep(Where(observationMask, inputs, negInf), ReduceMax, -1)
	ctxMax = Where(IsFinite(ctxMax), ctxMax, ZerosLike(ctxMax))

	ctxMin := ReduceAndKeep(Where(observationMask, inputs, posInf), ReduceMin, -1)
	ctxMin = Where(IsFinite(ctxMin), ctxMin, ZerosLike(ctxMin))

	c := m.Config
	_, scale := CausalPatchLocScale(inputs, observationMask, c.PatchSize, c.MinimumScale, c.StabilizeWithGlobal)
	t := scale.Shape().Dimensions[scale.Rank()-1]
	anchorScale := MulScalar(SliceAxis(scale, -1, AxisRange(t-1, t)), quantileRealCapK)

	lo := BroadcastLike(InsertAxes(Sub(ctxMin, anchorScale), -1), quantiles)
	hi := BroadcastLike(InsertAxes(Add(ctxMax, anchorScale), -1), quantiles)
	return Clamp(lo, quantiles, hi)
}

// GraphBuilder configures and builds a Timestral forecast computation graph.
// Created via Model.BuildGraph(scope, context, horizonSteps).
type GraphBuilder struct {
	model                  *Model
	scope                  *model.Scope
	context                *Node
	observationMask        *Node
	horizonSteps           int
	pastCovariates         *Node
	knownCovariates        *Node
	maxContext             int
	perShotHorizonPatches  int
	quantileRealCapK       float64
	extremeValueThreshold  float64
	symmetricAveraging     bool
	clipNonNegative        bool
}

// BuildGraph starts building a Timestral forecast graph for the given context and horizonSteps.
//
//   - scope: GoMLX model scope containing the loaded Timestral variables.
//   - context: float32 *graph.Node shaped [T], [B, T], or [B, V, T]. NaN marks missing values.
//   - horizonSteps: number of future time steps H to forecast (H >= 1).
func (m *Model) BuildGraph(scope *model.Scope, context *Node, horizonSteps int) *GraphBuilder {
	perShot := min(32, m.Config.BlockSize)
	return &GraphBuilder{
		model:                 m,
		scope:                 scope,
		context:               context,
		horizonSteps:          horizonSteps,
		maxContext:            m.Config.ContextLength,
		perShotHorizonPatches: perShot,
		quantileRealCapK:      1e4,
		extremeValueThreshold: 1e30,
		symmetricAveraging:    false,
		clipNonNegative:       false,
	}
}

// WithObservationMask provides an explicit boolean observation mask shaped like context (false = missing).
func (b *GraphBuilder) WithObservationMask(mask *Node) *GraphBuilder {
	b.observationMask = mask
	return b
}

// WithPastCovariates adds past-only numeric covariates shaped [B, C_past, T] or [B, T].
func (b *GraphBuilder) WithPastCovariates(past *Node) *GraphBuilder {
	b.pastCovariates = past
	return b
}

// WithKnownCovariates adds known-future numeric covariates shaped [B, C_known, T + horizonSteps] or [B, T + horizonSteps].
func (b *GraphBuilder) WithKnownCovariates(known *Node) *GraphBuilder {
	b.knownCovariates = known
	return b
}

// MaxContext sets the maximum context length in time steps (defaults to Config.ContextLength).
func (b *GraphBuilder) MaxContext(steps int) *GraphBuilder {
	b.maxContext = steps
	return b
}

// PerShotHorizonPatches sets the maximum patches decoded per autoregressive block (defaults to min(32, Config.BlockSize)).
func (b *GraphBuilder) PerShotHorizonPatches(patches int) *GraphBuilder {
	b.perShotHorizonPatches = min(patches, b.model.Config.BlockSize)
	return b
}

// QuantileRealCapK sets the real-space quantile clamp factor k (defaults to 1e4; 0 disables).
func (b *GraphBuilder) QuantileRealCapK(k float64) *GraphBuilder {
	b.quantileRealCapK = k
	return b
}

// ExtremeValueThreshold sets the magnitude threshold above which fed-back medians are replaced by 0 (defaults to 1e30).
func (b *GraphBuilder) ExtremeValueThreshold(t float64) *GraphBuilder {
	b.extremeValueThreshold = t
	return b
}

// SymmetricAveraging enables/disables forecasting both context and -context and averaging
// with the reflected quantiles inside the graph.
func (b *GraphBuilder) SymmetricAveraging(enabled bool) *GraphBuilder {
	b.symmetricAveraging = enabled
	return b
}

// ClipNonNegative enables/disables clamping quantiles at 0 for series with non-negative context.
func (b *GraphBuilder) ClipNonNegative(enabled bool) *GraphBuilder {
	b.clipNonNegative = enabled
	return b
}

// Done constructs the GoMLX computation graph and returns the quantiles *Node:
//   - [H, Q] if context was 1D [T]
//   - [B, H, Q] if context was 2D [B, T]
//   - [B, V, H, Q] if context was 3D [B, V, T]
func (b *GraphBuilder) Done() *Node {
	if b.horizonSteps < 1 {
		exceptions.Panicf("timestral.GraphBuilder: horizonSteps must be >= 1, got %d", b.horizonSteps)
	}
	origRank := b.context.Rank()
	inputs, mask := b.normalizeContextInput(b.context, b.observationMask)
	batchSize := inputs.Shape().Dimensions[0]
	numVariates := inputs.Shape().Dimensions[1]
	ctxLen := inputs.Shape().Dimensions[2]

	past := b.normalizeCovariateInput(b.pastCovariates, batchSize, ctxLen, "pastCovariates")
	known := b.normalizeCovariateInput(b.knownCovariates, batchSize, ctxLen+b.horizonSteps, "knownCovariates")
	if known != nil && !b.model.Config.SupportsKnownCovariates() {
		exceptions.Panicf("timestral.GraphBuilder: checkpoint cannot condition on known-future covariates")
	}

	var quantiles *Node
	if b.symmetricAveraging {
		if !b.model.Config.AreQuantilesSymmetric() {
			exceptions.Panicf("timestral.GraphBuilder: symmetric averaging requires quantile knots symmetric around 0.5")
		}
		var negPast, negKnown *Node
		if past != nil {
			negPast = Neg(past)
		}
		if known != nil {
			negKnown = Neg(known)
		}
		posQ := b.forecastRawGraph(inputs, mask, past, known)
		negQ := b.forecastRawGraph(Neg(inputs), mask, negPast, negKnown)
		reflected := Reverse(Neg(negQ), -1)
		quantiles = MulScalar(Add(posQ, reflected), 0.5)
	} else {
		quantiles = b.forecastRawGraph(inputs, mask, past, known)
	}

	if b.clipNonNegative {
		eligible := NonNegativeContext(inputs, mask) // [B, V]
		elig4D := BroadcastLike(Reshape(eligible, batchSize, numVariates, 1, 1), quantiles)
		quantiles = Where(elig4D, MaxScalar(quantiles, 0.0), quantiles)
	}

	numQ := b.model.Config.NumQuantiles()
	switch origRank {
	case 1:
		return Reshape(quantiles, b.horizonSteps, numQ)
	case 2:
		return Reshape(quantiles, batchSize, b.horizonSteps, numQ)
	default:
		return quantiles
	}
}

func (b *GraphBuilder) normalizeContextInput(ctx, userMask *Node) (inputs, mask *Node) {
	x := ctx
	switch x.Rank() {
	case 1:
		x = Reshape(x, 1, 1, x.Shape().Dimensions[0])
	case 2:
		x = Reshape(x, x.Shape().Dimensions[0], 1, x.Shape().Dimensions[1])
	case 3:
	default:
		exceptions.Panicf("timestral: context must have rank 1 [T], 2 [B, T], or 3 [B, V, T], got %s", x.Shape())
	}
	if x.DType() != dtypes.Float32 {
		x = ConvertDType(x, dtypes.Float32)
	}
	mask = IsFinite(x)
	if userMask != nil {
		um := Reshape(userMask, x.Shape().Dimensions...)
		mask = LogicalAnd(mask, um)
	}
	inputs = Where(mask, x, ZerosLike(x))
	return inputs, mask
}

func (b *GraphBuilder) normalizeCovariateInput(cov *Node, batch, length int, name string) *Node {
	if cov == nil {
		return nil
	}
	if !b.model.Config.CovariateEnabled {
		exceptions.Panicf("timestral: checkpoint was not trained with covariates; got %s", name)
	}
	v := cov
	if v.Rank() == 2 {
		v = InsertAxes(v, 1)
	}
	if v.Rank() != 3 || v.Shape().Dimensions[0] != batch || v.Shape().Dimensions[2] != length {
		exceptions.Panicf("timestral: %s must have shape [B=%d, C, %d], got %s", name, batch, length, v.Shape())
	}
	if v.DType() != dtypes.Float32 {
		v = ConvertDType(v, dtypes.Float32)
	}
	return v
}

func (b *GraphBuilder) forecastRawGraph(inputs, observationMask, pastCovariates, knownCovariates *Node) *Node {
	rows, mask, roleIDs, knownFuture, knownFutureMask, nTgt := b.assembleRowsGraph(
		inputs, observationMask, pastCovariates, knownCovariates,
	)
	quantiles := b.blockARDecodeGraph(rows, mask, roleIDs, knownFuture, knownFutureMask)
	quantiles = SliceAxis(quantiles, 1, AxisRange(0, nTgt))
	if b.quantileRealCapK > 0 {
		tgtRows := SliceAxis(rows, 1, AxisRange(0, nTgt))
		tgtMask := SliceAxis(mask, 1, AxisRange(0, nTgt))
		quantiles = b.model.CapQuantiles(quantiles, tgtRows, tgtMask, b.quantileRealCapK)
	}
	return quantiles
}

func (b *GraphBuilder) assembleRowsGraph(
	inputs, observationMask, pastCovariates, knownCovariates *Node,
) (rows, mask, roleIDs, knownFuture, knownFutureMask *Node, nTgt int) {
	g := inputs.Graph()
	ps := b.model.Config.PatchSize
	batchSize := inputs.Shape().Dimensions[0]
	nTgt = inputs.Shape().Dimensions[1]
	ctx0 := inputs.Shape().Dimensions[2]
	horizonPatches := (b.horizonSteps + ps - 1) / ps
	horizon := horizonPatches * ps

	ctxRows := []*Node{FFillWithMask(inputs, observationMask)}
	ctxMasks := []*Node{observationMask}
	roles := make([]int32, 0, nTgt)
	for range nTgt {
		roles = append(roles, RoleTarget)
	}

	nPast := 0
	if pastCovariates != nil {
		nPast = pastCovariates.Shape().Dimensions[1]
		pastObs := IsFinite(pastCovariates)
		ctxMasks = append(ctxMasks, pastObs)
		ctxRows = append(ctxRows, Where(pastObs, pastCovariates, ZerosLike(pastCovariates)))
		for range nPast {
			roles = append(roles, RolePastCov)
		}
	}

	nKnown := 0
	var knownFutureVals, knownFutureObs *Node
	if knownCovariates != nil {
		nKnown = knownCovariates.Shape().Dimensions[1]
		kcCtx := SliceAxis(knownCovariates, -1, AxisRange(0, ctx0))
		kcFut := SliceAxis(knownCovariates, -1, AxisRange(ctx0, ctx0+b.horizonSteps))
		kcCtxObs := IsFinite(kcCtx)
		ctxMasks = append(ctxMasks, kcCtxObs)
		ctxRows = append(ctxRows, Where(kcCtxObs, kcCtx, ZerosLike(kcCtx)))
		for range nKnown {
			roles = append(roles, RoleKnownCov)
		}
		knownFutureObs = IsFinite(kcFut)
		knownFutureVals = Where(knownFutureObs, kcFut, ZerosLike(kcFut))
	}

	rows = Concatenate(ctxRows, 1)
	mask = Concatenate(ctxMasks, 1)
	if rows.Shape().Dimensions[2] > b.maxContext {
		totalT := rows.Shape().Dimensions[2]
		rows = SliceAxis(rows, 2, AxisRange(totalT-b.maxContext, totalT))
		mask = SliceAxis(mask, 2, AxisRange(totalT-b.maxContext, totalT))
	}
	rows, mask = PadToPatch(rows, mask, ps)
	vAll := rows.Shape().Dimensions[1]
	if nPast == 0 && nKnown == 0 {
		return rows, mask, nil, nil, nil, nTgt
	}

	repeatedRoles := make([]int32, 0, batchSize*vAll)
	for range batchSize {
		repeatedRoles = append(repeatedRoles, roles...)
	}
	roleIDs = Const(g, repeatedRoles)

	if nKnown > 0 {
		// Pad knownFutureVals [B, nKnown, horizonSteps] to [B, vAll, horizon]:
		// Channels: pad Start=nTgt+nPast, End=0; Time: pad Start=0, End=horizon-horizonSteps.
		padAxes := []PadAxis{
			{},
			{Start: nTgt + nPast},
			{End: horizon - b.horizonSteps},
		}
		kf := Pad(knownFutureVals, ScalarZero(g, rows.DType()), padAxes...)
		kfm := Pad(knownFutureObs, Const(g, false), padAxes...)
		knownFuture = Reshape(kf, batchSize*vAll, horizon)
		knownFutureMask = Reshape(kfm, batchSize*vAll, horizon)
	}
	return rows, mask, roleIDs, knownFuture, knownFutureMask, nTgt
}

func (b *GraphBuilder) blockARDecodeGraph(
	inputs, observationMask, roleIDs, knownFuture, knownFutureMask *Node,
) *Node {
	g := inputs.Graph()
	ps := b.model.Config.PatchSize
	batchSize := inputs.Shape().Dimensions[0]
	v := inputs.Shape().Dimensions[1]
	ctxLen := inputs.Shape().Dimensions[2]
	totalRows := batchSize * v

	context := Reshape(inputs, totalRows, ctxLen)
	mask := Reshape(observationMask, totalRows, ctxLen)
	groupIDsSlice := make([]int32, totalRows)
	for i := range batchSize {
		for j := range v {
			groupIDsSlice[i*v+j] = int32(i)
		}
	}
	groupIDs := Const(g, groupIDsSlice)

	var knownRows, pastRows *Node
	if roleIDs != nil {
		knownRows = Equal(roleIDs, Scalar(g, roleIDs.DType(), RoleKnownCov))
		targetRows := Equal(roleIDs, Scalar(g, roleIDs.DType(), RoleTarget))
		pastRows = LogicalAnd(LogicalNot(targetRows), LogicalNot(knownRows))
	}

	var blocks []*Node
	remaining := (b.horizonSteps + ps - 1) / ps
	offset := 0
	medIdx := b.model.Config.MedianIndex()

	for remaining > 0 {
		block := min(b.perShotHorizonPatches, remaining)
		blen := block * ps
		var kfBlock, kfmBlock *Node
		if knownFuture != nil {
			kfBlock = SliceAxis(knownFuture, 1, AxisRange(offset, offset+blen))
			kfmBlock = SliceAxis(knownFutureMask, 1, AxisRange(offset, offset+blen))
		}

		// Reuse the same model variable scope across AR steps and symmetric passes.
		blockScope := b.scope.ResetVisited()
		quantiles := b.model.NewBlockBuilder(blockScope, context, mask, block).
			WithGroupIDs(groupIDs).
			WithCovariates(roleIDs, kfBlock, kfmBlock).
			RefineAdvancingFrame(true).
			Done()
		blocks = append(blocks, quantiles)

		remaining -= block
		offset += blen
		if remaining <= 0 {
			break
		}

		medianPred := Reshape(SliceAxis(quantiles, -1, AxisRange(medIdx, medIdx+1)), totalRows, blen)
		feedback := ReplaceExtremeValues(medianPred, b.extremeValueThreshold, 0.0)
		feedbackMask := BroadcastToDims(Const(g, true), totalRows, blen)

		if roleIDs != nil {
			if kfBlock != nil {
				knownSel := BroadcastToDims(InsertAxes(knownRows, 1), totalRows, blen)
				feedback = Where(knownSel, kfBlock, feedback)
				feedbackMask = Where(knownSel, kfmBlock, feedbackMask)
			}
			pastSel := BroadcastToDims(InsertAxes(pastRows, 1), totalRows, blen)
			feedback = Where(pastSel, ZerosLike(feedback), feedback)
			feedbackMask = LogicalAnd(feedbackMask, LogicalNot(pastSel))
		}

		context = Concatenate([]*Node{context, feedback}, -1)
		mask = Concatenate([]*Node{mask, feedbackMask}, -1)
		if context.Shape().Dimensions[1] > b.maxContext {
			keep := max(ps, (b.maxContext/ps)*ps)
			curLen := context.Shape().Dimensions[1]
			context = SliceAxis(context, 1, AxisRange(curLen-keep, curLen))
			mask = SliceAxis(mask, 1, AxisRange(curLen-keep, curLen))
		}
	}

	var packed *Node
	if len(blocks) == 1 {
		packed = blocks[0]
	} else {
		packed = Concatenate(blocks, 1)
	}
	packed = SliceAxis(packed, 1, AxisRange(0, b.horizonSteps))
	return Reshape(packed, batchSize, v, b.horizonSteps, b.model.Config.NumQuantiles())
}
