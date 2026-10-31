// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package timestral

import (
	"math"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
)

const (
	// float32Eps is torch.finfo(torch.float32).eps (2^-23).
	float32Eps = 1.1920928955078125e-7

	// float64Tiny is torch.finfo(torch.float64).tiny (2^-1022).
	float64Tiny = 2.2250738585072014e-308
)

// ComputeCausalStatistics computes the causal running mean and std along the last axis
// using Welford's algorithm in float64, matching timestral.scaler.compute_causal_statistics.
//
// Returns (means, scales), both shaped like data and with data's dtype.
func ComputeCausalStatistics(data, observationMask *Node, minimumScale float64, stabilizeWithGlobal bool) (means, scales *Node) {
	g := data.Graph()
	origDType := data.DType()

	x := ConvertDType(data, dtypes.Float64)
	w := ConvertDType(observationMask, dtypes.Float64)

	weighted := Mul(w, x)
	cumW := CumSum(w, -1)
	cumX := CumSum(weighted, -1)
	denominator := MaxScalar(cumW, 1.0)
	means64 := Div(cumX, denominator)

	lastAxis := means64.Rank() - 1
	lastDim := means64.Shape().Dimensions[lastAxis]
	var shifted *Node
	if lastDim <= 1 {
		shifted = ZerosLike(means64)
	} else {
		padAxes := make([]PadAxis, means64.Rank())
		padAxes[lastAxis] = PadAxis{Start: 1}
		padded := Pad(SliceAxis(means64, -1, AxisRange(0, lastDim-1)), ScalarZero(g, dtypes.Float64), padAxes...)
		shifted = padded
	}

	increment := Mul(Mul(Sub(x, shifted), Sub(x, means64)), w)
	m2 := CumSum(increment, -1)
	variance := Div(m2, MaxScalar(AddScalar(denominator, -1.0), 1.0))
	scales64 := MaxScalar(Sqrt(MaxScalar(variance, 0.0)), minimumScale)

	if stabilizeWithGlobal {
		lo, hi := 1e-10, 1e10
		globalDen := MaxScalar(ReduceAndKeep(w, ReduceSum, -1), 1.0)
		globalMeans := Div(ReduceAndKeep(weighted, ReduceSum, -1), globalDen)
		globalMeans = Where(IsFinite(globalMeans), globalMeans, ZerosLike(globalMeans))
		globalVar := Div(ReduceAndKeep(Square(Mul(Sub(x, globalMeans), w)), ReduceSum, -1), globalDen)
		globalScale := BroadcastLike(Sqrt(AddScalar(globalVar, minimumScale)), scales64)
		floor := MaxScalar(MulScalar(globalScale, lo), minimumScale)
		scales64 = Clamp(floor, scales64, MulScalar(globalScale, hi))
	}

	return ConvertDType(means64, origDType), ConvertDType(scales64, origDType)
}

// CausalPatchLocScale computes per-patch (loc, scale) at the last time step of each patch
// and broadcasts them across all time steps of the patch.
//
// The last dimension of data must be a multiple of patchSize.
func CausalPatchLocScale(data, observationMask *Node, patchSize int, minimumScale float64, stabilizeWithGlobal bool) (loc, scale *Node) {
	means, scales := ComputeCausalStatistics(data, observationMask, minimumScale, stabilizeWithGlobal)
	origShape := data.Shape()
	rank := origShape.Rank()
	totalSteps := origShape.Dimensions[rank-1]
	numPatches := totalSteps / patchSize

	unfoldedDims := make([]int, rank+1)
	copy(unfoldedDims, origShape.Dimensions[:rank-1])
	unfoldedDims[rank-1] = numPatches
	unfoldedDims[rank] = patchSize

	meansUnfolded := Reshape(means, unfoldedDims...)
	scalesUnfolded := Reshape(scales, unfoldedDims...)

	// Take the last step of each patch: [..., numPatches, 1] and broadcast to [..., numPatches, patchSize].
	patchMeans := SliceAxis(meansUnfolded, -1, AxisRange(patchSize-1, patchSize))
	patchScales := SliceAxis(scalesUnfolded, -1, AxisRange(patchSize-1, patchSize))

	loc = Reshape(BroadcastToDims(patchMeans, unfoldedDims...), origShape.Dimensions...)
	scale = Reshape(BroadcastToDims(patchScales, unfoldedDims...), origShape.Dimensions...)
	return loc, scale
}

// ApplyAffineAsinhMasked computes asinh((data - loc) / scale) on observed positions,
// while unobserved positions become 0.
func ApplyAffineAsinhMasked(data, loc, scale, observationMask *Node, useAsinh bool) *Node {
	scaled := Div(Sub(data, loc), MaxScalar(scale, float32Eps))
	scaled = Where(observationMask, scaled, ZerosLike(scaled))
	if useAsinh {
		scaled = Where(observationMask, Asinh(scaled), scaled)
	}
	return scaled
}

// InvertAffineAsinh computes the inverse of the forward normalization: sinh(x) * scale + loc.
func InvertAffineAsinh(x, loc, scale *Node, useAsinh bool) *Node {
	if useAsinh {
		x = Sinh(x)
	}
	return Add(Mul(x, scale), loc)
}

// ClampNonFinite maps +/-Inf (from a sinh overflow) to the nearest finite value along the last axis.
// A position whose entire last axis is non-finite becomes 0. NaN is left untouched.
func ClampNonFinite(vals *Node) *Node {
	g := vals.Graph()
	dtype := vals.DType()
	posInf := BroadcastLike(Scalar(g, dtype, math.Inf(1)), vals)
	negInf := BroadcastLike(Scalar(g, dtype, math.Inf(-1)), vals)

	finiteMask := IsFinite(vals)
	maxFinite := BroadcastLike(ReduceAndKeep(Where(finiteMask, vals, negInf), ReduceMax, -1), vals)
	minFinite := BroadcastLike(ReduceAndKeep(Where(finiteMask, vals, posInf), ReduceMin, -1), vals)

	isPosInf := Equal(vals, posInf)
	isNegInf := Equal(vals, negInf)

	clamped := Where(isPosInf, maxFinite, Where(isNegInf, minFinite, vals))
	isStillInf := LogicalOr(Equal(clamped, posInf), Equal(clamped, negInf))
	return Where(isStillInf, ZerosLike(clamped), clamped)
}

// DenormalizeQuantiles inverts the affine+asinh normalization and clamps infinities.
func DenormalizeQuantiles(preds, loc, scale *Node, useAsinh bool) *Node {
	return ClampNonFinite(InvertAffineAsinh(preds, loc, scale, useAsinh))
}

// welfordCombine merges two Welford states (weight, mean, m2) element-wise; (0, 0, 0) is the identity.
func welfordCombine(wA, meanA, m2A, wB, meanB, m2B *Node) (w, mean, m2 *Node) {
	g := wA.Graph()
	w = Add(wA, wB)
	wSafe := MaxScalar(w, float64Tiny)
	delta := Sub(meanB, meanA)
	mean = Add(meanA, Div(Mul(delta, wB), wSafe))
	m2 = Add(Add(m2A, m2B), Div(Mul(Mul(Square(delta), wA), wB), wSafe))

	isZero := Equal(w, ScalarZero(g, w.DType()))
	mean = Where(isZero, ZerosLike(mean), mean)
	m2 = Where(isZero, ZerosLike(m2), m2)
	return w, mean, m2
}

// cumulativeWelford performs an inclusive prefix scan of Welford states along axis 1 (Hillis-Steele doubling).
func cumulativeWelford(w, mean, m2 *Node) (*Node, *Node, *Node) {
	g := w.Graph()
	n := w.Shape().Dimensions[1]
	zero := ScalarZero(g, w.DType())
	for d := 1; d < n; d *= 2 {
		padAxes := []PadAxis{{}, {Start: d}}
		wL := Pad(SliceAxis(w, 1, AxisRange(0, n-d)), zero, padAxes...)
		meanL := Pad(SliceAxis(mean, 1, AxisRange(0, n-d)), zero, padAxes...)
		m2L := Pad(SliceAxis(m2, 1, AxisRange(0, n-d)), zero, padAxes...)
		w, mean, m2 = welfordCombine(wL, meanL, m2L, w, mean, m2)
	}
	return w, mean, m2
}

// AdvancingBlockFrames computes per-patch (loc, scale) of a decoded block in the advancing refine frame,
// returning (locs, scales) each of shape [R, H] in float64, matching timestral.scaler.advancing_block_frames.
//
//   - context: [R, T] block context, T a multiple of patchSize.
//   - observationMask: [R, T] bool.
//   - sinhMedian: [R, H, patchSize] float64 — the block's median in the last context patch's frame
//     with asinh undone: (median - loc0) / scale0.
func AdvancingBlockFrames(context, observationMask, sinhMedian *Node, minimumScale float64) (locs, scales *Node) {
	g := context.Graph()
	r := sinhMedian.Shape().Dimensions[0]
	nPred := sinhMedian.Shape().Dimensions[1]
	ps := sinhMedian.Shape().Dimensions[2]
	ctxPatches := context.Shape().Dimensions[1] / ps

	sinhMedian64 := ConvertDType(sinhMedian, dtypes.Float64)
	blank := Zeros(g, shapes.Make(dtypes.Float64, r, nPred, ps))

	ctxValues := ConvertDType(Reshape(context, r, ctxPatches, ps), dtypes.Float64)
	ctxWeights := ConvertDType(Reshape(observationMask, r, ctxPatches, ps), dtypes.Float64)

	values := Concatenate([]*Node{ctxValues, blank}, 1)
	weights := Concatenate([]*Node{ctxWeights, blank}, 1)

	cnt := ReduceSum(weights, -1)
	mean := Div(ReduceSum(Mul(values, weights), -1), MaxScalar(cnt, 1.0))
	m2 := ReduceSum(Mul(weights, Square(Sub(values, InsertAxes(mean, -1)))), -1)

	isZero := Equal(cnt, ScalarZero(g, dtypes.Float64))
	mean = Where(isZero, ZerosLike(mean), mean)
	m2 = Where(isZero, ZerosLike(m2), m2)

	cntAll, meanAll, m2All := cumulativeWelford(cnt, mean, m2)
	// Slice index ctxPatches - 1 (which equals totalPatches - nPred - 1) and squeeze to [R].
	lastCtxIdx := ctxPatches - 1
	cnt = Reshape(SliceAxis(cntAll, 1, AxisRange(lastCtxIdx, lastCtxIdx+1)), r)
	mean = Reshape(SliceAxis(meanAll, 1, AxisRange(lastCtxIdx, lastCtxIdx+1)), r)
	m2 = Reshape(SliceAxis(m2All, 1, AxisRange(lastCtxIdx, lastCtxIdx+1)), r)

	offset := mean
	clip := math.Sqrt(math.MaxFloat32) / 1e3
	clipMin := Scalar(g, dtypes.Float64, -clip)
	clipMax := Scalar(g, dtypes.Float64, clip)
	psFull := BroadcastToDims(Scalar(g, dtypes.Float64, float64(ps)), r)

	locSlices := make([]*Node, nPred)
	scaleSlices := make([]*Node, nPred)
	offsetCol := InsertAxes(offset, 1) // [R, 1]

	for h := range nPred {
		variance := MaxScalar(Div(m2, MaxScalar(AddScalar(cnt, -1.0), 1.0)), 0.0)
		scale := MaxScalar(Sqrt(variance), minimumScale)
		locSlices[h] = InsertAxes(mean, 1)   // [R, 1]
		scaleSlices[h] = InsertAxes(scale, 1) // [R, 1]

		sinhMedH := Reshape(SliceAxis(sinhMedian64, 1, AxisRange(h, h+1)), r, ps)
		meanCol := InsertAxes(mean, 1)
		scaleCol := InsertAxes(scale, 1)

		rel := Sub(Add(Mul(sinhMedH, scaleCol), meanCol), offsetCol)
		folded := Add(Clamp(clipMin, rel, clipMax), offsetCol)
		bMean := ReduceMean(folded, -1)
		bM2 := ReduceSum(Square(Sub(folded, InsertAxes(bMean, 1))), -1)
		cnt, mean, m2 = welfordCombine(cnt, mean, m2, psFull, bMean, bM2)
	}

	return Concatenate(locSlices, 1), Concatenate(scaleSlices, 1)
}

// InstanceNormStats computes non-causal per-row mean/std over observed positions for known-future covariates.
// Rows whose observed values are all 0 or 1 (indicator covariates) keep loc=0, scale=1.
// Returns (loc, scale, isBinary), each shaped [R, 1].
func InstanceNormStats(values, observationMask *Node, minimumScale float64) (loc, scale, isBinary *Node) {
	g := values.Graph()
	origDType := values.DType()

	x := ConvertDType(values, dtypes.Float32)
	maskF := ConvertDType(observationMask, dtypes.Float32)

	n := MaxScalar(ReduceAndKeep(maskF, ReduceSum, -1), 1.0)
	loc = Div(ReduceAndKeep(Mul(x, maskF), ReduceSum, -1), n)
	variance := Div(ReduceAndKeep(Square(Mul(Sub(x, loc), maskF)), ReduceSum, -1), n)
	scale = MaxScalar(Sqrt(variance), minimumScale)

	zero := ScalarZero(g, values.DType())
	one := ScalarOne(g, values.DType())
	isZeroOrOne := LogicalOr(LogicalOr(Equal(values, zero), Equal(values, one)), LogicalNot(observationMask))
	allZeroOne := ReduceAndKeep(isZeroOrOne, ReduceLogicalAnd, -1)
	anyObs := ReduceAndKeep(observationMask, ReduceLogicalOr, -1)
	isBinary = LogicalAnd(allZeroOne, anyObs)

	loc = Where(isBinary, ZerosLike(loc), loc)
	scale = Where(isBinary, OnesLike(scale), scale)
	return ConvertDType(loc, origDType), ConvertDType(scale, origDType), isBinary
}
