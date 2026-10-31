// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package timestral

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
)

// TwoStreamMask holds a boolean time attention mask over the [Enc ; Dec] sequence
// together with the split point EncLen (number of encoder patches).
type TwoStreamMask struct {
	Mask   *Node // [R, 1, S, S] bool, true = attend
	EncLen int
}

// ComputeEncKeyValidity computes the [R, N] boolean encoder key validity mask.
// A patch is a valid key if it is observed, or if attendMissing is true and the row's
// series has started at or before that patch. Leading all-unobserved patches are left-padding
// and are never valid keys.
func ComputeEncKeyValidity(obsPerPatch *Node, attendMissing bool) *Node {
	if !attendMissing {
		return obsPerPatch
	}
	g := obsPerPatch.Graph()
	obsInt := ConvertDType(obsPerPatch, dtypes.Int32)
	started := GreaterThan(CumMax(obsInt, -1), ScalarZero(g, dtypes.Int32))
	return LogicalOr(obsPerPatch, started)
}

// eyeBool returns a 2D [n, n] boolean identity matrix with true on the main diagonal.
func eyeBool(g *Graph, n int) *Node {
	rowIdx := Iota(g, shapes.Make(dtypes.Int32, n, n), 0)
	colIdx := Iota(g, shapes.Make(dtypes.Int32, n, n), 1)
	return Equal(rowIdx, colIdx)
}

// EncCausalMask builds the [R, L, L] causal encoder mask over valid keys, with the diagonal always open.
func EncCausalMask(keyValid *Node) *Node {
	g := keyValid.Graph()
	rows := keyValid.Shape().Dimensions[0]
	ll := keyValid.Shape().Dimensions[1]

	causal := BroadcastToDims(InsertAxes(LowerTriangular(g, ll), 0), rows, ll, ll)
	eye := BroadcastToDims(InsertAxes(eyeBool(g, ll), 0), rows, ll, ll)
	validCols := BroadcastToDims(InsertAxes(keyValid, 1), rows, ll, ll)

	return LogicalOr(LogicalAnd(causal, validCols), eye)
}

// BuildForecastTimeMask builds the [R, 1, L+h, L+h] boolean time mask for one block without covariates.
// The encoder is causal; every decoder token reads the whole valid context and its own causal decoder chain;
// the context never reads the horizon.
func BuildForecastTimeMask(contextLen, horizon int, keyValid *Node) *Node {
	g := keyValid.Graph()
	rows := keyValid.Shape().Dimensions[0]

	enc := EncCausalMask(keyValid)
	encDec := BroadcastToDims(Const(g, false), rows, contextLen, horizon)
	decEnc := BroadcastToDims(InsertAxes(keyValid, 1), rows, horizon, contextLen)
	decDec := BroadcastToDims(InsertAxes(LowerTriangular(g, horizon), 0), rows, horizon, horizon)

	top := Concatenate([]*Node{enc, encDec}, 2)
	bottom := Concatenate([]*Node{decEnc, decDec}, 2)
	return InsertAxes(Concatenate([]*Node{top, bottom}, 1), 1)
}

// BuildForecastPositions builds [R, contextLen+horizon] integer time positions 0..L+h-1,
// anchored at each row's first observed patch.
func BuildForecastPositions(contextLen, horizon int, firstObs *Node) *Node {
	g := firstObs.Graph()
	rows := firstObs.Shape().Dimensions[0]
	total := contextLen + horizon

	p := BroadcastToDims(Iota(g, shapes.Make(dtypes.Int32, 1, total), 1), rows, total)
	offset := BroadcastToDims(InsertAxes(firstObs, 1), rows, total)
	return Sub(p, offset)
}

// ComputeBlockLayout computes per decoder position the anchor of its block and whether it is active (p >= s).
func ComputeBlockLayout(minContext *Node, nPatches, blockSize int) (anchor, active *Node) {
	g := minContext.Graph()
	rows := minContext.Shape().Dimensions[0]

	p := BroadcastToDims(Iota(g, shapes.Make(dtypes.Int32, 1, nPatches), 1), rows, nPatches)
	s := BroadcastToDims(InsertAxes(minContext, 1), rows, nPatches)
	active = GreaterOrEqual(p, s)
	offset := MaxScalar(Sub(p, s), 0)
	bs := Scalar(g, dtypes.Int32, int32(blockSize))
	anchor = Add(s, Mul(bs, Div(offset, bs)))
	anchor = Where(active, anchor, ZerosLike(anchor))
	return anchor, active
}

// BuildTwoStreamTimeMask builds the [R, 1, 2N, 2N] time mask over a full [Enc ; Dec] window (the covariate path).
func BuildTwoStreamTimeMask(minContext *Node, nPatches, blockSize int, keyValid, bidirectionalRows *Node) *Node {
	anchor, active := ComputeBlockLayout(minContext, nPatches, blockSize)
	g := keyValid.Graph()
	rows := active.Shape().Dimensions[0]
	n := nPatches

	causal := BroadcastToDims(InsertAxes(LowerTriangular(g, n), 0), rows, n, n)
	keyCol := BroadcastToDims(InsertAxes(keyValid, 1), rows, n, n)
	j := BroadcastToDims(Iota(g, shapes.Make(dtypes.Int32, 1, 1, n), 2), rows, n, n)
	eye := BroadcastToDims(InsertAxes(eyeBool(g, n), 0), rows, n, n)

	encEnc := EncCausalMask(keyValid)
	if bidirectionalRows != nil {
		encBi := LogicalOr(keyCol, eye)
		bi3D := BroadcastToDims(Reshape(bidirectionalRows, rows, 1, 1), rows, n, n)
		encEnc = Where(bi3D, encBi, encEnc)
	}
	encDec := BroadcastToDims(Const(g, false), rows, n, n)

	activeRow := BroadcastToDims(InsertAxes(active, 2), rows, n, n)
	activeCol := BroadcastToDims(InsertAxes(active, 1), rows, n, n)
	anchorRow := BroadcastToDims(InsertAxes(anchor, 2), rows, n, n)
	anchorCol := BroadcastToDims(InsertAxes(anchor, 1), rows, n, n)

	decEnc := LogicalAnd(LogicalAnd(activeRow, LessThan(j, anchorRow)), keyCol)
	sameBlock := Equal(anchorRow, anchorCol)
	decDec := LogicalOr(LogicalAnd(LogicalAnd(LogicalAnd(sameBlock, causal), activeRow), activeCol), eye)

	top := Concatenate([]*Node{encEnc, encDec}, 2)
	bottom := Concatenate([]*Node{decEnc, decDec}, 2)
	return InsertAxes(Concatenate([]*Node{top, bottom}, 1), 1)
}

// VariateFirewall returns the [R, R] forbidden query-to-key row pairs, or nil if neither
// targetRows nor bidirectionalRows is provided.
//   - A covariate row may not read a target row (covariates are exogenous).
//   - A known-future (bidirectional) row may not read a causal row.
func VariateFirewall(r int, g *Graph, targetRows, bidirectionalRows *Node) *Node {
	if targetRows == nil && bidirectionalRows == nil {
		return nil
	}
	forbid := BroadcastToDims(Const(g, false), r, r)
	if targetRows != nil {
		notTargetRow := BroadcastToDims(InsertAxes(LogicalNot(targetRows), 1), r, r)
		targetCol := BroadcastToDims(InsertAxes(targetRows, 0), r, r)
		forbid = LogicalOr(forbid, LogicalAnd(notTargetRow, targetCol))
	}
	if bidirectionalRows != nil {
		biRow := BroadcastToDims(InsertAxes(bidirectionalRows, 1), r, r)
		notBiCol := BroadcastToDims(InsertAxes(LogicalNot(bidirectionalRows), 0), r, r)
		forbid = LogicalOr(forbid, LogicalAnd(biRow, notBiCol))
	}
	return forbid
}

// BuildTwoStreamVariateMask builds the [N_enc + nDec, 1, R, R] boolean variate mask, or nil when R == 1.
func BuildTwoStreamVariateMask(groupIDs, encKeyValid *Node, nDec int, targetRows, bidirectionalRows *Node) *Node {
	r := encKeyValid.Shape().Dimensions[0]
	if r == 1 {
		return nil
	}
	g := encKeyValid.Graph()
	rowEq := Equal(
		BroadcastToDims(InsertAxes(groupIDs, 1), r, r),
		BroadcastToDims(InsertAxes(groupIDs, 0), r, r),
	)
	decValid := BroadcastToDims(Const(g, true), r, nDec)
	slotValid := Concatenate([]*Node{encKeyValid, decValid}, 1)
	s := slotValid.Shape().Dimensions[1]
	keyValid := TransposeAllAxes(slotValid, 1, 0) // [S, R]

	rowEq4D := BroadcastToDims(Reshape(rowEq, 1, 1, r, r), s, 1, r, r)
	keyValid4D := BroadcastToDims(Reshape(keyValid, s, 1, 1, r), s, 1, r, r)
	mask := LogicalAnd(rowEq4D, keyValid4D)

	forbid := VariateFirewall(r, g, targetRows, bidirectionalRows)
	if forbid != nil {
		forbid4D := BroadcastToDims(Reshape(forbid, 1, 1, r, r), s, 1, r, r)
		mask = LogicalAnd(mask, LogicalNot(forbid4D))
	}

	eye4D := BroadcastToDims(Reshape(eyeBool(g, r), 1, 1, r, r), s, 1, r, r)
	return LogicalOr(mask, eye4D)
}

// BuildTwoStreamPositions builds [R, 2*nPatches] integer positions where encoder and decoder
// tokens of the same patch share a time position.
func BuildTwoStreamPositions(nPatches int, firstObs *Node) *Node {
	g := firstObs.Graph()
	rows := firstObs.Shape().Dimensions[0]
	p := BroadcastToDims(Iota(g, shapes.Make(dtypes.Int32, 1, nPatches), 1), rows, nPatches)
	row := Sub(p, BroadcastToDims(InsertAxes(firstObs, 1), rows, nPatches))
	return Concatenate([]*Node{row, row}, -1)
}

// RowFirstObservedPatch returns a [R] int32 tensor containing, for each row, the index of the
// first patch observed by any row belonging to the same series group (or 0 if none).
// This corresponds to group_first_observed_patch(obs_per_patch, inverse, num_groups)[inverse] in Python.
func RowFirstObservedPatch(obsPerPatch, groupIDs *Node) *Node {
	r := obsPerPatch.Shape().Dimensions[0]
	n := obsPerPatch.Shape().Dimensions[1]

	var groupObsPerRow *Node
	if r == 1 {
		groupObsPerRow = obsPerPatch
	} else {
		sameGroup := Equal(
			BroadcastToDims(InsertAxes(groupIDs, 1), r, r),
			BroadcastToDims(InsertAxes(groupIDs, 0), r, r),
		)
		sameGroup3D := BroadcastToDims(InsertAxes(sameGroup, 2), r, r, n)
		obs3D := BroadcastToDims(InsertAxes(obsPerPatch, 0), r, r, n)
		groupObsPerRow = ReduceLogicalOr(LogicalAnd(sameGroup3D, obs3D), 1)
	}

	obsInt := ConvertDType(groupObsPerRow, dtypes.Int32)
	started := CumMax(obsInt, -1) // 0 before first observed patch, 1 from first observed patch onwards
	leadingUnobserved := ReduceSum(Sub(OnesLike(started), started), -1)
	anyObs := ReduceLogicalOr(groupObsPerRow, -1)
	return Where(anyObs, leadingUnobserved, ZerosLike(leadingUnobserved))
}
