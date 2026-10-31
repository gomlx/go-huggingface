// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package timestral

import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	"github.com/gomlx/gomlx/support/exceptions"
)

// RMSNorm applies Root Mean Square Layer Normalization along the last axis,
// matching timestral.layers.RMSNorm.
// Uses scope "rms_norm" and variable "scale".
func RMSNorm(scope *model.Scope, x *Node, eps float64) *Node {
	scope = scope.In("rms_norm")
	g := x.Graph()
	origDType := x.DType()
	dim := x.Shape().Dimensions[x.Rank()-1]

	xf := x
	if origDType != dtypes.Float32 {
		xf = ConvertDType(x, dtypes.Float32)
	}
	ms := ReduceAndKeep(Square(xf), ReduceMean, -1)
	if eps != 0 {
		ms = AddScalar(ms, eps)
	}
	out := Mul(xf, Rsqrt(ms))
	if origDType != dtypes.Float32 {
		out = ConvertDType(out, origDType)
	}

	scaleVar := scope.WithInitializer(initializer.One).VariableWithShape("scale", shapes.Make(origDType, dim))
	scale := ExpandLeftToRank(scaleVar.NodeValue(g), out.Rank())
	return Mul(out, scale)
}

// ResidualBlock computes output_layer(silu(hidden_layer(x))) + residual_layer(x),
// used by PatchRepresentation and QuantileHead.
func ResidualBlock(scope *model.Scope, x *Node, hiddenDims, outputDims int, useBias bool) *Node {
	hidden := layers.Dense(scope.In("hidden_layer"), x, useBias, hiddenDims)
	act := activation.Swish(hidden)
	out := layers.Dense(scope.In("output_layer"), act, useBias, outputDims)
	res := layers.Dense(scope.In("residual_layer"), x, useBias, outputDims)
	return Add(out, res)
}

// FusedSwiGLU computes the SwiGLU MLP with one fused input projection w12: w3(gate * silu(up)).
func FusedSwiGLU(scope *model.Scope, x *Node, hiddenDim int, useBias bool) *Node {
	embedDim := x.Shape().Dimensions[x.Rank()-1]
	w12 := layers.Dense(scope.In("w12"), x, useBias, 2*hiddenDim)
	gate := SliceAxis(w12, -1, AxisRange(0, hiddenDim))
	up := SliceAxis(w12, -1, AxisRange(hiddenDim, 2*hiddenDim))
	return layers.Dense(scope.In("w3"), Mul(gate, activation.Swish(up)), useBias, embedDim)
}

// SinCosRoPE computes the rotary position embedding (cos, sin) tables of shape [R, N, headDim]
// for integer positionIDs [R, N], rotating only the first rotaryDim dimensions of each head.
func SinCosRoPE(x, positionIDs *Node, headDim int, base float64, rotaryDim int) (cos, sin *Node) {
	g := x.Graph()
	idx := MulScalar(Iota(g, shapes.Make(dtypes.Float32, rotaryDim/2), 0), 2.0)
	activeInvFreq := Reciprocal(Pow(Scalar(g, dtypes.Float32, float32(base)), DivScalar(idx, float64(rotaryDim))))

	invFreq := activeInvFreq
	if rotaryDim < headDim {
		invFreq = Pad(activeInvFreq, ScalarZero(g, dtypes.Float32), PadAxis{End: (headDim - rotaryDim) / 2})
	}

	// positionIDs: [R, N] -> [R, N, 1]; invFreq: [1, 1, headDim/2] -> freqs: [R, N, headDim/2]
	posF32 := InsertAxes(ConvertDType(positionIDs, dtypes.Float32), -1)
	invFreq3D := Reshape(invFreq, 1, 1, headDim/2)
	freqs := Mul(posF32, invFreq3D)
	emb := Concatenate([]*Node{freqs, freqs}, -1)

	cos = ConvertDType(Cos(emb), x.DType())
	sin = ConvertDType(Sin(emb), x.DType())
	return cos, sin
}

// rotateHalf splits the last axis in half [x1, x2] and returns [-x2, x1].
func rotateHalf(x *Node) *Node {
	half := x.Shape().Dimensions[x.Rank()-1] / 2
	x1 := SliceAxis(x, -1, AxisRange(0, half))
	x2 := SliceAxis(x, -1, AxisRange(half, 2*half))
	return Concatenate([]*Node{Neg(x2), x1}, -1)
}

// applyRotary applies rotary embeddings (cos, sin) of shape [batch, seq, headDim]
// to q and k of shape [batch, seq, heads, headDim].
func applyRotary(q, k, cos, sin *Node) (*Node, *Node) {
	cos4D := InsertAxes(cos, 2)
	sin4D := InsertAxes(sin, 2)
	qRot := Add(Mul(q, cos4D), Mul(rotateHalf(q), sin4D))
	kRot := Add(Mul(k, cos4D), Mul(rotateHalf(k), sin4D))
	return qRot, kRot
}

// sdpaBSHD executes scaled dot-product attention on [batch, seq, heads, headDim] tensors
// and returns [batch, qSeq, heads * headDim].
func sdpaBSHD(query, key, value, mask *Node, isCausal bool) *Node {
	batch := query.Shape().Dimensions[0]
	qSeq := query.Shape().Dimensions[1]
	numHeads := query.Shape().Dimensions[2]
	headDim := query.Shape().Dimensions[3]

	var mask3D *Node
	if mask != nil {
		if mask.Rank() == 4 && mask.Shape().Dimensions[1] == 1 {
			kvSeq := mask.Shape().Dimensions[3]
			mask3D = Reshape(mask, mask.Shape().Dimensions[0], qSeq, kvSeq)
		} else {
			mask3D = mask
		}
	}

	out, _ := attention.Core(query, key, value, attention.LayoutBSHD, attention.CoreOptions{
		AttentionMask: mask3D,
		UseCausalMask: isCausal && mask3D == nil,
	})
	return Reshape(out, batch, qSeq, numHeads*headDim)
}

// Attention computes multi-head attention over the time axis (isTime=true) or the variate axis (isTime=false).
//
// Note: wQKV weights are permuted at LoadStore time from [3, headDim, numHeads] to [3, numHeads, headDim]
// so that wQKV(inputs) directly produces [batch, seq, 3, numHeads, headDim].
func Attention(
	scope *model.Scope,
	x *Node,
	embedDim, numHeads int,
	isTime, useBias bool,
	gate string,
	mask *Node,
	twoStream *TwoStreamMask,
	cos, sin *Node,
) *Node {
	rows := x.Shape().Dimensions[0]
	numPatches := x.Shape().Dimensions[1]
	headDim := embedDim / numHeads

	inputs := x
	if !isTime {
		inputs = TransposeAllAxes(x, 1, 0, 2) // [N, R, D]
	}
	batch := inputs.Shape().Dimensions[0]
	seq := inputs.Shape().Dimensions[1]

	qkv := layers.Dense(scope.In("wQKV"), inputs, useBias, 3, numHeads, headDim)
	q := Reshape(Slice(qkv, AxisRange(), AxisRange(), AxisElem(0)), batch, seq, numHeads, headDim)
	k := Reshape(Slice(qkv, AxisRange(), AxisRange(), AxisElem(1)), batch, seq, numHeads, headDim)
	v := Reshape(Slice(qkv, AxisRange(), AxisRange(), AxisElem(2)), batch, seq, numHeads, headDim)

	if isTime {
		q, k = applyRotary(q, k, cos, sin)
	}

	var out *Node
	if twoStream != nil {
		if !isTime {
			exceptions.Panicf("TwoStreamMask context/horizon split applies to time attention only")
		}
		n := twoStream.EncLen
		qEnc := SliceAxis(q, 1, AxisRange(0, n))
		kEnc := SliceAxis(k, 1, AxisRange(0, n))
		vEnc := SliceAxis(v, 1, AxisRange(0, n))
		maskEnc := Slice(twoStream.Mask, AxisRange(), AxisRange(), AxisRange(0, n), AxisRange(0, n))
		outEnc := sdpaBSHD(qEnc, kEnc, vEnc, maskEnc, false)

		qDec := SliceAxis(q, 1, AxisRange(n, seq))
		maskDec := Slice(twoStream.Mask, AxisRange(), AxisRange(), AxisRange(n, seq), AxisRange())
		outDec := sdpaBSHD(qDec, k, v, maskDec, false)

		out = Concatenate([]*Node{outEnc, outDec}, 1)
	} else {
		out = sdpaBSHD(q, k, v, mask, isTime && mask == nil)
	}

	if gate == GateSDPASigmoid {
		gVal := Logistic(layers.Dense(scope.In("_gate"), inputs, useBias, embedDim))
		out = Mul(out, gVal)
	}

	if isTime {
		out = Reshape(out, rows, numPatches, embedDim)
	} else {
		out = TransposeAllAxes(Reshape(out, numPatches, rows, embedDim), 1, 0, 2)
	}
	return layers.Dense(scope.In("wO"), out, useBias, embedDim)
}

// TransformerBlock runs the attention sublayer(s) of one LayerKind in order (each with pre-RMSNorm residual),
// followed by the shared pre-RMSNorm FusedSwiGLU MLP residual.
func TransformerBlock(
	scope *model.Scope,
	x *Node,
	kind LayerKind,
	cfg *Config,
	timeMask *Node,
	twoStream *TwoStreamMask,
	variateMask *Node,
	cos, sin *Node,
) *Node {
	axes, err := kind.Axes()
	if err != nil {
		panic(err)
	}
	attnNormsScope := scope.In("attn_norms")
	attnsScope := scope.In("attns")
	for i, isTime := range axes {
		normX := RMSNorm(attnNormsScope.In("%d", i), x, cfg.NormEps)
		gate := cfg.VariateAttnGate
		var attnMask *Node
		var tsMask *TwoStreamMask
		if isTime {
			gate = cfg.TimeAttnGate
			attnMask = timeMask
			tsMask = twoStream
		} else {
			attnMask = variateMask
		}
		attnOut := Attention(
			attnsScope.In("%d", i),
			normX,
			cfg.DModel,
			cfg.NumHeads,
			isTime,
			cfg.AttnBias,
			gate,
			attnMask,
			tsMask,
			cos,
			sin,
		)
		x = Add(x, attnOut)
	}

	mlpNormX := RMSNorm(scope.In("mlp_norm"), x, cfg.NormEps)
	mlpOut := FusedSwiGLU(scope.In("mlp"), mlpNormX, cfg.MLPHiddenDim, cfg.MLPBias)
	return Add(x, mlpOut)
}

// Transformer runs the full stack of TransformerBlocks on x [R, N, D] with integer positions [R, N].
func Transformer(
	scope *model.Scope,
	x *Node,
	cfg *Config,
	timeMask *Node,
	twoStream *TwoStreamMask,
	variateMask *Node,
	positions *Node,
) *Node {
	cos, sin := SinCosRoPE(x, positions, cfg.HeadDim(), cfg.RoPETheta, cfg.RoPEDim())
	layersScope := scope.In("layers")
	for i, kind := range cfg.BlockSchedule {
		layerScope := layersScope.In("%d", i)
		x = TransformerBlock(layerScope, x, kind, cfg, timeMask, twoStream, variateMask, cos, sin)
	}
	return RMSNorm(scope.In("out_norm"), x, cfg.NormEps)
}

// PatchRepresentation embeds normalized context values [R, T] and observation mask [R, T]
// into [R, T / patchSize, dModel].
func PatchRepresentation(scope *model.Scope, scaled, observed *Node, patchSize, dModel int) *Node {
	r := scaled.Shape().Dimensions[0]
	t := scaled.Shape().Dimensions[1]
	n := t / patchSize

	values := Reshape(scaled, r, n, patchSize)
	missing := ConvertDType(Reshape(LogicalNot(observed), r, n, patchSize), scaled.DType())
	x := Concatenate([]*Node{values, missing}, -1)
	return ResidualBlock(scope.In("patch_proj"), x, 4*dModel, dModel, true)
}

// ForecastTokens broadcasts the learned forecast_query parameter [dModel] to [rows, nPatches, dModel].
func ForecastTokens(scope *model.Scope, g *Graph, dtype dtypes.DType, rows, nPatches, dModel int) *Node {
	qVar := scope.WithInitializer(initializer.Zero).VariableWithShape("forecast_query", shapes.Make(dtype, dModel))
	q := qVar.NodeValue(g)
	return BroadcastToDims(Reshape(q, 1, 1, dModel), rows, nPatches, dModel)
}

// MonotoneQuantiles maps raw head outputs [..., Q] to non-crossing monotone quantiles along the last axis.
func MonotoneQuantiles(raw *Node, levels []float64, widthFactor float64) *Node {
	g := raw.Graph()
	dtype := raw.DType()
	numQ := len(levels)

	m := -1
	for i, q := range levels {
		if q == 0.5 {
			m = i
			break
		}
	}
	if m < 0 {
		exceptions.Panicf("MonotoneQuantiles: levels must contain 0.5, got %v", levels)
	}

	widths := make([]float32, numQ)
	for i := m + 1; i < numQ; i++ {
		widths[i] = float32(levels[i] - levels[i-1])
	}
	for i := range m {
		widths[i] = float32(levels[i+1] - levels[i])
	}

	widthsShape := make([]int, raw.Rank())
	for i := range raw.Rank() - 1 {
		widthsShape[i] = 1
	}
	widthsShape[raw.Rank()-1] = numQ
	widthsNode := ConvertDType(Reshape(Const(g, widths), widthsShape...), dtype)

	inc := Mul(MulScalar(widthsNode, widthFactor), Softplus(raw))
	median := SliceAxis(raw, -1, AxisRange(m, m+1))

	var parts []*Node
	if m > 0 {
		incDown := SliceAxis(inc, -1, AxisRange(0, m))
		down := CumSum(incDown, -1, &CumSumOptions{Reverse: true})
		parts = append(parts, Sub(BroadcastLike(median, down), down))
	}
	parts = append(parts, median)
	if m+1 < numQ {
		incUp := SliceAxis(inc, -1, AxisRange(m+1, numQ))
		up := CumSum(incUp, -1)
		parts = append(parts, Add(BroadcastLike(median, up), up))
	}

	res := Concatenate(parts, -1)
	big := highestForDTypeScalar(g, dtype)
	negBig := Neg(big)
	return Clamp(negBig, res, big)
}

func highestForDTypeScalar(g *Graph, dtype dtypes.DType) *Node {
	switch dtype {
	case dtypes.Float32:
		return Scalar(g, dtype, float32(3.4028234663852886e+38))
	case dtypes.Float64:
		return Scalar(g, dtype, float64(1.7976931348623157e+308))
	default:
		return Scalar(g, dtype, float32(65504.0))
	}
}

// QuantileHead maps decoder tokens [R, N, D] to normalized quantiles [R, N * patchSize, Q].
func QuantileHead(scope *model.Scope, x *Node, cfg *Config) *Node {
	ps := cfg.PatchSize
	numQ := cfg.NumQuantiles()
	raw := ResidualBlock(scope.In("projection"), x, 4*cfg.DModel, ps*numQ, false)

	r := raw.Shape().Dimensions[0]
	nPatches := raw.Shape().Dimensions[1]
	preds := Reshape(raw, r, nPatches, ps, numQ)
	if preds.DType() != dtypes.Float32 && preds.DType() != dtypes.Float64 {
		preds = ConvertDType(preds, dtypes.Float32)
	}
	preds = MonotoneQuantiles(preds, cfg.QuantileKnots, cfg.QuantileWidthFactor)
	return Reshape(preds, r, nPatches*ps, numQ)
}
