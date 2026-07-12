// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sam2

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors/images"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/model"
)

// twoWayAttention implements the two-way multi-head attention block used in the decoder.
func twoWayAttention(scope *model.Scope, query, key, value, attentionMask *Node, downsampleRate int, numHeads int) *Node {
	hiddenSize := query.Shape().Dimensions[query.Rank()-1]
	internalDim := hiddenSize / downsampleRate
	headDim := internalDim / numHeads
	scaling := 1.0 / math.Sqrt(float64(headDim))

	// Projects: [batchSize, num_tokens, internalDim]
	q := layers.Dense(scope.In("q_proj"), query, true, internalDim)
	k := layers.Dense(scope.In("k_proj"), key, true, internalDim)
	v := layers.Dense(scope.In("v_proj"), value, true, internalDim)

	batchSize := query.Shape().Dimensions[0]

	// Reshape to [batchSize, num_tokens, numHeads, headDim]
	q = Reshape(q, batchSize, -1, numHeads, headDim)
	k = Reshape(k, batchSize, -1, numHeads, headDim)
	v = Reshape(v, batchSize, -1, numHeads, headDim)

	// Transpose to [batchSize, numHeads, num_tokens, headDim]
	q = TransposeAllAxes(q, 0, 2, 1, 3)
	k = TransposeAllAxes(k, 0, 2, 1, 3)
	v = TransposeAllAxes(v, 0, 2, 1, 3)

	// Use fused attention via attention.Core
	attnOutput, _ := attention.Core(q, k, v, attention.LayoutBHSD, attention.CoreOptions{
		Scale:         scaling,
		AttentionMask: attentionMask,
	})

	// Transpose to [batchSize, num_queries, numHeads, headDim]
	attnOutput = TransposeAllAxes(attnOutput, 0, 2, 1, 3)

	// Reshape to [batchSize, num_queries, internalDim]
	attnOutput = Reshape(attnOutput, batchSize, -1, internalDim)

	// o_proj
	return layers.Dense(scope.In("o_proj"), attnOutput, true, hiddenSize)
}

// Sam2MultiScaleAttention builds the multi-head attention block used in the Hiera backbone.
func Sam2MultiScaleAttention(scope *model.Scope, hiddenStates *Node, dim, dimOut, numHeads int, queryStride []int) *Node {
	batchSize := hiddenStates.Shape().Dimensions[0]
	height := hiddenStates.Shape().Dimensions[1]
	width := hiddenStates.Shape().Dimensions[2]
	headDim := dimOut / numHeads
	scale := 1.0 / math.Sqrt(float64(headDim))

	// qkv linear: [batchSize, height * width, dimOut * 3]
	hiddenStates2D := Reshape(hiddenStates, batchSize, height*width, -1)
	qkv := layers.Dense(scope.In("qkv"), hiddenStates2D, true, dimOut*3)

	// Reshape to [batchSize, height * width, 3, numHeads, headDim]
	qkv = Reshape(qkv, batchSize, height*width, 3, numHeads, headDim)
	query := Squeeze(Slice(qkv, AxisRange(), AxisRange(), AxisElem(0), AxisRange(), AxisRange()), 2)
	key := Squeeze(Slice(qkv, AxisRange(), AxisRange(), AxisElem(1), AxisRange(), AxisRange()), 2)
	value := Squeeze(Slice(qkv, AxisRange(), AxisRange(), AxisElem(2), AxisRange(), AxisRange()), 2)

	// Q pooling (spatial downsampling)
	if len(queryStride) > 0 {
		queryReshaped := Reshape(query, batchSize, height, width, -1)
		queryPooled := MaxPool(queryReshaped).
			ChannelsAxis(images.ChannelsLast).
			WindowPerAxis(queryStride...).
			StridePerAxis(queryStride...).
			Done()
		newHeight := queryPooled.Shape().Dimensions[1]
		newWidth := queryPooled.Shape().Dimensions[2]
		query = Reshape(queryPooled, batchSize, newHeight*newWidth, numHeads, headDim)
		height = newHeight
		width = newWidth
	}

	// Transpose to [batchSize, numHeads, num_tokens, headDim]
	query = TransposeAllAxes(query, 0, 2, 1, 3)
	key = TransposeAllAxes(key, 0, 2, 1, 3)
	value = TransposeAllAxes(value, 0, 2, 1, 3)

	// Compute attention weights: [batchSize, numHeads, num_queries, num_keys]
	attnWeights := Einsum("bhqd,bhkd->bhqk", query, key)
	attnWeights = MulScalar(attnWeights, scale)
	attnWeights = Softmax(attnWeights, -1)

	// Compute attention output: [batchSize, numHeads, num_queries, headDim]
	attnOutput := Einsum("bhqk,bhkd->bhqd", attnWeights, value)

	// Transpose to [batchSize, num_queries, numHeads, headDim]
	attnOutput = TransposeAllAxes(attnOutput, 0, 2, 1, 3)

	// Reshape to [batchSize, height, width, dimOut]
	attnOutput = Reshape(attnOutput, batchSize, height, width, dimOut)

	// proj linear
	return layers.Dense(scope.In("proj"), attnOutput, true, dimOut)
}

// WindowPartition partitions input into non-overlapping windows.
func WindowPartition(x *Node, windowSize int) (windows *Node, paddedShape []int) {
	dims := x.Shape().Dimensions
	b := dims[0]
	h := dims[1]
	w := dims[2]
	c := dims[3]

	padH := (windowSize - h%windowSize) % windowSize
	padW := (windowSize - w%windowSize) % windowSize

	if padH > 0 || padW > 0 {
		x = Pad(x, ScalarZero(x.Graph(), x.DType()),
			compute.PadAxis{},
			compute.PadAxis{End: padH},
			compute.PadAxis{End: padW},
			compute.PadAxis{},
		)
	}

	hp := h + padH
	wp := w + padW

	// Reshape to [b, hp // windowSize, windowSize, wp // windowSize, windowSize, c]
	x = Reshape(x, b, hp/windowSize, windowSize, wp/windowSize, windowSize, c)
	// Permute to [b, hp // windowSize, wp // windowSize, windowSize, windowSize, c]
	x = TransposeAllAxes(x, 0, 1, 3, 2, 4, 5)
	// Reshape to [b * (hp // windowSize) * (wp // windowSize), windowSize, windowSize, c]
	windows = Reshape(x, -1, windowSize, windowSize, c)
	return windows, []int{hp, wp}
}

// WindowUnpartition merges non-overlapping windows back.
func WindowUnpartition(windows *Node, windowSize int, hp, wp, h, w int) *Node {
	dims := windows.Shape().Dimensions
	c := dims[3]

	numWindows := (hp / windowSize) * (wp / windowSize)
	b := dims[0] / numWindows

	// Reshape to [b, hp // windowSize, wp // windowSize, windowSize, windowSize, c]
	x := Reshape(windows, b, hp/windowSize, wp/windowSize, windowSize, windowSize, c)
	// Permute to [b, hp // windowSize, windowSize, wp // windowSize, windowSize, c]
	x = TransposeAllAxes(x, 0, 1, 3, 2, 4, 5)
	// Reshape to [b, hp, wp, c]
	x = Reshape(x, b, hp, wp, c)

	if hp > h || wp > w {
		x = Slice(x, AxisRange(), AxisRange(0, h), AxisRange(0, w), AxisRange())
	}
	return x
}

// doPool downsamples the input tensor along height and width using max pooling.
func doPool(x *Node, queryStride []int) *Node {
	if len(queryStride) == 0 {
		return x
	}
	return MaxPool(x).
		ChannelsAxis(images.ChannelsLast).
		WindowPerAxis(queryStride...).
		StridePerAxis(queryStride...).
		Done()
}

// Sam2MultiScaleBlock runs one hierarchical block (attention + FFN).
func Sam2MultiScaleBlock(scope *model.Scope, x *Node, dim, dimOut, numHeads, windowSize int, queryStride []int, useGlobalAttn bool) *Node {
	shortcut := x

	// x has shape [batch, H, W, dim]
	xNorm := norm.LayerNorm(scope.In("layer_norm1"), x, 3).Epsilon(1e-6).Done()

	var attnOut *Node
	if useGlobalAttn {
		attnOut = Sam2MultiScaleAttention(scope.In("attn"), xNorm, dim, dimOut, numHeads, queryStride)
	} else {
		h := xNorm.Shape().Dimensions[1]
		w := xNorm.Shape().Dimensions[2]
		windows, paddedShape := WindowPartition(xNorm, windowSize)
		windowsAttn := Sam2MultiScaleAttention(scope.In("attn"), windows, dim, dimOut, numHeads, queryStride)

		actualWindowSize := windowSize
		actualH := h
		actualW := w
		actualPaddedH := paddedShape[0]
		actualPaddedW := paddedShape[1]
		if len(queryStride) > 0 {
			actualWindowSize = windowSize / queryStride[0]
			actualH = h / queryStride[0]
			actualW = w / queryStride[1]
			actualPaddedH = paddedShape[0] / queryStride[0]
			actualPaddedW = paddedShape[1] / queryStride[1]
		}
		attnOut = WindowUnpartition(windowsAttn, actualWindowSize, actualPaddedH, actualPaddedW, actualH, actualW)
	}

	if dim != dimOut {
		shortcut = layers.Dense(scope.In("proj"), xNorm, true, dimOut)
	}
	if len(queryStride) > 0 {
		shortcut = doPool(shortcut, queryStride)
	}

	x = Add(shortcut, attnOut)

	shortcut = x
	xNorm2 := norm.LayerNorm(scope.In("layer_norm2"), x, 3).Epsilon(1e-6).Done()

	// FFN
	mlpScope := scope.In("mlp")
	mlpOut := layers.Dense(mlpScope.In("proj_in"), xNorm2, true, int(float64(dimOut)*4.0))
	mlpOut = activation.Gelu(mlpOut)
	mlpOut = layers.Dense(mlpScope.In("proj_out"), mlpOut, true, dimOut)

	x = Add(shortcut, mlpOut)
	return x
}

// Sam2PositionalEmbedding generates positional sine/cosine embeddings.
func Sam2PositionalEmbedding(scope *model.Scope, coords *Node, inputShape []int) *Node {
	g := coords.Graph()
	if len(inputShape) >= 2 {
		coordsSliceX := Slice(coords, AxisRange().Spacer(), AxisRange(0, 1))
		coordsSliceY := Slice(coords, AxisRange().Spacer(), AxisRange(1, 2))

		coordsSliceX = DivScalar(coordsSliceX, float64(inputShape[1]))
		coordsSliceY = DivScalar(coordsSliceY, float64(inputShape[0]))
		coords = Concatenate([]*Node{coordsSliceX, coordsSliceY}, -1)
	}

	coords = SubScalar(MulScalar(coords, 2.0), 1.0)

	posEmbedVar := scope.GetVariable("positional_embedding")
	posEmbedMat := posEmbedVar.NodeValue(g) // [2, H_dim // 2]

	// Perform matrix multiplication: coords x posEmbedMat
	// coords shape: [..., 2]
	// posEmbedMat shape: [2, H_dim // 2]
	s := coords.Shape().Dimensions
	lastDim := s[len(s)-1]
	n := 1
	for i := 0; i < len(s)-1; i++ {
		n *= s[i]
	}
	coords2D := Reshape(coords, n, lastDim)
	output2D := Dot(coords2D, posEmbedMat).Product()

	targetShape := append([]int{}, s...)
	targetShape[len(targetShape)-1] = posEmbedMat.Shape().Dimensions[1]
	coords = Reshape(output2D, targetShape...)

	coords = MulScalar(coords, 2.0*math.Pi)

	sinCoords := Sin(coords)
	cosCoords := Cos(coords)
	return Concatenate([]*Node{sinCoords, cosCoords}, -1)
}

// RepeatInterleaveAxis0 repeats a tensor along axis 0.
func RepeatInterleaveAxis0(x *Node, repeats int) *Node {
	dims := x.Shape().Dimensions
	batchSize := dims[0]
	x = ExpandAxes(x, 1)

	targetDims := make([]int, len(dims)+1)
	targetDims[0] = batchSize
	targetDims[1] = repeats
	copy(targetDims[2:], dims[1:])
	x = BroadcastToDims(x, targetDims...)

	finalDims := make([]int, len(dims))
	finalDims[0] = batchSize * repeats
	copy(finalDims[1:], dims[1:])
	return Reshape(x, finalDims...)
}

// BuildSinePositionEmbedding builds the FPN neck sine/cosine position embeddings.
func BuildSinePositionEmbedding(g *Graph, batchSize, height, width int, numPositionFeatures int, normalize bool, scale float64, temperature float64, dtype dtypes.DType) *Node {
	mask := Ones(g, shapes.Make(dtype, batchSize, height, width))

	yEmbed := CumSum(mask, 1)
	xEmbed := CumSum(mask, 2)

	if normalize {
		eps := 1e-6
		yLast := Slice(yEmbed, AxisRange(), AxisRange(-1), AxisRange())
		yEmbed = MulScalar(Div(yEmbed, AddScalar(yLast, eps)), scale)

		xLast := Slice(xEmbed, AxisRange(), AxisRange(), AxisRange(-1))
		xEmbed = MulScalar(Div(xEmbed, AddScalar(xLast, eps)), scale)
	}

	dimT := IotaFull(g, shapes.Make(dtype, numPositionFeatures))
	dimT = DivScalar(MulScalar(Floor(DivScalar(dimT, 2.0)), 2.0), float64(numPositionFeatures))
	dimT = Pow(Scalar(g, dtype, temperature), dimT)
	dimT = Reshape(dimT, 1, 1, 1, numPositionFeatures)

	posX := Div(ExpandAxes(xEmbed, -1), dimT)
	posY := Div(ExpandAxes(yEmbed, -1), dimT)

	posXEven := Slice(posX, AxisRange().Spacer(), AxisRange().Stride(2))
	posXOdd := Slice(posX, AxisRange().Spacer(), AxisRange(1).Stride(2))

	posYEven := Slice(posY, AxisRange().Spacer(), AxisRange().Stride(2))
	posYOdd := Slice(posY, AxisRange().Spacer(), AxisRange(1).Stride(2))

	sinX := Sin(posXEven)
	cosX := Cos(posXOdd)

	sinY := Sin(posYEven)
	cosY := Cos(posYOdd)

	posXInterleaved := Reshape(
		Concatenate([]*Node{ExpandAxes(sinX, -1), ExpandAxes(cosX, -1)}, -1),
		batchSize, height, width, numPositionFeatures,
	)
	posYInterleaved := Reshape(
		Concatenate([]*Node{ExpandAxes(sinY, -1), ExpandAxes(cosY, -1)}, -1),
		batchSize, height, width, numPositionFeatures,
	)

	pos := Concatenate([]*Node{posYInterleaved, posXInterleaved}, 3)
	return TransposeAllAxes(pos, 0, 3, 1, 2)
}

// GetPosEmbed generates Hiera Det positional embeddings.
func GetPosEmbed(scope *model.Scope, g *Graph, h, w int, config *Config) *Node {
	posEmbedVar := scope.GetVariable("pos_embed")
	posEmbed := posEmbedVar.NodeValue(g)

	posEmbedWindowVar := scope.GetVariable("pos_embed_window")
	posEmbedWindow := posEmbedWindowVar.NodeValue(g)

	// Interpolate pos_embed to [1, C, h, w]
	posEmbed = Interpolate(posEmbed, -1, -1, h, w).
		Bilinear().
		AlignCorner(true).
		HalfPixelCenters(false).
		Done()

	peDim := posEmbed.Shape().Dimensions
	winDim := posEmbedWindow.Shape().Dimensions

	tileH := peDim[2] / winDim[2]
	tileW := peDim[3] / winDim[3]

	// Tile window positional embedding
	windowEmbedReshaped := Reshape(posEmbedWindow, winDim[0], winDim[1], 1, winDim[2], 1, winDim[3])
	windowEmbedBroadcasted := BroadcastToDims(windowEmbedReshaped, winDim[0], winDim[1], tileH, winDim[2], tileW, winDim[3])
	windowEmbedTiled := Reshape(windowEmbedBroadcasted, winDim[0], winDim[1], tileH*winDim[2], tileW*winDim[3])

	posEmbed = Add(posEmbed, windowEmbedTiled)
	return TransposeAllAxes(posEmbed, 0, 2, 3, 1) // [1, h, w, C]
}

// Sam2HieraDetModel runs the vision transformer backbone.
func Sam2HieraDetModel(scope *model.Scope, x *Node, config *Config) ([]*Node, *Node) {
	bConfig := config.VisionConfig.BackboneConfig

	var stageEnds []int
	var cumSum int
	for _, numBlocks := range bConfig.BlocksPerStage {
		cumSum += numBlocks
		stageEnds = append(stageEnds, cumSum-1)
	}

	var intermediateHiddenStates []*Node
	var totalBlockIdx int

	dim := bConfig.EmbedDimPerStage[0]

	blocksScope := scope.In("blocks")

	for stageIdx, numBlocks := range bConfig.BlocksPerStage {
		dimOut := bConfig.EmbedDimPerStage[stageIdx]
		numHeads := bConfig.NumAttentionHeadsPerStage[stageIdx]

		for blockIdx := 0; blockIdx < numBlocks; blockIdx++ {
			var queryStride []int
			if blockIdx == 0 && stageIdx > 0 {
				queryStride = bConfig.QueryStride
			}

			windowSize := bConfig.WindowSizePerStage[stageIdx]
			if stageIdx > 0 && blockIdx == 0 {
				windowSize = bConfig.WindowSizePerStage[stageIdx-1]
			}

			useGlobalAttn := false
			for _, gBlockIdx := range bConfig.GlobalAttentionBlocks {
				if totalBlockIdx == gBlockIdx {
					useGlobalAttn = true
					break
				}
			}

			blockScope := blocksScope.In("%d", totalBlockIdx)
			x = Sam2MultiScaleBlock(blockScope, x, dim, dimOut, numHeads, windowSize, queryStride, useGlobalAttn)

			for _, endIdx := range stageEnds {
				if totalBlockIdx == endIdx {
					intermediateHiddenStates = append(intermediateHiddenStates, x)
					break
				}
			}

			dim = dimOut
			totalBlockIdx++
		}
	}

	return intermediateHiddenStates, x
}

// Sam2VisionNeck projects backbone features and performs FPN fusion.
func Sam2VisionNeck(scope *model.Scope, hiddenStates []*Node, config *Config) (fpnHiddenStates []*Node, fpnPositionEncoding []*Node) {
	g := hiddenStates[0].Graph()
	n := len(hiddenStates) - 1

	fpnFeatures := make([]*Node, 4)
	fpnPosEncodings := make([]*Node, 4)
	var prevFeatures *Node

	convsScope := scope.In("convs")

	for i := n; i >= 0; i-- {
		lat := TransposeAllAxes(hiddenStates[i], 0, 3, 1, 2)

		latConvScope := convsScope.In("%d", n-i)
		lat = layers.Convolution(latConvScope, lat).
			Channels(config.VisionConfig.FpnHiddenSize).
			KernelSize(config.VisionConfig.FpnKernelSize).
			Strides(config.VisionConfig.FpnStride).
			PadSame().
			ChannelsAxis(images.ChannelsFirst).
			Done()

		if !contains(config.VisionConfig.FpnTopDownLevels, i) || i == n {
			prevFeatures = lat
		} else {
			h := lat.Shape().Dimensions[2]
			w := lat.Shape().Dimensions[3]
			topDown := Interpolate(prevFeatures, -1, -1, h, w).
				Nearest().
				Done()
			prevFeatures = Add(lat, topDown)
		}

		fpnFeatures[i] = prevFeatures

		// Generate position embedding
		dims := prevFeatures.Shape().Dimensions
		posEnc := BuildSinePositionEmbedding(g, dims[0], dims[2], dims[3], config.VisionConfig.FpnHiddenSize/2, true, 2.0*math.Pi, 10000.0, prevFeatures.DType())
		fpnPosEncodings[i] = posEnc
	}

	// Select last num_feature_levels (3) and reverse to get high-to-low resolution
	fpnHiddenStates = []*Node{fpnFeatures[0], fpnFeatures[1], fpnFeatures[2]}
	fpnPositionEncoding = []*Node{fpnPosEncodings[0], fpnPosEncodings[1], fpnPosEncodings[2]}
	return fpnHiddenStates, fpnPositionEncoding
}

// EmbedPoints embeds point/label prompts.
func EmbedPoints(scope *model.Scope, points, labels *Node, pad bool, config *Config) (*Node, *Node, *Node) {
	g := points.Graph()
	points = AddScalar(points, 0.5)

	if pad {
		zeroPoint := Zeros(g, shapes.Make(points.DType(), points.Shape().Dimensions[0], points.Shape().Dimensions[1], 1, 2))
		points = Concatenate([]*Node{points, zeroPoint}, 2)

		minusOne := Scalar(g, labels.DType(), -1)
		minusOne = BroadcastToDims(minusOne, labels.Shape().Dimensions[0], labels.Shape().Dimensions[1], 1)
		labels = Concatenate([]*Node{labels, minusOne}, 2)
	}

	pointEmbedding := Sam2PositionalEmbedding(scope.In("shared_embedding"), points, []int{config.PromptEncoderConfig.ImageSize, config.PromptEncoderConfig.ImageSize})

	labelsExpanded := ExpandAxes(labels, -1)
	targetShape := pointEmbedding.Shape().Dimensions

	notAPointEmbedVar := scope.In("not_a_point_embed").GetVariable("embeddings")
	notAPointEmbed := notAPointEmbedVar.NodeValue(g)

	isMinusOne := Equal(labelsExpanded, Scalar(g, labelsExpanded.DType(), -1))
	isMinusOne = BroadcastToDims(isMinusOne, targetShape...)
	notAPointEmbedBroadcasted := BroadcastToDims(Reshape(notAPointEmbed, 1, 1, 1, config.PromptEncoderConfig.HiddenSize), targetShape...)
	pointEmbedding = Where(isMinusOne, notAPointEmbedBroadcasted, pointEmbedding)

	isNotMinusTen := NotEqual(labelsExpanded, Scalar(g, labelsExpanded.DType(), -10))
	isNotMinusTen = BroadcastToDims(isNotMinusTen, targetShape...)
	pointEmbedding = Where(isNotMinusTen, pointEmbedding, ZerosLike(pointEmbedding))

	labelsClamped := Max(labels, Scalar(g, labels.DType(), 0))
	pointEmbedValues := layers.Embedding(scope.In("point_embed"), labelsClamped, pointEmbedding.DType(), 4, config.PromptEncoderConfig.HiddenSize)

	isGreaterOrEqualZero := GreaterOrEqual(labelsExpanded, Scalar(g, labelsExpanded.DType(), 0))
	isGreaterOrEqualZero = BroadcastToDims(isGreaterOrEqualZero, targetShape...)
	pointEmbedding = Add(pointEmbedding, Mul(pointEmbedValues, ConvertDType(isGreaterOrEqualZero, pointEmbedding.DType())))

	return pointEmbedding, labels, pointEmbedValues
}

// EmbedBoxes embeds box prompts.
func EmbedBoxes(scope *model.Scope, boxes *Node, inputImageSize int, hiddenSize int) *Node {
	g := boxes.Graph()
	boxes = AddScalar(boxes, 0.5)

	batchSize := boxes.Shape().Dimensions[0]
	pointBatchSize := boxes.Shape().Dimensions[1]
	coords := Reshape(boxes, batchSize, pointBatchSize, 2, 2)

	zeroPoint := Zeros(g, shapes.Make(coords.DType(), batchSize, pointBatchSize, 1, 2))
	coords = Concatenate([]*Node{coords, zeroPoint}, 2)

	cornerEmbedding := Sam2PositionalEmbedding(scope.In("shared_embedding"), coords, []int{inputImageSize, inputImageSize})

	pointEmbedVar := scope.In("point_embed").GetVariable("embeddings")
	pointEmbedWeight := pointEmbedVar.NodeValue(g)

	notAPointEmbedVar := scope.In("not_a_point_embed").GetVariable("embeddings")
	notAPointEmbedWeight := notAPointEmbedVar.NodeValue(g)

	c0 := Squeeze(Slice(cornerEmbedding, AxisRange(), AxisRange(), AxisElem(0), AxisRange()), 2)
	c1 := Squeeze(Slice(cornerEmbedding, AxisRange(), AxisRange(), AxisElem(1), AxisRange()), 2)

	pe2 := Squeeze(Slice(pointEmbedWeight, AxisElem(2), AxisRange()), 0)
	pe3 := Squeeze(Slice(pointEmbedWeight, AxisElem(3), AxisRange()), 0)

	c0 = Add(c0, BroadcastToDims(Reshape(pe2, 1, 1, -1), c0.Shape().Dimensions...))
	c1 = Add(c1, BroadcastToDims(Reshape(pe3, 1, 1, -1), c1.Shape().Dimensions...))

	c2 := BroadcastToDims(Reshape(notAPointEmbedWeight, 1, 1, -1), c0.Shape().Dimensions...)

	return Concatenate([]*Node{ExpandAxes(c0, 2), ExpandAxes(c1, 2), ExpandAxes(c2, 2)}, 2)
}

// Sam2MaskEmbedding embeds input masks.
func Sam2MaskEmbedding(scope *model.Scope, masks *Node, config *Config) *Node {
	c1 := layers.Convolution(scope.In("conv1"), masks).
		Channels(config.PromptEncoderConfig.MaskInputChannels / 4).
		KernelSize(2).
		Strides(2).
		ChannelsAxis(images.ChannelsFirst).
		Done()
	c1 = norm.LayerNorm(scope.In("layer_norm1"), c1, 1).Epsilon(1e-6).Done()
	c1 = activation.Apply(activation.FromName(config.PromptEncoderConfig.HiddenAct), c1)

	c2 := layers.Convolution(scope.In("conv2"), c1).
		Channels(config.PromptEncoderConfig.MaskInputChannels).
		KernelSize(2).
		Strides(2).
		ChannelsAxis(images.ChannelsFirst).
		Done()
	c2 = norm.LayerNorm(scope.In("layer_norm2"), c2, 1).Epsilon(1e-6).Done()
	c2 = activation.Apply(activation.FromName(config.PromptEncoderConfig.HiddenAct), c2)

	c3 := layers.Convolution(scope.In("conv3"), c2).
		Channels(config.PromptEncoderConfig.HiddenSize).
		KernelSize(1).
		Strides(1).
		ChannelsAxis(images.ChannelsFirst).
		Done()

	return c3
}

// twoWayAttentionBlock runs one layer of self- and cross-attention blocks.
func twoWayAttentionBlock(scope *model.Scope, queries, keys, queryPE, keyPE *Node, skipFirstPE bool) (outQueries, outKeys, queriesAfterLN1, queriesAfterLN2, queriesAfterLN3, keysAfterLN4 *Node) {
	// Self-attention
	if skipFirstPE {
		queries = twoWayAttention(scope.In("self_attn"), queries, queries, queries, nil, 1, 8)
	} else {
		q := Add(queries, queryPE)
		selfAttnOut := twoWayAttention(scope.In("self_attn"), q, q, queries, nil, 1, 8)
		queries = Add(queries, selfAttnOut)
	}
	queries = norm.LayerNorm(scope.In("layer_norm1"), queries, 2).Epsilon(1e-5).Done()
	queriesAfterLN1 = queries

	// Cross-attention: tokens attending to image
	q := Add(queries, queryPE)
	k := Add(keys, keyPE)
	crossAttnOut1 := twoWayAttention(scope.In("cross_attn_token_to_image"), q, k, keys, nil, 2, 8)
	queries = Add(queries, crossAttnOut1)
	queries = norm.LayerNorm(scope.In("layer_norm2"), queries, 2).Epsilon(1e-5).Done()
	queriesAfterLN2 = queries

	// MLP
	mlpScope := scope.In("mlp")
	mlpProjIn := layers.Dense(mlpScope.In("proj_in"), queries, true, 2048)
	mlpProjInAct := activation.Relu(mlpProjIn)
	mlpOut := layers.Dense(mlpScope.In("proj_out"), mlpProjInAct, true, 256)
	queries = Add(queries, mlpOut)
	queries = norm.LayerNorm(scope.In("layer_norm3"), queries, 2).Epsilon(1e-5).Done()

	// Cross-attention: image attending to tokens
	q = Add(queries, queryPE)
	k = Add(keys, keyPE)
	crossAttnOut2 := twoWayAttention(scope.In("cross_attn_image_to_token"), k, q, queries, nil, 2, 8)
	keys = Add(keys, crossAttnOut2)
	keys = norm.LayerNorm(scope.In("layer_norm4"), keys, 2).Epsilon(1e-5).Done()
	keysAfterLN4 = keys

	return queries, keys, queriesAfterLN1, queriesAfterLN2, queries, keysAfterLN4
}

// twoWayTransformer runs the transformer stack.
func twoWayTransformer(scope *model.Scope, pointEmbeddings, imageEmbeddings, imagePE *Node) (outPointEmbeds, outImageEmbeds, queriesL0AfterLN1, queriesL0AfterLN2, queriesL0AfterLN3, keysL0AfterLN4 *Node) {
	// Flatten image embeddings [B*PB, C, H, W] -> [B*PB, H*W, C]
	dims := imageEmbeddings.Shape().Dimensions
	bp := dims[0]
	c := dims[1]
	h := dims[2]
	w := dims[3]

	// Reshape pointEmbeddings to rank 3: [B*PB, num_tokens, C]
	queries := Reshape(pointEmbeddings, bp, -1, c)
	keys := TransposeAllAxes(Reshape(imageEmbeddings, bp, c, h*w), 0, 2, 1)

	// Reshape image PE [B*PB, C, H, W] -> [B*PB, H*W, C]
	keyPE := TransposeAllAxes(Reshape(imagePE, bp, c, h*w), 0, 2, 1)

	queryPE := queries

	layersScope := scope.In("layers")

	// Run Two-Way Attention layers
	for i := 0; i < 2; i++ {
		layerScope := layersScope.In("%d", i)
		var qLN1, qLN2, qLN3, kLN4 *Node
		queries, keys, qLN1, qLN2, qLN3, kLN4 = twoWayAttentionBlock(layerScope, queries, keys, queryPE, keyPE, i == 0)
		if i == 1 {
			queriesL0AfterLN1 = qLN1
			queriesL0AfterLN2 = qLN2
			queriesL0AfterLN3 = qLN3
			keysL0AfterLN4 = kLN4
		}
	}

	// Final attention token to image
	q := Add(queries, queryPE)
	k := Add(keys, keyPE)
	finalAttnOut := twoWayAttention(scope.In("final_attn_token_to_image"), q, k, keys, nil, 2, 8)
	queries = Add(queries, finalAttnOut)
	queries = norm.LayerNorm(scope.In("layer_norm_final_attn"), queries, 2).Epsilon(1e-5).Done()

	return queries, keys, queriesL0AfterLN1, queriesL0AfterLN2, queriesL0AfterLN3, keysL0AfterLN4
}

// Sam2FeedForward3Layers implements a 3-layer MLP.
func Sam2FeedForward3Layers(scope *model.Scope, x *Node, inDim, hiddenDim, outDim int, hiddenAct string) *Node {
	x = layers.Dense(scope.In("proj_in"), x, true, hiddenDim)
	x = activation.Apply(activation.FromName(hiddenAct), x)

	x = layers.Dense(scope.In("layers").In("0"), x, true, hiddenDim)
	x = activation.Apply(activation.FromName(hiddenAct), x)

	x = layers.Dense(scope.In("proj_out"), x, true, outDim)
	return x
}

// maskDecoder decodes masks from the image and prompt embeddings.
func maskDecoder(scope *model.Scope, imageEmbeddings, imagePE, sparseEmbeds, denseEmbeds *Node, multimaskOutput bool, highResFeatures []*Node, config *Config) (masks, iouScores *Node, iouTokenOut, maskTokensOut, upscaled, upscaled2, hyperIn *Node, qL0AfterLN1, qL0AfterLN2, qL0AfterLN3, kL0AfterLN4 *Node) {
	g := imageEmbeddings.Graph()

	batchSize := imageEmbeddings.Shape().Dimensions[0]
	numChannels := imageEmbeddings.Shape().Dimensions[1]
	height := imageEmbeddings.Shape().Dimensions[2]
	width := imageEmbeddings.Shape().Dimensions[3]

	pointBatchSize := sparseEmbeds.Shape().Dimensions[1]

	objScoreToken := scope.In("obj_score_token").GetVariable("embeddings").NodeValue(g)
	iouToken := scope.In("iou_token").GetVariable("embeddings").NodeValue(g)
	maskTokens := scope.In("mask_tokens").GetVariable("embeddings").NodeValue(g)

	outputTokens := Concatenate([]*Node{objScoreToken, iouToken, maskTokens}, 0)
	outputTokens = BroadcastToDims(Reshape(outputTokens, 1, 1, 6, config.MaskDecoderConfig.HiddenSize), batchSize, pointBatchSize, 6, config.MaskDecoderConfig.HiddenSize)

	pointEmbeddings := Concatenate([]*Node{outputTokens, sparseEmbeds}, 2)

	// Add dense_prompt_embeddings to image_embeddings
	imageEmbeddings = Add(imageEmbeddings, denseEmbeds)
	imageEmbeddings = RepeatInterleaveAxis0(imageEmbeddings, pointBatchSize)
	imagePE = RepeatInterleaveAxis0(imagePE, pointBatchSize)

	// Run transformer
	var imageEmbedsFlat *Node
	pointEmbeddings, imageEmbedsFlat, qL0AfterLN1, qL0AfterLN2, qL0AfterLN3, kL0AfterLN4 = twoWayTransformer(scope.In("transformer"), pointEmbeddings, imageEmbeddings, imagePE)
	pointEmbeddings = Reshape(pointEmbeddings, batchSize, pointBatchSize, -1, config.MaskDecoderConfig.HiddenSize)

	iouTokenOut = Squeeze(Slice(pointEmbeddings, AxisRange(), AxisRange(), AxisElem(1), AxisRange()), 2)
	maskTokensOut = Slice(pointEmbeddings, AxisRange(), AxisRange(), AxisRange(2, 6), AxisRange())

	// Reshape back image_embeddings: [B * PB, C, H, W]
	imageEmbedsReshaped := TransposeAllAxes(Reshape(imageEmbedsFlat, batchSize*pointBatchSize, height, width, numChannels), 0, 3, 1, 2)

	featS0 := RepeatInterleaveAxis0(highResFeatures[0], pointBatchSize)
	featS1 := RepeatInterleaveAxis0(highResFeatures[1], pointBatchSize)

	// Upscale image embeddings:
	uConv1Scope := scope.In("upscale_conv1").In("conv")
	uConv1W := uConv1Scope.GetVariable("weights").NodeValue(g)
	uConv1W = TransposeAllAxes(uConv1W, 1, 0, 2, 3)
	upscaled = Einsum("bchw,ockl->bohkwl", imageEmbedsReshaped, uConv1W)
	upscaled = Reshape(upscaled, batchSize*pointBatchSize, 64, height*2, width*2)

	uConv1B := uConv1Scope.GetVariable("biases").NodeValue(g)
	upscaled = Add(upscaled, Reshape(uConv1B, 1, 64, 1, 1))

	upscaled = Add(upscaled, featS1)
	upscaled = norm.LayerNorm(scope.In("upscale_layer_norm"), upscaled, 1).Epsilon(1e-6).Done()
	upscaled = activation.Apply(activation.FromName(config.MaskDecoderConfig.HiddenAct), upscaled)

	// upscale_conv2: ConvTranspose2d with kernel=2, stride=2
	uConv2Scope := scope.In("upscale_conv2").In("conv")
	uConv2W := uConv2Scope.GetVariable("weights").NodeValue(g)
	uConv2W = TransposeAllAxes(uConv2W, 1, 0, 2, 3)
	upscaled2 = Einsum("bchw,ockl->bohkwl", upscaled, uConv2W)
	upscaled2 = Reshape(upscaled2, batchSize*pointBatchSize, 32, height*4, width*4)

	uConv2B := uConv2Scope.GetVariable("biases").NodeValue(g)
	upscaled2 = Add(upscaled2, Reshape(uConv2B, 1, 32, 1, 1))

	upscaled2 = Add(upscaled2, featS0)
	upscaled2 = activation.Apply(activation.FromName(config.MaskDecoderConfig.HiddenAct), upscaled2)

	// Run output MLPs to get hypernetworks weights
	outputMlpsScope := scope.In("output_hypernetworks_mlps")
	var hyperInList []*Node
	for i := 0; i < 4; i++ {
		tokenI := Squeeze(Slice(maskTokensOut, AxisRange(), AxisRange(), AxisElem(i), AxisRange()), 2)
		mlpI := Sam2FeedForward3Layers(outputMlpsScope.In("%d", i), tokenI, config.MaskDecoderConfig.HiddenSize, config.MaskDecoderConfig.HiddenSize, config.MaskDecoderConfig.HiddenSize/8, config.MaskDecoderConfig.HiddenAct)
		hyperInList = append(hyperInList, ExpandAxes(mlpI, 2))
	}
	hyperIn = Concatenate(hyperInList, 2) // [B, PB, 4, 32]
	hyperIn = Reshape(hyperIn, batchSize*pointBatchSize, 4, 32)
	masks = Einsum("btc,bcxy->btxy", hyperIn, upscaled2)
	masks = Reshape(masks, batchSize, pointBatchSize, 4, height*4, width*4)

	iouPred := Sam2FeedForward3Layers(scope.In("iou_prediction_head"), iouTokenOut, config.MaskDecoderConfig.HiddenSize, config.MaskDecoderConfig.IoUHeadHiddenDim, 4, config.MaskDecoderConfig.HiddenAct)
	iouPred = Sigmoid(iouPred)
	iouScores = Reshape(iouPred, batchSize, pointBatchSize, 4)

	// Slicing out multi-mask or single-mask
	if multimaskOutput {
		masks = Slice(masks, AxisRange(), AxisRange(), AxisRange(1, 4), AxisRange(), AxisRange())
		iouScores = Slice(iouScores, AxisRange(), AxisRange(), AxisRange(1, 4))
	} else {
		masks = Slice(masks, AxisRange(), AxisRange(), AxisRange(0, 1), AxisRange(), AxisRange())
		iouScores = Slice(iouScores, AxisRange(), AxisRange(), AxisRange(0, 1))
	}

	return masks, iouScores, iouTokenOut, maskTokensOut, upscaled, upscaled2, hyperIn, qL0AfterLN1, qL0AfterLN2, qL0AfterLN3, kL0AfterLN4
}

// GetImageWidePositionalEmbeddings builds the 2D positional embeddings for the mask decoder.
func GetImageWidePositionalEmbeddings(scope *model.Scope, g *Graph, config *Config) *Node {
	h := 64
	w := 64

	grid := Ones(g, shapes.Make(dtypes.Float32, h, w))

	yEmbed := CumSum(grid, 0)
	yEmbed = SubScalar(yEmbed, 0.5)
	yEmbed = DivScalar(yEmbed, float64(h))

	xEmbed := CumSum(grid, 1)
	xEmbed = SubScalar(xEmbed, 0.5)
	xEmbed = DivScalar(xEmbed, float64(w))

	coords := Concatenate([]*Node{ExpandAxes(xEmbed, -1), ExpandAxes(yEmbed, -1)}, -1)
	coords = ExpandAxes(ExpandAxes(coords, 0), 0)

	posEmbed := Sam2PositionalEmbedding(scope.In("shared_image_embedding"), coords, nil)
	posEmbed = Squeeze(posEmbed, 1)

	return TransposeAllAxes(posEmbed, 0, 3, 1, 2)
}

// Forward performs the full model forward pass.
func Forward(scope *model.Scope, pixelValues *Node, inputPoints, inputLabels, inputBoxes, inputMasks *Node, config *Config, multimaskOutput bool) (predMasks, iouScores *Node, fpnHiddenStates []*Node, sparseEmbeds, denseEmbeds *Node, iouTokenOut, maskTokensOut, upscaled, upscaled2, hyperIn *Node) {
	g := pixelValues.Graph()
	batchSize := pixelValues.Shape().Dimensions[0]

	visionScope := scope.In("vision_encoder")
	backboneScope := visionScope.In("backbone")
	neckScope := visionScope.In("neck")
	promptScope := scope.In("prompt_encoder")
	maskDecoderScope := scope.In("mask_decoder")

	// 1. Vision Encoder
	// patch_embed
	x := layers.Convolution(backboneScope.In("patch_embed").In("projection"), pixelValues).
		Channels(config.VisionConfig.BackboneConfig.EmbedDimPerStage[0]).
		KernelSize(7).
		Strides(4).
		PadSame().
		ChannelsAxis(images.ChannelsFirst).
		Done()

	// Permute to [batch, H, W, C]
	x = TransposeAllAxes(x, 0, 2, 3, 1)

	// Add positional embedding
	h := x.Shape().Dimensions[1]
	w := x.Shape().Dimensions[2]
	x = Add(x, GetPosEmbed(backboneScope, g, h, w, config))

	// Backbone
	backboneFeatures, _ := Sam2HieraDetModel(backboneScope, x, config)

	// Neck
	var fpnPE []*Node
	fpnHiddenStates, fpnPE = Sam2VisionNeck(neckScope, backboneFeatures, config)
	_ = fpnPE

	// Get last feature map and add no_memory_embedding
	lastFeat := fpnHiddenStates[2] // shape [batch, 256, 64, 64]
	noMemVar := scope.GetVariable("no_memory_embedding")
	noMemEmbed := noMemVar.NodeValue(g)
	noMemEmbed = TransposeAllAxes(noMemEmbed, 0, 2, 1)
	noMemEmbed = ExpandAxes(noMemEmbed, -1)

	lastFeat = Add(lastFeat, noMemEmbed)
	fpnHiddenStates[2] = lastFeat

	// 2. Preprocess prompt encoder inputs
	if inputPoints == nil && inputBoxes == nil {
		inputPoints = Zeros(g, shapes.Make(dtypes.Float32, batchSize, 1, 1, 2))
		inputLabels = FillScalar(g, shapes.Make(dtypes.Int32, batchSize, 1, 1), -1.0)
	} else if inputPoints != nil && inputLabels == nil {
		inputLabels = Ones(g, shapes.Make(dtypes.Int32, inputPoints.Shape().Dimensions[0], inputPoints.Shape().Dimensions[1], inputPoints.Shape().Dimensions[2]))
	}

	// 3. Prompt Encoder
	if inputPoints != nil {
		sparseEmbeds, _, _ = EmbedPoints(promptScope, inputPoints, inputLabels, inputBoxes == nil, config)
	}
	if inputBoxes != nil {
		boxEmbeds := EmbedBoxes(promptScope, inputBoxes, config.PromptEncoderConfig.ImageSize, config.PromptEncoderConfig.HiddenSize)
		if sparseEmbeds == nil {
			sparseEmbeds = boxEmbeds
		} else {
			sparseEmbeds = Concatenate([]*Node{sparseEmbeds, boxEmbeds}, 2)
		}
	}

	if inputMasks != nil {
		if inputMasks.Shape().Dimensions[3] != 256 || inputMasks.Shape().Dimensions[4] != 256 {
			inputMasks = Interpolate(inputMasks, -1, -1, 256, 256).
				Bilinear().
				Done()
		}
		inputMasksFlat := Squeeze(inputMasks, 1)
		denseEmbeds = Sam2MaskEmbedding(promptScope.In("mask_embed"), inputMasksFlat, config)
	} else {
		noMaskEmbedVar := promptScope.In("no_mask_embed").GetVariable("embeddings")
		noMaskEmbed := noMaskEmbedVar.NodeValue(g)
		noMaskEmbed = Reshape(noMaskEmbed, 1, 256, 1, 1)
		denseEmbeds = BroadcastToDims(noMaskEmbed, batchSize, 256, 64, 64)
	}

	// 4. Precompute high-res features projected by conv_s0 and conv_s1
	featS0 := layers.Convolution(maskDecoderScope.In("conv_s0"), fpnHiddenStates[0]).
		Channels(32).
		KernelSize(1).
		Strides(1).
		ChannelsAxis(images.ChannelsFirst).
		Done()

	featS1 := layers.Convolution(maskDecoderScope.In("conv_s1"), fpnHiddenStates[1]).
		Channels(64).
		KernelSize(1).
		Strides(1).
		ChannelsAxis(images.ChannelsFirst).
		Done()

	highResProjected := []*Node{featS0, featS1}

	// Wide positional embeddings:
	imageWidePE := GetImageWidePositionalEmbeddings(scope, g, config)

	// 5. Mask Decoder
	predMasks, iouScores, iouTokenOut, maskTokensOut, upscaled, upscaled2, hyperIn, _, _, _, _ = maskDecoder(maskDecoderScope, fpnHiddenStates[2], imageWidePE, sparseEmbeds, denseEmbeds, multimaskOutput, highResProjected, config)

	return predMasks, iouScores, fpnHiddenStates, sparseEmbeds, denseEmbeds, iouTokenOut, maskTokensOut, upscaled, upscaled2, hyperIn
}

func contains(slice []int, val int) bool {
	for _, item := range slice {
		if item == val {
			return true
		}
	}
	return false
}
