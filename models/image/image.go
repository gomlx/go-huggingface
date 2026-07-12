// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package image implements helpers for image models.
package image

import (
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/support/exceptions"
)

// NodeOrFloat constraints to a *Node or a []float32 or []float64.
type NodeOrFloats interface{ *Node | []float32 | []float64 }

// PreprocessGraph takes an image tensor of shape [batch, height, width, channels] (or [height, width, channels]),
// resizes it to targetHeight x targetWidth using bilinear interpolation,
// converts it to channels-first [batch, channels, targetHeight, targetWidth],
// and normalizes it with the given mean and std.
//
// The parameters mean and std can be either given as a Node or as a slice of floats (float32 or float64)
// and should have shape [channels].
func PreprocessGraph[T NodeOrFloats](x *Node, targetHeight, targetWidth int, mean, std T) *Node {
	g := x.Graph()
	if x.Rank() == 3 {
		x = ExpandAxes(x, 0) // [1, height, width, channels]
	}

	// Bilinear interpolate to target size
	x = Interpolate(x, -1, targetHeight, targetWidth, -1).Bilinear().Done()

	// Convert from [batch, height, width, channels] to [batch, channels, height, width] (channels first)
	x = TransposeAllAxes(x, 0, 3, 1, 2)

	// Convert to float32/float64 if needed, and rescale to [0, 1] by dividing by 255.
	dtype := x.DType()
	if !dtype.IsFloat() {
		x = ConvertDType(x, dtypes.Float32)
		dtype = dtypes.Float32
		x = DivScalar(x, 255.0)
	}

	// Normalize: (x - mean) / std
	var meanNode, stdNode *Node
	switch any(mean).(type) {
	case *Node:
		meanNode = any(mean).(*Node)
		stdNode = any(std).(*Node)
	case []float32:
		meanNode = Const(g, any(mean).([]float32))
		stdNode = Const(g, any(std).([]float32))
	case []float64:
		meanNode = Const(g, any(mean).([]float64))
		stdNode = Const(g, any(std).([]float64))
	default:
		exceptions.Panicf("invalid type for mean or std: %T, %T", mean, std)
	}

	// Reshape mean and std to [1, channels, 1, 1] for broadcasting
	meanNode = Reshape(meanNode, 1, -1, 1, 1)
	stdNode = Reshape(stdNode, 1, -1, 1, 1)

	// Convert mean/std to matching dtype if needed
	if meanNode.DType() != dtype {
		meanNode = ConvertDType(meanNode, dtype)
	}
	if stdNode.DType() != dtype {
		stdNode = ConvertDType(stdNode, dtype)
	}

	x = Div(Sub(x, meanNode), stdNode)
	return x
}
