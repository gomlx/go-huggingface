// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package timestral_test

import (
	"fmt"
	"math"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/compute/gobackend"
	"github.com/gomlx/go-huggingface/models/timestral"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
)

// ExampleForecaster_Forecast demonstrates how to forecast a time series with missing (NaN) values
// and extract median and quantile predictions using [timestral.Forecaster].
//
// In real usage, load a pretrained checkpoint via hub.New(timestral.RepoTimestral28M) and
// timestral.LoadModel(repo), then call timestral.NewForecaster(backend, tmModel).
func ExampleForecaster_Forecast() {
	backend, err := compute.NewWithConfig("go")
	if err != nil {
		panic(err)
	}

	// Here we construct a tiny synthetic model so the example runs offline in milliseconds.
	// In production, use:
	//   repo := hub.New(timestral.RepoTimestral28M)
	//   tmModel, err := timestral.LoadModel(repo)
	//   fc, err := timestral.NewForecaster(backend, tmModel)
	cfg := timestral.DefaultConfig()
	cfg.PatchSize = 8
	cfg.DModel = 16
	cfg.NumHeads = 2
	cfg.MLPHiddenDim = 32
	cfg.BlockSchedule = []timestral.LayerKind{timestral.LayerTime, timestral.LayerTimeVariate}
	cfg.BlockSize = 4
	cfg.ContextLength = 64

	tmModel, err := timestral.NewModel(cfg)
	if err != nil {
		panic(err)
	}
	store := model.NewStore()
	fc := timestral.NewForecasterWithStore(backend, tmModel, store)

	// Input context: 1 batch item, 1 variate, 24 time steps (with a missing NaN value).
	context := [][]float32{
		{
			10, 11, 12, 13, 14, 15, float32(math.NaN()), 17,
			18, 19, 20, 21, 22, 23, 24, 25,
			26, 27, 28, 29, 30, 31, 32, 33,
		},
	}

	const horizon = 12
	res, err := fc.Forecast(context, horizon).Done()
	if err != nil {
		panic(err)
	}
	defer res.Close()

	median, err := res.Median()
	if err != nil {
		panic(err)
	}
	defer median.FinalizeAll()

	q90, err := res.Quantile(0.9)
	if err != nil {
		panic(err)
	}
	defer q90.FinalizeAll()

	fmt.Printf("Quantiles shape: %v\n", res.Quantiles.Shape().Dimensions)
	fmt.Printf("Median shape:    %v\n", median.Shape().Dimensions)
	fmt.Printf("Q90 shape:       %v\n", q90.Shape().Dimensions)

	// Output:
	// Quantiles shape: [1 1 12 9]
	// Median shape:    [1 1 12]
	// Q90 shape:       [1 1 12]
}

// ExampleModel_BuildGraph demonstrates how to embed Timestral directly into a compiled GoMLX
// computation graph using [timestral.Model.BuildGraph].
func ExampleModel_BuildGraph() {
	backend, err := compute.NewWithConfig("go")
	if err != nil {
		panic(err)
	}

	cfg := timestral.DefaultConfig()
	cfg.PatchSize = 8
	cfg.DModel = 16
	cfg.NumHeads = 2
	cfg.MLPHiddenDim = 32
	cfg.BlockSchedule = []timestral.LayerKind{timestral.LayerTime, timestral.LayerTimeVariate}
	cfg.BlockSize = 4
	cfg.ContextLength = 64

	tmModel, err := timestral.NewModel(cfg)
	if err != nil {
		panic(err)
	}

	store := model.NewStore()
	const horizonSteps = 8
	exec, err := model.NewExec(backend, store, func(scope *model.Scope, x *graph.Node) *graph.Node {
		return tmModel.BuildGraph(scope, x, horizonSteps).
			SymmetricAveraging(true).
			ClipNonNegative(true).
			Done()
	})
	if err != nil {
		panic(err)
	}

	// Pass a 2D [Batch=2, ContextSteps=16] tensor.
	input := tensors.FromFlatDataAndDimensions(make([]float32, 2*16), 2, 16)
	defer input.FinalizeAll()

	quantiles, err := exec.Exec1(input)
	if err != nil {
		panic(err)
	}
	defer quantiles.FinalizeAll()

	fmt.Printf("Output shape: %v\n", quantiles.Shape().Dimensions)

	// Output:
	// Output shape: [2 8 9]
}
