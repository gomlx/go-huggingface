// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/compute/gobackend"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/timestral"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	mlmodel "github.com/gomlx/gomlx/ml/model"
	"github.com/stretchr/testify/require"
)

func testModelAccuracyAgainstPyTorch(t *testing.T, modelID, resultsFile string) {
	t.Helper()
	if testing.Short() {
		t.Skipf("Skipping %s PyTorch reference test in -short mode", modelID)
	}

	const (
		batch        = 2
		contextSteps = 12096
		horizonSteps = 864
	)

	context := loadOrGenerateContext(contextSteps)
	reference := readF32(resultsFile)

	input := make([]float32, batch*contextSteps)
	copy(input, context)
	for i, v := range context {
		input[contextSteps+i] = -v
	}

	backend, err := compute.NewWithConfig("go")
	require.NoError(t, err)

	repo := hub.New(modelID)
	if err := repo.FetchFiles("config.json", "model.safetensors"); err != nil {
		t.Skipf("Skipping test because %s could not be fetched/cached: %v", modelID, err)
	}
	tmModel, err := timestral.LoadModel(repo)
	require.NoError(t, err)

	store := mlmodel.NewStore().WithVariableAsConst(true)
	require.NoError(t, tmModel.LoadStore(backend, store))
	defer store.Finalize()

	exec, err := mlmodel.NewExec(backend, store, func(scope *mlmodel.Scope, x *graph.Node) *graph.Node {
		return tmModel.BuildGraph(scope, x, horizonSteps).
			SymmetricAveraging(false).
			ClipNonNegative(false).
			Done()
	})
	require.NoError(t, err)

	x := tensors.FromFlatDataAndDimensions(input, batch, contextSteps)
	defer x.FinalizeAll()
	y, err := exec.Call1(x)
	require.NoError(t, err)
	defer y.FinalizeAll()

	got, err := tensors.CopyFlatData[float32](y)
	require.NoError(t, err)
	require.Equal(t, len(reference), len(got))

	var maxDiff float64
	for i, v := range got {
		diff := math.Abs(float64(v) - float64(reference[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	t.Logf("%s: max abs diff vs PyTorch (%s) = %.3e (across %d values)", modelID, resultsFile, maxDiff, len(got))
	require.Less(t, maxDiff, 1e-4, "max abs diff vs PyTorch exceeded 1e-4")
}

func TestTimestral28MAccuracyAgainstPyTorch(t *testing.T) {
	testModelAccuracyAgainstPyTorch(t, timestral.RepoTimestral28M, "model-28m_test_results.f32")
}

func TestTimestral123MAccuracyAgainstPyTorch(t *testing.T) {
	testModelAccuracyAgainstPyTorch(t, timestral.RepoTimestral123M, "model-123m_test_results.f32")
}
