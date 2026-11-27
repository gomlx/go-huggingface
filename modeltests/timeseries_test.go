// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package modeltests

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"
	"time"

	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/models/timestral"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	mlmodel "github.com/gomlx/gomlx/ml/model"
	"github.com/stretchr/testify/require"
)

func runTimeSeriesModel(t *testing.T, loaded loadedSpec, backendConfigs []string) {
	spec := loaded.Spec

	for _, variant := range spec.Variants {
		t.Run(variant.Name, func(t *testing.T) {
			require.NotNil(t, variant.TimeSeries, "variant %q missing timeseries spec", variant.Name)
			ts := variant.TimeSeries

			repoID := spec.RepoID
			if variant.RepoID != "" {
				repoID = variant.RepoID
			}
			revision := spec.Revision
			if variant.Revision != "" {
				revision = variant.Revision
			}
			opts := mergeModelOptions(spec.ModelOptions, variant.ModelOptions)

			repo, ok := openRepoForSpec(t, repoID, revision, spec.Gated, "model.safetensors")
			if !ok {
				return
			}
			if err := repo.FetchFiles("config.json", "model.safetensors"); err != nil {
				if !*flagStrict {
					t.Skipf("Skipping %s because files could not be fetched: %v", repoID, err)
				}
				t.Fatalf("Failed to fetch %s: %v", repoID, err)
			}

			contextData, err := readF32File(filepath.Join(loaded.Dir, ts.ContextFile))
			require.NoError(t, err)
			require.GreaterOrEqual(t, len(contextData), ts.ContextSteps)
			contextData = contextData[:ts.ContextSteps]

			reference, err := readF32File(filepath.Join(loaded.Dir, ts.ResultsFile))
			require.NoError(t, err)

			batch := ts.Batch
			if batch == 0 {
				batch = 2
			}
			input := make([]float32, batch*ts.ContextSteps)
			copy(input, contextData)
			if batch >= 2 {
				for i, v := range contextData {
					input[ts.ContextSteps+i] = -v
				}
			}

			maxAbsTol := ts.MaxAbsDiff
			if maxAbsTol == 0 {
				maxAbsTol = 1e-4
			}

			for _, bCfg := range backendConfigs {
				labelForName := bCfg
				if labelForName == "" {
					labelForName = "default"
				}
				t.Run(formatBackendSubtestName(bCfg, labelForName), func(t *testing.T) {
					if shouldSkipBackend(opts.SkipBackends, bCfg, labelForName) {
						t.Skipf("Backend %q listed in skip_backends", labelForName)
					}

					memTracker := startMemoryTracker()
					row := runReportRow{
						Model:   spec.Name,
						Variant: variant.Name,
						Backend: labelForName,
					}

					backend, resolvedLabel, err := createBackend(bCfg)
					if err != nil {
						memTracker.Stop()
						t.Fatalf("Failed to initialize backend %q: %v", bCfg, err)
					}
					row.Backend = resolvedLabel
					if shouldSkipBackend(opts.SkipBackends, bCfg, backend.Name()) {
						backend.Finalize()
						memTracker.Stop()
						t.Skipf("Backend %q (%s) listed in skip_backends", resolvedLabel, backend.Name())
					}

					tmModel, err := timestral.LoadModel(repo)
					require.NoError(t, err)
					if opts.ForceFloat32 != nil {
						tmModel.WithForceFloat32(*opts.ForceFloat32)
					}
					if opts.UseMMap != nil {
						tmModel.WithMMap(*opts.UseMMap)
					}

					store := mlmodel.NewStore().WithVariableAsConst(true)
					var inferTime time.Duration
					var notes string

					defer func() {
						store.Finalize()
						backend.Finalize()
						mem := memTracker.Stop()
						cleanupMemory()
						row.Mem = mem
						row.InferTime = inferTime
						row.Notes = notes
						row.Status = testStatus(t)
						recordReport(row)
						t.Logf("[%s/%s/%s] load=%s infer=%s %s",
							spec.Name, variant.Name, resolvedLabel,
							humanize.Duration(row.LoadTime), humanize.Duration(inferTime), mem)
					}()

					tLoad := time.Now()
					require.NoError(t, tmModel.LoadStore(backend, store))
					row.LoadTime = time.Since(tLoad)

					exec, err := mlmodel.NewExec(backend, store, func(scope *mlmodel.Scope, x *graph.Node) *graph.Node {
						return tmModel.BuildGraph(scope, x, ts.HorizonSteps).
							SymmetricAveraging(ts.SymmetricAveraging).
							ClipNonNegative(ts.ClipNonNegative).
							Done()
					})
					require.NoError(t, err)
					defer exec.Finalize()

					x := tensors.FromFlatDataAndDimensions(input, batch, ts.ContextSteps)
					t0 := time.Now()
					y, err := exec.Call1(x)
					inferTime = time.Since(t0)
					x.FinalizeAll()
					require.NoError(t, err)

					got, err := tensors.CopyFlatData[float32](y)
					y.FinalizeAll()
					require.NoError(t, err)
					require.Equal(t, len(reference), len(got))

					var maxDiff float64
					for i, v := range got {
						if d := math.Abs(float64(v) - float64(reference[i])); d > maxDiff {
							maxDiff = d
						}
					}
					notes = fmt.Sprintf("max_diff=%.2e", maxDiff)
					t.Logf("%s (%s): max abs diff vs PyTorch = %.3e (across %d values)",
						repoID, variant.Name, maxDiff, len(got))
					require.Less(t, maxDiff, maxAbsTol,
						"max abs diff vs PyTorch exceeded %.2e", maxAbsTol)
				})
			}
		})
	}
}
