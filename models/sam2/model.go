// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sam2

import (
	"fmt"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/safetensors"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Model holds the configuration and reference to the HuggingFace repository.
type Model struct {
	Repo   *hub.Repo
	Config *Config

	totalParameters *int64
	totalBytes      *int64
}

// LoadModel loads the config.json and returns a Model.
func LoadModel(repo *hub.Repo) (*Model, error) {
	m := &Model{
		Repo: repo,
	}

	path, err := repo.DownloadFile("config.json")
	if err != nil {
		return nil, fmt.Errorf("failed to download config.json: %w", err)
	}

	c, err := LoadConfig(path)
	if err != nil {
		return nil, err
	}
	m.Config = c

	return m, nil
}

// transposeTensor2D transposes a 2D tensor from shape [A, B] to [B, A] using a temporary graph.
func transposeTensor2D(backend compute.Backend, t *tensors.Tensor) (*tensors.Tensor, error) {
	exec, err := graph.NewExec(backend, func(x *graph.Node) *graph.Node {
		return graph.TransposeAllAxes(x, 1, 0)
	})
	if err != nil {
		return nil, err
	}
	res, err := exec.Call(t)
	if err != nil {
		return nil, err
	}
	return res[0], nil
}

// alignWeightsWithGoMLX post-processes the loaded weights in the store to match the scopes expected by GoMLX
// (dense, conv, layer_normalization) and transposes dense layers' weight matrices to match GoMLX layout.
func alignWeightsWithGoMLX(backend compute.Backend, store *model.Store) error {
	var vars []*model.Variable
	for v := range store.IterVariables() {
		vars = append(vars, v)
	}

	// 1. Build a map of parentPath -> rank of its weight
	layerRanks := make(map[string]int)
	for _, v := range vars {
		fullPath := v.Path()
		parts := strings.Split(fullPath, "/")
		if len(parts) < 2 {
			continue
		}
		varName := parts[len(parts)-1]
		parentPath := strings.Join(parts[:len(parts)-1], "/")
		if varName == "weights" {
			layerRanks[parentPath] = v.Shape().Rank()
		}
	}

	// 2. Perform moving and transposing
	for _, v := range vars {
		currPath := v.Path()
		parts := strings.Split(currPath, "/")
		if len(parts) < 2 {
			continue
		}
		varName := parts[len(parts)-1]
		parentPath := strings.Join(parts[:len(parts)-1], "/")
		secondToLast := parts[len(parts)-2]

		// Check if it's a LayerNorm parameter
		if strings.Contains(secondToLast, "layer_norm") || strings.Contains(secondToLast, "norm") {
			if varName == "gain" || varName == "weights" {
				toPath := parentPath + "/layer_normalization/gain"
				if err := store.MoveVariable(currPath, toPath); err != nil {
					return err
				}
			} else if varName == "offset" || varName == "biases" {
				toPath := parentPath + "/layer_normalization/offset"
				if err := store.MoveVariable(currPath, toPath); err != nil {
					return err
				}
			}
			continue
		}

		// Check if it's an embedding or constant (skip)
		if strings.HasSuffix(secondToLast, "_embed") || strings.HasSuffix(secondToLast, "_token") || strings.HasSuffix(secondToLast, "_tokens") {
			continue
		}
		if varName != "weights" && varName != "biases" {
			continue
		}

		// Look up the rank of this layer (based on original parent path)
		rank, hasRank := layerRanks[parentPath]
		if !hasRank {
			continue
		}

		if rank == 4 {
			// Convolution
			toPath := parentPath + "/conv/" + varName
			if err := store.MoveVariable(currPath, toPath); err != nil {
				return err
			}
		} else if rank == 2 {
			// Dense/Linear
			if varName == "weights" {
				// Transpose dense weight
				transposed, err := transposeTensor2D(backend, v.MustValue())
				if err != nil {
					return err
				}
				v.MustValue().FinalizeAll()
				if err := v.SetValue(transposed); err != nil {
					return err
				}
			}
			toPath := parentPath + "/dense/" + varName
			if err := store.MoveVariable(currPath, toPath); err != nil {
				return err
			}
		}
	}
	return nil
}

// LoadStore loads the safetensors weights into the GoMLX model.Store.
func (m *Model) LoadStore(backend compute.Backend, store *model.Store) error {
	var totalParams int64
	var totalBytes int64

	for tensorAndName, err := range safetensors.IterTensorsFromRepo(backend, m.Repo) {
		if err != nil {
			return errors.WithMessagef(err, "failed loading variables of model %q", m.Repo.ID)
		}

		scopePath, varName, ok := mapTensorName(tensorAndName.Name)
		if !ok {
			klog.V(1).Infof("Skipping unmapped tensor: %s\n", tensorAndName.Name)
			tensorAndName.Tensor.FinalizeAll()
			continue
		}

		tensorToLoad := tensorAndName.Tensor
		shape := tensorToLoad.Shape()
		totalParams += int64(shape.Size())
		totalBytes += int64(shape.ByteSize())

		subScope := store.RootScope()
		for _, subScopeName := range scopePath {
			subScope = subScope.In("%s", subScopeName)
		}

		subScope.VariableWithValue(varName, tensorToLoad)
	}

	// Post-process loaded variables to move them to GoMLX-aligned scopes (dense/conv/layer_normalization) and transpose dense weights.
	if err := alignWeightsWithGoMLX(backend, store); err != nil {
		return fmt.Errorf("failed to post-process loaded weights: %w", err)
	}

	klog.V(1).Infof("Loaded %s parameters (%s bytes)",
		humanize.Count(totalParams), humanize.Bytes(totalBytes))
	m.totalParameters = &totalParams
	m.totalBytes = &totalBytes

	return nil
}

// mapTensorName translates PyTorch parameter names into GoMLX variable scopes.
func mapTensorName(safetensorsName string) (scopePath []string, varName string, ok bool) {
	parts := strings.Split(safetensorsName, ".")
	if len(parts) == 0 {
		return nil, "", false
	}

	lastPart := parts[len(parts)-1]
	secondToLast := ""
	if len(parts) > 1 {
		secondToLast = parts[len(parts)-2]
		// If secondToLast is a number (like block/layer index), look further back
		if len(secondToLast) > 0 && secondToLast[0] >= '0' && secondToLast[0] <= '9' {
			if len(parts) > 2 {
				secondToLast = parts[len(parts)-3]
			}
		}
	}

	// The rest forms the scope path
	scopePath = parts[:len(parts)-1]

	// Determine variable name
	if lastPart == "weight" {
		if strings.Contains(secondToLast, "layer_norm") || strings.Contains(secondToLast, "norm") {
			varName = "gain"
		} else if strings.HasSuffix(secondToLast, "_embed") || strings.HasSuffix(secondToLast, "_token") || strings.HasSuffix(secondToLast, "_tokens") {
			varName = "embeddings"
		} else {
			varName = "weights"
		}
	} else if lastPart == "bias" {
		if strings.Contains(secondToLast, "layer_norm") || strings.Contains(secondToLast, "norm") {
			varName = "offset"
		} else {
			varName = "biases"
		}
	} else {
		// Other parameters (like no_memory_embedding or pos_embed)
		varName = lastPart
	}

	return scopePath, varName, true
}
