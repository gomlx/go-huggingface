// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sam2

import (
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"testing"
	"time"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/hub"
	modelimage "github.com/gomlx/go-huggingface/models/image"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/core/tensors/images"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"
)

type ExpectedData struct {
	PixelValuesShape        []int         `json:"pixel_values_shape"`
	PixelValuesMean         float64       `json:"pixel_values_mean"`
	PixelValuesStd          float64       `json:"pixel_values_std"`
	ImageEmbeddingsShapes   [][]int       `json:"image_embeddings_shapes"`
	ImageEmbeddingsLastMean float64       `json:"image_embeddings_last_mean"`
	SparseEmbeddingsShape   []int         `json:"sparse_embeddings_shape"`
	DenseEmbeddingsShape    []int         `json:"dense_embeddings_shape"`
	DenseEmbeddingsMean     float64       `json:"dense_embeddings_mean"`
	IoUScores               [][][]float32 `json:"iou_scores"`
	PredMasksShape          []int         `json:"pred_masks_shape"`
	PredMasksMean           float32       `json:"pred_masks_mean"`
}

func TestSAM2Inference(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping SAM2 inference test in short mode")
	}

	// 1. Initialize backend
	backend := testutil.BuildTestBackend()

	// 2. Load expected ground truth from Python
	expectedPath := "./sam2_test_data.json"
	b, err := os.ReadFile(expectedPath)
	require.NoError(t, err)
	var expected ExpectedData
	require.NoError(t, json.Unmarshal(b, &expected))

	// 3. Load SAM2 model configuration and weights
	repo := hub.New("facebook/sam2-hiera-base-plus")
	require.NoError(t, repo.DownloadInfo(false))

	modelObj, err := LoadModel(repo)
	require.NoError(t, err)

	store := model.NewStore()
	require.NoError(t, modelObj.LoadStore(backend, store))
	fmt.Printf("- Number of variables: %d\n", store.NumVariables())

	// 4. Load the test image
	imagePath := "../image/test_image_1024.png"
	imgFile, err := os.Open(imagePath)
	require.NoError(t, err)
	defer imgFile.Close()

	img, _, err := image.Decode(imgFile)
	require.NoError(t, err)

	imgTensor := images.ToTensor(dtypes.Float32).Single(img)

	// Create inputs: points and labels
	// coordinates: (1024, 1024) in original 2048x2048 image -> scaled by 0.5 to (512, 512)
	pointsTensor := tensors.FromValue([][][][]float32{{{{512, 512}}}})
	labelsTensor := tensors.FromValue([][][]int32{{{1}}})

	// 5. Build and compile the computation graph
	exec, err := model.NewExec(backend, store, func(scope *model.Scope, rawImage, inputPoints, inputLabels *Node) []*Node {
		g := rawImage.Graph()

		// ImageNet normalization mean and std
		mean := []float64{0.485, 0.456, 0.406}
		std := []float64{0.229, 0.224, 0.225}

		// Preprocess: Bilinear resize to 1024x1024, channels first, rescale, normalize
		preprocessed := modelimage.PreprocessGraph(rawImage, 1024, 1024, mean, std)

		predMasks, states := Forward(scope, preprocessed, inputPoints, inputLabels, nil, nil, modelObj.Config, true)

		// Compute means/reduce of intermediate nodes to print
		preprocessedMean := ReduceMean(preprocessed)
		diff := Sub(preprocessed, preprocessedMean)
		preprocessedStd := Sqrt(ReduceMean(Mul(diff, diff)))
		imageEmbedsLastMean := ReduceMean(states.FPNHiddenStates[2])
		denseEmbedsMean := ReduceMean(states.DenseEmbeddings)
		sparseEmbedsMean := ReduceMean(states.SparseEmbeddings)
		iouTokenOutMean := ReduceMean(states.IoUTokenOut)
		maskTokensOutMean := ReduceMean(states.MaskTokensOut)
		upscaledMean := ReduceMean(states.Upscaled)
		upscaled2Mean := ReduceMean(states.Upscaled2)
		hyperInMean := ReduceMean(states.HyperIn)
		qL0AfterLN1Mean := ReduceMean(states.FPNHiddenStates[0])
		qL0AfterLN2Mean := ReduceMean(states.FPNHiddenStates[1])
		qL0AfterLN3Mean := ScalarZero(g, dtypes.Float32)
		kL0AfterLN4Mean := ScalarZero(g, dtypes.Float32)
		debugLabels := ScalarZero(g, dtypes.Int32)
		debugPointEmbedValues := ScalarZero(g, dtypes.Float32)

		return []*Node{predMasks, states.IoUScores, preprocessedMean, preprocessedStd, imageEmbedsLastMean, denseEmbedsMean, states.SparseEmbeddings, iouTokenOutMean, maskTokensOutMean, upscaledMean, upscaled2Mean, hyperInMean, qL0AfterLN1Mean, qL0AfterLN2Mean, qL0AfterLN3Mean, kL0AfterLN4Mean, sparseEmbedsMean, debugLabels, debugPointEmbedValues}
	})
	require.NoError(t, err)

	// 6. Run the graph
	fmt.Printf("- Compiling and executing graph ...")
	start := time.Now()
	results, err := exec.Exec(imgTensor, pointsTensor, labelsTensor)
	fmt.Printf(" done (elapsed: %s)\n", humanize.Duration(time.Since(start)))
	require.NoError(t, err)

	predMasksTensor := results[0]
	iouScoresTensor := results[1]
	preprocessedMeanTensor := results[2]
	preprocessedStdTensor := results[3]
	imageEmbedsLastMeanTensor := results[4]
	denseEmbedsMeanTensor := results[5]
	sparseEmbedsTensor := results[6]
	iouTokenOutMeanTensor := results[7]
	maskTokensOutMeanTensor := results[8]
	upscaledMeanTensor := results[9]
	upscaled2MeanTensor := results[10]
	hyperInMeanTensor := results[11]
	qL0AfterLN1Tensor := results[12]
	qL0AfterLN2Tensor := results[13]
	qL0AfterLN3Tensor := results[14]
	kL0AfterLN4Tensor := results[15]
	sparseEmbedsMeanTensor := results[16]
	_ = results[17]
	_ = results[18]

	fmt.Printf("- Model output compared to original (Python):\n")
	fmt.Printf("  - Preprocessed image mean: %v (expected: %v)\n", preprocessedMeanTensor.Value(), expected.PixelValuesMean)
	fmt.Printf("  - Preprocessed image std:  %v (expected: %v)\n", preprocessedStdTensor.Value(), expected.PixelValuesStd)
	fmt.Printf("  - Backbone Stage 2 mean:   %v (expected: 0.018796)\n", imageEmbedsLastMeanTensor.Value())
	fmt.Printf("  - Dense embeds mean:       %v (expected: %v)\n", denseEmbedsMeanTensor.Value(), expected.DenseEmbeddingsMean)
	fmt.Printf("  - Sparse embeds mean:      %v (expected: 0.0196135)\n", sparseEmbedsMeanTensor.Value())
	fmt.Printf("  - iouTokenOut mean:        %v (expected: 0.002727542)\n", iouTokenOutMeanTensor.Value())
	fmt.Printf("  - maskTokensOut mean:      %v (expected: -0.02376326)\n", maskTokensOutMeanTensor.Value())
	fmt.Printf("  - upscaled mean:           %v (expected: 0.05955713)\n", upscaledMeanTensor.Value())
	fmt.Printf("  - upscaled2 mean:          %v (expected: 0.010523845)\n", upscaled2MeanTensor.Value())
	fmt.Printf("  - hyperIn mean:            %v (expected: -0.6389069)\n", hyperInMeanTensor.Value())
	fmt.Printf("  - Backbone Stage 0 mean:   %v (expected: -0.046504)\n", qL0AfterLN1Tensor.Value())
	fmt.Printf("  - Backbone Stage 1 mean:   %v (expected: 0.056898)\n", qL0AfterLN2Tensor.Value())
	fmt.Printf("  - Backbone Stage 3 mean:   %v (expected: -0.001673)\n", qL0AfterLN3Tensor.Value())
	fmt.Printf("  - L1 keys after LN4:       %v (expected: 0.044682167)\n", kL0AfterLN4Tensor.Value())

	// Print sparseEmbeds shape and sample
	fmt.Printf("  - Sparse embeds shape: %v\n", sparseEmbedsTensor.Shape())
	sparseEmbedsTensor.ConstFlatData(func(flatAny any) {
		flat := flatAny.([]float32)
		var sum0, sum1 float32
		for i := range 256 {
			sum0 += flat[i]
			sum1 += flat[256+i]
		}
		fmt.Printf("    - Token 0 mean: %v\n", sum0/256.0)
		fmt.Printf("    - Token 0 first 10: %v\n", flat[0:10])
		fmt.Printf("    - Token 1 mean: %v\n", sum1/256.0)
		fmt.Printf("    - Token 1 first 10: %v\n", flat[256:266])
	})

	// 7. Verify the predictions
	fmt.Printf("  - Predicted masks shape: %v\n", predMasksTensor.Shape())
	fmt.Printf("  - Predicted IoU scores: %v\n", iouScoresTensor.Value())

	gotIoUScores := iouScoresTensor.Value().([][][]float32)
	require.Len(t, gotIoUScores, 1)
	require.Len(t, gotIoUScores[0], 1)
	require.Len(t, gotIoUScores[0][0], 3)

	for i := range gotIoUScores[0][0] {
		require.InDelta(t, expected.IoUScores[0][0][i], gotIoUScores[0][0][i], 6e-2,
			"IoU score at index %d doesn't match Python reference", i)
	}

	// Calculate and verify mean predicted mask logit
	// Convert flat data and calculate mean
	predMasksTensor.ConstFlatData(func(flatAny any) {
		flat := flatAny.([]float32)
		var sum float64
		for _, val := range flat {
			sum += float64(val)
		}
		meanVal := float32(sum / float64(len(flat)))
		fmt.Printf("  - Predicted masks mean logit: %f (expected: %f)\n", meanVal, expected.PredMasksMean)
		require.InDelta(t, expected.PredMasksMean, meanVal, 1.0,
			"Mean predicted mask logit doesn't match Python reference closely enough")
	})

	// Measure execution time, without recompilation:
	fmt.Printf("- Re-executing graph ...")
	start = time.Now()
	_, err = exec.Exec(imgTensor, pointsTensor, labelsTensor)
	fmt.Printf(" done (elapsed: %s)\n", humanize.Duration(time.Since(start)))
	require.NoError(t, err)
}
