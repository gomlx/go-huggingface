// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package timestral implements Mistral AI's Timestral family of probabilistic time-series
// foundation models in GoMLX.
//
// Timestral is a causal patch-based transformer trained with Decoupled Horizon Forecasting (DHF)
// and monotone quantile heads. It supports univariate and multivariate forecasting, missing
// values (represented as NaN or via an explicit boolean observation mask), and optional past
// or known-future covariates (both numerical and target-encoded categorical).
//
// # Supported Checkpoints and Repository Modes
//
// Any compatible checkpoint can be loaded via a [*hub.Repo], including official Mistral AI
// checkpoints on Hugging Face Hub ([RepoTimestral5M], [RepoTimestral28M], [RepoTimestral123M],
// [RepoTimestral209M]), custom fine-tuned repositories, local directories ([hub.NewLocal]),
// or embedded filesystems ([hub.NewEmbed]):
//
//	repo := hub.New(timestral.RepoTimestral28M)
//	model, err := timestral.LoadModel(repo)
//	if err != nil {
//		log.Fatal(err)
//	}
//
// # High-Level Forecasting with Forecaster
//
// For end-to-end inference from Go slices or [*tensors.Tensor] inputs, wrap a loaded [Model]
// in a [Forecaster] using [NewForecaster]. [Forecaster] manages the GoMLX [model.Store] and
// caches compiled computation graphs across calls.
//
// Each forecast is configured using the fluent [ForecastCall] builder returned by
// [Forecaster.Forecast] and executed by calling [ForecastCall.Done]:
//
//	backend := compute.MustNew()
//	repo := hub.New(timestral.RepoTimestral28M)
//	tmModel, err := timestral.LoadModel(repo)
//	if err != nil {
//		log.Fatal(err)
//	}
//	fc, err := timestral.NewForecaster(backend, tmModel)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	// Context can be 1D [T], 2D [B, T], or 3D [B, V, T] float32/float64 slices or *tensors.Tensor.
//	// Missing values may be represented directly as float32(math.NaN()).
//	context := [][]float32{
//		{10.2, 10.5, float32(math.NaN()), 11.1, 11.4, 11.8},
//	}
//
//	res, err := fc.Forecast(context, 24).Done()
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer res.Close()
//
//	// res.Quantiles has shape [B, V, PredictionLength, Q].
//	median, err := res.Median() // shape [B, V, PredictionLength]
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer median.FinalizeAll()
//
//	q90, err := res.Quantile(0.9) // shape [B, V, PredictionLength]
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer q90.FinalizeAll()
//
// # Using Exogenous Covariates
//
// When forecasting with past-only or known-future covariates, chain the corresponding
// builder methods on [ForecastCall]:
//   - [ForecastCall.WithPastCovariates]: numerical past covariates of shape [B, C_past, T]
//   - [ForecastCall.WithKnownCovariates]: numerical known covariates of shape [B, C_known, T + PredictionLength]
//   - [ForecastCall.WithCategoricalPastCovariates]: integer/category codes of shape [B, C_cat_past, T]
//   - [ForecastCall.WithCategoricalKnownCovariates]: integer/category codes of shape [B, C_cat_known, T + PredictionLength]
//
// Example with known-future covariates:
//
//	res, err := fc.Forecast(context, 24).
//		WithPastCovariates(pastCov).
//		WithKnownCovariates(knownCov).
//		Done()
//
// # Low-Level GoMLX Graph Construction
//
// For custom GoMLX pipelines or low-memory production serving (where weights are baked as
// constants into a pre-compiled graph and the variable store is freed immediately after
// compilation), use [Model.LoadStore] and [Model.BuildGraph] (or [Model.NewBlockBuilder]
// for single-block decoding):
//
//	backend := compute.MustNew()
//	repo := hub.New(timestral.RepoTimestral28M)
//	tmModel, err := timestral.LoadModel(repo)
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	store := model.NewStore().WithVariableAsConst(true)
//	if err := tmModel.LoadStore(backend, store); err != nil {
//		log.Fatal(err)
//	}
//
//	const (
//		batchSize    = 1
//		contextSteps = 512
//		horizonSteps = 96
//	)
//	exec, err := model.NewExec(backend, store, func(scope *model.Scope, x *graph.Node) *graph.Node {
//		return tmModel.BuildGraph(scope, x, horizonSteps).Done()
//	})
//	if err != nil {
//		log.Fatal(err)
//	}
//	// Pre-compile for a fixed input shape and release the variable store:
//	if _, err := exec.Compile(shapes.Make(dtypes.Float32, batchSize, contextSteps)); err != nil {
//		log.Fatal(err)
//	}
//	store.Finalize()
package timestral
