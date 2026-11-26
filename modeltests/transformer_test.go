// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package modeltests

import (
	"fmt"
	"math"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/go-huggingface/tokenizers"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/attention/kvcache"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/zoo/transformer/generate"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func runTransformerModel(
	t *testing.T,
	loaded loadedSpec,
	sharedSentences *SharedSentences,
	sharedPrompts *SharedPrompts,
	backendConfigs []string,
) {
	spec := loaded.Spec

	for _, variant := range spec.Variants {
		t.Run(variant.Name, func(t *testing.T) {
			repoID := spec.RepoID
			if variant.RepoID != "" {
				repoID = variant.RepoID
			}
			revision := spec.Revision
			if variant.Revision != "" {
				revision = variant.Revision
			}
			opts := mergeModelOptions(spec.ModelOptions, variant.ModelOptions)

			repo, ok := openRepoForSpec(t, repoID, revision, spec.Gated, variant.GGUFFile)
			if !ok {
				return
			}

			hfModel, err := transformer.LoadModel(repo)
			require.NoError(t, err, "transformer.LoadModel(%s)", repoID)
			if variant.GGUFFile != "" {
				hfModel.WithGGUFFile(variant.GGUFFile)
			}
			if opts.CausalMask != nil {
				hfModel.WithCausalMask(*opts.CausalMask)
			}
			if opts.ApproximateGelu != nil {
				hfModel.WithApproximateGelu(*opts.ApproximateGelu)
			}
			if opts.ForceFloat32 != nil {
				hfModel.WithForceFloat32(*opts.ForceFloat32)
			}
			if opts.UseMMap != nil {
				hfModel.WithMMap(*opts.UseMMap)
			}

			tokenizer, err := hfModel.GetTokenizer()
			require.NoError(t, err, "GetTokenizer()")

			// 1. Tokenizer parity (backend-independent, run once per variant).
			tokSpec := variant.Tokenizer
			if tokSpec == nil {
				tokSpec = spec.Tokenizer
			}
			if tokSpec != nil {
				t.Run("tokenize", func(t *testing.T) {
					checkTokenizerParity(t, tokenizer, tokSpec)
				})
			}

			// 2. Run model checks across each requested backend configuration.
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

					store := createStoreWithSettings(t, opts)
					var totalInferTime time.Duration
					var notes string

					defer func() {
						store.Finalize()
						backend.Finalize()
						mem := memTracker.Stop()
						cleanupMemory()
						row.Mem = mem
						row.InferTime = totalInferTime
						row.Notes = notes
						row.Status = testStatus(t)
						recordReport(row)
						t.Logf("[%s/%s/%s] load=%s infer=%s %s",
							spec.Name, variant.Name, resolvedLabel,
							humanize.Duration(row.LoadTime), humanize.Duration(totalInferTime), mem)
					}()

					tLoad := time.Now()
					err = hfModel.LoadStore(backend, store)
					require.NoError(t, err, "LoadStore(%s)", resolvedLabel)
					row.LoadTime = time.Since(tLoad)
					t.Logf("Loaded weights for %s (%s) on %s in %s",
						repoID, variant.Name, backend.Description(), humanize.Duration(row.LoadTime))

					loadedPaths := snapshotVariablePaths(store)

					if variant.Embedding != nil {
						inferDur, embNotes := runTransformerEmbeddingChecks(
							t, backend, store, hfModel, tokenizer, loaded.Dir, variant.Embedding, sharedSentences, loadedPaths)
						totalInferTime += inferDur
						notes = embNotes
					}

					if variant.Generate != nil {
						inferDur, genNotes := runTransformerGenerateChecks(
							t, backend, store, hfModel, tokenizer, variant.Generate, sharedPrompts, loadedPaths)
						totalInferTime += inferDur
						if notes != "" && genNotes != "" {
							notes += "; " + genNotes
						} else if genNotes != "" {
							notes = genNotes
						}
					}

					if *flagDebugLayers {
						t.Run("debug_layers", func(t *testing.T) {
							t.Logf("-debug_layers enabled (looking in %s)", filepath.Join(loaded.Dir, "debug_layers"))
						})
					}
				})
			}
		})
	}
}

func checkTokenizerParity(t *testing.T, tok tokenizers.Tokenizer, spec *TokenizerSpec) {
	t.Helper()

	specialMap := map[string]api.SpecialToken{
		"bos":  api.TokBeginningOfSentence,
		"eos":  api.TokEndOfSentence,
		"unk":  api.TokUnknown,
		"pad":  api.TokPad,
		"mask": api.TokMask,
		"cls":  api.TokClassification,
	}
	for name, wantID := range spec.SpecialTokens {
		st, ok := specialMap[name]
		if !ok {
			continue
		}
		gotID, err := tok.SpecialTokenID(st)
		if assert.NoErrorf(t, err, "SpecialTokenID(%s)", name) {
			assert.Equalf(t, wantID, gotID, "SpecialTokenID(%s)", name)
		}
	}

	for i, tc := range spec.Cases {
		gotIDs := tok.Encode(tc.Text)
		assert.Equalf(t, tc.TokenIDs, gotIDs, "case %d Encode(%q)", i, tc.Text)

		if !tc.SkipDecodeCheck && tc.Decoded != "" {
			gotDecoded := tok.Decode(tc.TokenIDs)
			if gotDecoded != tc.Decoded && (tc.DecodedSkipSpecial == "" || gotDecoded != tc.DecodedSkipSpecial) {
				assert.Equalf(t, tc.Decoded, gotDecoded, "case %d Decode(%v)", i, tc.TokenIDs)
			}
		}
	}
}

func snapshotVariablePaths(store *model.Store) map[string]bool {
	paths := make(map[string]bool)
	for v := range store.IterVariables() {
		paths[v.Path()] = true
	}
	return paths
}

func assertNoUnloadedVariables(t *testing.T, store *model.Store, loadedPaths map[string]bool) {
	t.Helper()
	var missing []string
	for v := range store.IterVariables() {
		if !loadedPaths[v.Path()] {
			missing = append(missing, v.Path())
		}
	}
	assert.Empty(t, missing, "expected zero unloaded variables created during graph construction")
}

func runTransformerEmbeddingChecks(
	t *testing.T,
	backend compute.Backend,
	store *model.Store,
	hfModel *transformer.Model,
	tok tokenizers.Tokenizer,
	modelDir string,
	embSpec *EmbeddingSpec,
	sharedSentences *SharedSentences,
	loadedPaths map[string]bool,
) (time.Duration, string) {
	sentences := resolveSentences(embSpec, sharedSentences)
	require.NotEmpty(t, sentences, "embedding spec resolved to zero sentences")

	embPath := filepath.Join(modelDir, embSpec.EmbeddingsFile)
	expectedFlat, err := readF32File(embPath)
	require.NoError(t, err, "reading expected embeddings %q", embPath)

	dim := embSpec.EmbeddingDim
	if dim == 0 {
		dim = hfModel.Config.HiddenSize
	}
	require.Equal(t, len(sentences)*dim, len(expectedFlat),
		"expected %d floats (%d x %d) in %q, got %d",
		len(sentences)*dim, len(sentences), dim, embPath, len(expectedFlat))

	minCos := embSpec.Tolerances.MinCosineSim
	if minCos == 0 {
		minCos = 0.999
	}
	maxAbsTol := embSpec.Tolerances.MaxAbsDiff
	if maxAbsTol == 0 {
		maxAbsTol = 0.05
	}
	maxMeanTol := embSpec.Tolerances.MaxMeanAbsDiff
	if maxMeanTol == 0 {
		maxMeanTol = 0.01
	}

	padID := int32(0)
	if id, err := tok.SpecialTokenID(api.TokPad); err == nil {
		padID = int32(id)
	}

	var totalInfer time.Duration
	worstCos := 1.0
	worstMaxAbs := 0.0

	// 1. Single-sentence unpadded execution + variable hygiene check.
	singleExec, err := model.NewExec(backend, store, func(scope *model.Scope, tokens *graph.Node) *graph.Node {
		x := hfModel.SentenceEmbeddingGraph(scope, tokens, nil)
		return graph.ConvertDType(x, dtypes.Float32)
	})
	require.NoError(t, err)
	defer singleExec.Finalize()

	var singleGot [][]float32
	t.Run("embed_single", func(t *testing.T) {
		for i, sentence := range sentences {
			tokIDs := tok.Encode(sentence)
			tokens := make([]int32, len(tokIDs))
			for j, id := range tokIDs {
				tokens[j] = int32(id)
			}
			inTensor := tensors.FromValue([][]int32{tokens})
			t0 := time.Now()
			results, err := singleExec.Exec(inTensor)
			totalInfer += time.Since(t0)
			inTensor.FinalizeAll()
			require.NoError(t, err, "singleExec.Exec sentence %d", i)

			outTensor := results[0]
			require.Equal(t, []int{1, dim}, outTensor.Shape().Dimensions)
			got := tensors.MustCopyFlatData[float32](outTensor)
			outTensor.FinalizeAll()

			want := expectedFlat[i*dim : (i+1)*dim]
			m := computeEmbeddingMetrics(got, want)
			t.Logf("sentence %d (%d toks): cos=%.7f max_abs=%.3e mean_abs=%.3e",
				i, len(tokens), m.CosineSim, m.MaxAbsDiff, m.MeanAbsDiff)
			if m.CosineSim < worstCos {
				worstCos = m.CosineSim
			}
			if m.MaxAbsDiff > worstMaxAbs {
				worstMaxAbs = m.MaxAbsDiff
			}
			assert.GreaterOrEqualf(t, m.CosineSim, minCos, "sentence %d cosine similarity", i)
			assert.LessOrEqualf(t, m.MaxAbsDiff, maxAbsTol, "sentence %d max abs diff", i)
			assert.LessOrEqualf(t, m.MeanAbsDiff, maxMeanTol, "sentence %d mean abs diff", i)
			singleGot = append(singleGot, got)
		}

		// Check optional similarity matrix.
		if len(embSpec.SimilarityMatrix) > 0 && len(singleGot) == len(embSpec.SimilarityMatrix) {
			simTol := embSpec.Tolerances.SimilarityMaxDelta
			if simTol == 0 {
				simTol = 1e-2
			}
			for i := range singleGot {
				for j := range embSpec.SimilarityMatrix[i] {
					gotSim := cosineSim(singleGot[i], singleGot[j])
					assert.InDeltaf(t, float64(embSpec.SimilarityMatrix[i][j]), gotSim, simTol,
						"similarity[%d][%d]", i, j)
				}
			}
		}
	})

	t.Run("variables", func(t *testing.T) {
		assertNoUnloadedVariables(t, store, loadedPaths)
	})

	// 2. Batched padded execution with seqLen mask.
	if !embSpec.SkipBatchTest && len(sentences) > 1 {
		t.Run("embed_batch", func(t *testing.T) {
			batchExec, err := model.NewExec(backend, store, func(scope *model.Scope, tokens *graph.Node) *graph.Node {
				constPad := graph.Scalar(tokens.Graph(), tokens.DType(), padID)
				seqLen := graph.ReduceSum(graph.ConvertDType(graph.NotEqual(tokens, constPad), dtypes.Int32), 1)
				x := hfModel.SentenceEmbeddingGraph(scope, tokens, seqLen)
				return graph.ConvertDType(x, dtypes.Float32)
			})
			require.NoError(t, err)
			defer batchExec.Finalize()

			var batchTokens [][]int32
			maxLen := 0
			for _, s := range sentences {
				ids := tok.Encode(s)
				row := xslices.Map(ids, func(v int) int32 { return int32(v) })
				batchTokens = append(batchTokens, row)
				if len(row) > maxLen {
					maxLen = len(row)
				}
			}
			// Add extra padding (+4) so even the longest sentence tests mask handling.
			paddedLen := maxLen + 4
			batchFlat := xslices.SliceWithValue(len(sentences)*paddedLen, padID)
			for i, row := range batchTokens {
				copy(batchFlat[i*paddedLen:i*paddedLen+len(row)], row)
			}
			batchTensor := tensors.FromFlatDataAndDimensions(batchFlat, len(sentences), paddedLen)
			t0 := time.Now()
			results, err := batchExec.Exec(batchTensor)
			totalInfer += time.Since(t0)
			batchTensor.FinalizeAll()
			require.NoError(t, err)

			outTensor := results[0]
			require.Equal(t, []int{len(sentences), dim}, outTensor.Shape().Dimensions)
			flat := tensors.MustCopyFlatData[float32](outTensor)
			outTensor.FinalizeAll()

			for i := range sentences {
				got := flat[i*dim : (i+1)*dim]
				want := expectedFlat[i*dim : (i+1)*dim]
				m := computeEmbeddingMetrics(got, want)
				t.Logf("batch[%d] (len=%d/%d): cos=%.7f max_abs=%.3e mean_abs=%.3e",
					i, len(batchTokens[i]), paddedLen, m.CosineSim, m.MaxAbsDiff, m.MeanAbsDiff)
				assert.GreaterOrEqualf(t, m.CosineSim, minCos, "batch[%d] cosine similarity", i)
				assert.LessOrEqualf(t, m.MaxAbsDiff, maxAbsTol, "batch[%d] max abs diff", i)
				assert.LessOrEqualf(t, m.MeanAbsDiff, maxMeanTol, "batch[%d] mean abs diff", i)
			}
		})
	}

	return totalInfer, fmt.Sprintf("min_cos=%.5f max_diff=%.2e", worstCos, worstMaxAbs)
}

func runTransformerGenerateChecks(
	t *testing.T,
	backend compute.Backend,
	store *model.Store,
	hfModel *transformer.Model,
	tok tokenizers.Tokenizer,
	genSpec *GenerateSpec,
	_ *SharedPrompts,
	loadedPaths map[string]bool,
) (time.Duration, string) {
	require.NotEmpty(t, genSpec.Cases, "generate spec has no test cases")

	var totalInfer time.Duration
	totalMatched := 0
	totalExpected := 0

	// 1. Prefill top-K logits check (fast, deterministic single forward pass).
	prefillExec, err := model.NewExec(backend, store, func(scope *model.Scope, tokens *graph.Node) *graph.Node {
		logits, _ := hfModel.Forward(scope, tokens, nil, nil, nil, nil)
		// Slice last sequence position: [1, seqLen, vocab] -> [vocab]
		seqLen := logits.Shape().Dimensions[1]
		lastLogits := graph.Slice(logits, graph.AxisElem(0), graph.AxisElem(seqLen-1), graph.AxisRange())
		return graph.ConvertDType(graph.Squeeze(lastLogits), dtypes.Float32)
	})
	require.NoError(t, err)
	defer prefillExec.Finalize()

	t.Run("prefill_logits", func(t *testing.T) {
		for _, gc := range genSpec.Cases {
			if len(gc.PrefillTopK) == 0 {
				continue
			}
			t.Run(gc.Name, func(t *testing.T) {
				promptIDs := gc.PromptTokenIDs
				if len(promptIDs) == 0 {
					text := gc.FormattedPrompt
					if text == "" {
						text = gc.Prompt
					}
					promptIDs = tok.Encode(text)
				}
				tokens := xslices.Map(promptIDs, func(v int) int32 { return int32(v) })
				inTensor := tensors.FromValue([][]int32{tokens})
				t0 := time.Now()
				results, err := prefillExec.Exec(inTensor)
				totalInfer += time.Since(t0)
				inTensor.FinalizeAll()
				require.NoError(t, err)

				gotLogits := tensors.MustCopyFlatData[float32](results[0])
				results[0].FinalizeAll()

				gotTopK := make([]float32, len(gc.PrefillTopK))
				wantTopK := make([]float32, len(gc.PrefillTopK))
				for i, entry := range gc.PrefillTopK {
					require.Less(t, entry.TokenID, len(gotLogits))
					gotTopK[i] = gotLogits[entry.TokenID]
					wantTopK[i] = entry.Logit
				}
				m := computeEmbeddingMetrics(gotTopK, wantTopK)

				// Find GoMLX argmax token ID.
				gotArgmax := 0
				for i := 1; i < len(gotLogits); i++ {
					if gotLogits[i] > gotLogits[gotArgmax] {
						gotArgmax = i
					}
				}
				wantTop1 := gc.PrefillTopK[0].TokenID
				margin := float64(math.Inf(1))
				if len(gc.PrefillTopK) >= 2 {
					margin = float64(gc.PrefillTopK[0].Logit - gc.PrefillTopK[1].Logit)
				}

				maxAbsTol := gc.Tolerances.LogitMaxAbsDiff
				if maxAbsTol == 0 {
					maxAbsTol = 0.5
				}
				minCos := gc.Tolerances.LogitMinCosineSim
				if minCos == 0 {
					minCos = 0.995
				}
				marginTol := gc.Tolerances.Top1MarginTolerance
				if marginTol == 0 {
					marginTol = 0.25
				}

				t.Logf("prefill %s: top1 got=%d (%.3f) want=%d (%.3f, margin=%.3f), top-%d cos=%.6f max_abs=%.4f",
					gc.Name, gotArgmax, gotLogits[gotArgmax], wantTop1, gc.PrefillTopK[0].Logit,
					margin, len(gc.PrefillTopK), m.CosineSim, m.MaxAbsDiff)

				assert.GreaterOrEqualf(t, m.CosineSim, minCos, "top-K logits cosine sim")
				assert.LessOrEqualf(t, m.MaxAbsDiff, maxAbsTol, "top-K logits max abs diff")
				if margin >= marginTol {
					assert.Equalf(t, wantTop1, gotArgmax, "top-1 argmax token mismatch (margin=%.3f >= %.3f)", margin, marginTol)
				}
			})
		}
	})

	t.Run("variables", func(t *testing.T) {
		assertNoUnloadedVariables(t, store, loadedPaths)
	})

	// Helper to configure stop tokens on a Generator.
	configureStops := func(dec *generate.Generator) map[int]bool {
		stops := make(map[int]bool)
		if eosID, err := tok.SpecialTokenID(api.TokEndOfSentence); err == nil {
			dec.WithEOS(eosID)
			stops[eosID] = true
		}
		if lookup, ok := tok.(interface{ TokenToID(string) (int, bool) }); ok {
			for _, stopStr := range genSpec.StopTokens {
				if id, found := lookup.TokenToID(stopStr); found {
					dec.WithStopTokens(id)
					stops[id] = true
				}
			}
		}
		return stops
	}

	buildKVCacheDecoder := func(maxLen int) (*generate.Generator, map[int]bool) {
		tm := hfModel.CreateGoMLXModel(store.RootScope())
		tm.PopulateKVCacheConfigs()
		kvCfg := hfModel.KVCacheConfig()
		var fn generate.KVCacheModelFn = func(scope *model.Scope, newTokens *graph.Node, position *graph.Node, cache kvcache.KVCacheNodes) (*graph.Node, kvcache.KVCacheNodes) {
			return hfModel.Forward(scope, newTokens, position, nil, nil, cache)
		}
		dec := generate.New(fn).
			WithKVCache(kvCfg, tm.NumKVHeads, tm.HeadDim, tm.DType).
			WithMaxLength(maxLen)
		dec.Temperature = 0
		dec.TopK = 0
		dec.TopP = 0
		return dec, configureStops(dec)
	}

	buildNaiveDecoder := func(maxLen int) (*generate.Generator, map[int]bool) {
		var fn generate.NaiveModelFn = func(scope *model.Scope, tokens *graph.Node, seqLen *graph.Node) *graph.Node {
			logits, _ := hfModel.Forward(scope, tokens, nil, seqLen, nil, nil)
			return logits
		}
		dec := generate.New(fn).WithMaxLength(maxLen)
		dec.Temperature = 0
		dec.TopK = 0
		dec.TopP = 0
		return dec, configureStops(dec)
	}

	// 2. Internal KV-cache vs Naive parity check on first case.
	if !genSpec.SkipKVCacheParity && !genSpec.UseNaiveOnly && len(genSpec.Cases) > 0 {
		t.Run("kvcache_parity", func(t *testing.T) {
			gc := genSpec.Cases[0]
			parityNewToks := gc.Tolerances.KVCacheParityTokens
			if parityNewToks == 0 {
				parityNewToks = 8
			}
			promptIDs := gc.PromptTokenIDs
			if len(promptIDs) == 0 {
				text := gc.FormattedPrompt
				if text == "" {
					text = gc.Prompt
				}
				promptIDs = tok.Encode(text)
			}
			maxLen := len(promptIDs) + parityNewToks

			naiveDec, _ := buildNaiveDecoder(maxLen)
			kvDec, _ := buildKVCacheDecoder(maxLen)

			naiveOut, err := naiveDec.Decode(backend, store.RootScope(), promptIDs)
			require.NoError(t, err, "naive Decode")
			naiveIDs := extractNewTokenIDs(naiveOut, len(promptIDs))
			naiveOut.FinalizeAll()

			kvOut, err := kvDec.Decode(backend, store.RootScope(), promptIDs)
			require.NoError(t, err, "kvcache Decode")
			kvIDs := extractNewTokenIDs(kvOut, len(promptIDs))
			kvOut.FinalizeAll()

			t.Logf("kvcache_parity (%s): naive=%v kv=%v", gc.Name, naiveIDs, kvIDs)
			assert.Equal(t, naiveIDs, kvIDs, "KV-cache greedy rollout diverged from naive rollout")
		})
	}

	// 3. Greedy generation (temperature = 0) against reference.
	t.Run("greedy_generate", func(t *testing.T) {
		for _, gc := range genSpec.Cases {
			if len(gc.GreedyTokenIDs) == 0 {
				continue
			}
			t.Run(gc.Name, func(t *testing.T) {
				promptIDs := gc.PromptTokenIDs
				if len(promptIDs) == 0 {
					text := gc.FormattedPrompt
					if text == "" {
						text = gc.Prompt
					}
					promptIDs = tok.Encode(text)
				}
				maxNew := genSpec.MaxNewTokens
				if maxNew == 0 {
					maxNew = len(gc.GreedyTokenIDs)
				}
				maxLen := len(promptIDs) + maxNew

				var dec *generate.Generator
				var stops map[int]bool
				if genSpec.UseNaiveOnly {
					dec, stops = buildNaiveDecoder(maxLen)
				} else {
					dec, stops = buildKVCacheDecoder(maxLen)
				}

				t0 := time.Now()
				outTensor, err := dec.Decode(backend, store.RootScope(), promptIDs)
				dt := time.Since(t0)
				totalInfer += dt
				require.NoError(t, err)
				gotNew := extractNewTokenIDs(outTensor, len(promptIDs))
				outTensor.FinalizeAll()

				// Trim trailing stop token if reference also trimmed it.
				if len(gotNew) > 0 && stops[gotNew[len(gotNew)-1]] &&
					(len(gc.GreedyTokenIDs) == 0 || gc.GreedyTokenIDs[len(gc.GreedyTokenIDs)-1] != gotNew[len(gotNew)-1]) {
					gotNew = gotNew[:len(gotNew)-1]
				}

				gotText := tok.Decode(gotNew)
				matched := commonPrefixLen(gotNew, gc.GreedyTokenIDs)
				minPrefix := gc.Tolerances.MinExactPrefixTokens
				if minPrefix == 0 {
					minPrefix = len(gc.GreedyTokenIDs)
				}
				totalMatched += matched
				totalExpected += len(gc.GreedyTokenIDs)

				tokSec := float64(len(gotNew)) / max(dt.Seconds(), 1e-9)
				t.Logf("greedy %s: matched %d/%d tokens (%.1f tok/s)\n  got:  %q\n  want: %q",
					gc.Name, matched, len(gc.GreedyTokenIDs), tokSec, gotText, gc.GreedyText)

				assert.GreaterOrEqualf(t, matched, minPrefix,
					"greedy prefix match %d < required %d\n  got IDs:  %v\n  want IDs: %v\n  got text:  %q\n  want text: %q",
					matched, minPrefix, gotNew, gc.GreedyTokenIDs, gotText, gc.GreedyText)
			})
		}
	})

	notes := ""
	if totalExpected > 0 {
		notes = fmt.Sprintf("greedy=%d/%d toks", totalMatched, totalExpected)
	}
	return totalInfer, notes
}

func extractNewTokenIDs(outTensor *tensors.Tensor, promptLen int) []int {
	vals := outTensor.Value().([][]int32)[0]
	if promptLen >= len(vals) {
		return nil
	}
	out := make([]int, len(vals)-promptLen)
	for i, v := range vals[promptLen:] {
		out[i] = int(v)
	}
	return out
}

func commonPrefixLen(a, b []int) int {
	n := min(len(a), len(b))
	for i := range n {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

func cosineSim(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

var _ = slices.Equal[[]int]
