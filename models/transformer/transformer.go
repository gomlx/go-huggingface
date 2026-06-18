package transformer

import (
	"math"
	"slices"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/attention/pos"
	"github.com/gomlx/gomlx/ml/model"
	mltransformer "github.com/gomlx/gomlx/ml/zoo/transformer"
	"github.com/gomlx/gomlx/support/exceptions"
)

// WithCausalMask sets whether to use a causal mask in the attention layers.
//
// Some models are trained with a causal mask, others without, but it is not documented
// in the usual model configuration.
//
//	The default is to use a causal mask.
func (m *Model) WithCausalMask(useCausalMask bool) *Model {
	m.useCausalMask = useCausalMask
	return m
}

// ForwardGraph takes the input tokens and creates the GoMLX graph for the model.
// It returns the final sentence embeddings if appropriate, otherwise just final hidden states.
//
//   - tokens: shaped [batchSize, seqLen] with the tokens, including padding.
//   - mask: shaped [batchSize, seqLen] with the mask, where 1 means valid token and 0 means padding. It can be nil,
//     in which case no mask is used and it's assumed all elements of the sentence are used.
//
// If the model was trained as an embedding model (e.g. sentence-transformers), it will return the sentence embeddings,
// usually as [batchSize, embedSize].
// Otherwise, it will return the final hidden states of all layers, usually as [batchSize, seqLen, hiddenSize].
func (m *Model) ForwardGraph(scope *model.Scope, tokens, mask *graph.Node) *graph.Node {
	if len(m.Modules) > 0 || m.PoolingConfig != nil {
		return m.SentenceEmbeddingGraph(scope, tokens, mask)
	}

	// Default to just getting the final hidden state of all layers
	lastLayer, _ := m.AllLayers(scope, tokens, mask)
	return lastLayer
}

// CreateGoMLXModel initializes the base GoMLX ml_transformer.Model configuration using the loaded fields.
//
// Usually, you won't call this directly, instead you would use the Model.SentenceEmbedding or Model.AllLayers.
// But you can use it for something custom.
//
// It takes the context ctx with the loaded variables.
func (m *Model) CreateGoMLXModel(scope *model.Scope) *mltransformer.Model {
	headDim := m.Config.HeadDim
	if headDim == 0 && m.Config.NumAttentionHeads > 0 {
		headDim = m.Config.HiddenSize / m.Config.NumAttentionHeads
	}
	ropeTheta := m.Config.RoPETheta
	if ropeTheta == 0 {
		ropeTheta = 10000.0
	}

	isBert := m.Config.ModelType == "bert" || (len(m.Config.Architectures) > 0 && m.Config.Architectures[0] == "BertModel")

	tm := mltransformer.New(
		m.Config.VocabSize,
		m.Config.HiddenSize,
		m.Config.NumHiddenLayers,
		m.Config.NumAttentionHeads,
		headDim,
	).
		WithFFNDim(m.Config.IntermediateSize).
		WithMaxPosEmbed(m.Config.MaxPositionEmbeddings).
		WithTransposedWeights(true).
		WithCausalMask(m.useCausalMask)

	if isBert {
		tm.WithArchitecture(mltransformer.ArchitectureStandard).
			WithNormalization(layers.NormalizationLayerNorm).
			WithBias(true).
			WithCausalMask(false)

		if m.Config.LayerNormEps > 0 {
			tm.WithNormEpsilon(m.Config.LayerNormEps)
		} else if m.Config.RMSNormEps > 0 {
			tm.WithNormEpsilon(m.Config.RMSNormEps)
		}

		activationName := m.Config.HiddenActivation
		if activationName == "" {
			activationName = m.Config.HiddenAct
		}
		tm.WithActivation(activation.FromName(activationName))

		tm.WithPositionalEncoder(pos.NewLearned(scope, m.Config.MaxPositionEmbeddings, m.Config.HiddenSize))
		tm.WithEmbedNormalization(layers.NormalizationLayerNorm)

		if typeVocabSize, ok := m.Config.Extra["type_vocab_size"].(float64); ok && typeVocabSize > 0 {
			tm.WithTokenTypeEmbedding(int(typeVocabSize))
		}

	} else {
		arch := mltransformer.ArchitectureGemma4
		if m.Config.ModelType != "gemma4" && m.Config.ModelType != "gemma4_text" {
			arch = mltransformer.ArchitectureGemma3
		}
		tm.WithArchitecture(arch).
									WithNormalization(layers.NormalizationRMSNorm).
									WithNormEpsilon(m.Config.RMSNormEps).
									WithActivation(activation.FromName(m.Config.HiddenActivation)).
									WithNumKVHeads(m.Config.NumKeyValueHeads).
									WithBias(false).
									WithFinalNormalization(layers.NormalizationRMSNorm)

		attnLogitCap := m.Config.AttentionLogitCap
		if attnLogitCap == 0 && (m.Config.ModelType == "gemma4" || m.Config.ModelType == "gemma4_text") {
			attnLogitCap = 50.0
			tm.WithRMSNormOffset(0.0)
			tm.WithQueryKeyScale(1.0)
		}

		tm.WithSlidingWindow(m.Config.SlidingWindow).
			WithNumKVSharedLayers(m.Config.NumKVSharedLayers).
			WithScoreSoftCap(attnLogitCap).
			WithFinalLogitSoftCap(m.Config.FinalLogitSoftcapping)
		if len(m.Config.LayerTypes) > 0 {
			layerTypes := make([]mltransformer.LayerType, m.Config.NumHiddenLayers)
			for i, lt := range m.Config.LayerTypes {
				if lt == "sliding_attention" {
					layerTypes[i] = mltransformer.LocalLayer
				} else {
					layerTypes[i] = mltransformer.GlobalLayer
				}
			}
			tm.WithLayerTypes(layerTypes)
		}

		if len(m.Config.RoPEParameters) > 0 {
			var fullRope, slidingRope *pos.RoPE
			if fp, ok := m.Config.RoPEParameters["full_attention"]; ok && fp.RopeTheta > 0 {
				fDim := m.Config.GlobalHeadDim
				if fDim == 0 {
					fDim = headDim
				}
				if fp.PartialRotaryFactor > 0 && fp.PartialRotaryFactor < 1.0 {
					rotatedDims := int(fp.PartialRotaryFactor * float64(fDim))
					fullRope = pos.NewRoPEWithDimRange(fp.RopeTheta, 0, rotatedDims).WithFrequencyDivisor(fDim)
				} else {
					fullRope = pos.NewRoPE(fp.RopeTheta)
				}
			}
			if sp, ok := m.Config.RoPEParameters["sliding_attention"]; ok && sp.RopeTheta > 0 {
				if sp.PartialRotaryFactor > 0 && sp.PartialRotaryFactor < 1.0 {
					rotatedDims := int(sp.PartialRotaryFactor * float64(headDim))
					slidingRope = pos.NewRoPEWithDimRange(sp.RopeTheta, 0, rotatedDims).WithFrequencyDivisor(headDim)
				} else {
					slidingRope = pos.NewRoPE(sp.RopeTheta)
				}
			}

			if fullRope == nil {
				fullRope = pos.NewRoPE(ropeTheta)
			}
			tm.WithPositionalEncoder(fullRope)

			for i, lt := range m.Config.LayerTypes {
				if lt == "sliding_attention" && slidingRope != nil {
					tm.WithLayerPositionalEncoder(i, slidingRope)
				} else if lt == "full_attention" && fullRope != nil {
					tm.WithLayerPositionalEncoder(i, fullRope)
				}
			}
		} else {
			defaultRope := pos.NewRoPE(ropeTheta)
			if m.Config.RoPEScaling.Type == "linear" {
				defaultRope.WithLinearScaling(m.Config.RoPEScaling.Factor)
			}
			tm.WithPositionalEncoder(defaultRope)

			if m.Config.RoPELocalBaseFreq > 0 {
				slidingRope := pos.NewRoPE(m.Config.RoPELocalBaseFreq)
				for i, lt := range m.Config.LayerTypes {
					if lt == "sliding_attention" {
						tm.WithLayerPositionalEncoder(i, slidingRope)
					}
				}
			}
		}

		if m.Config.HiddenSizePerLayerInput > 0 {
			tm.WithVocabSizePerLayerInput(m.Config.VocabSizePerLayerInput).
				WithHiddenSizePerLayerInput(m.Config.HiddenSizePerLayerInput).
				WithPerLayerInputScale(1.0 / math.Sqrt(2.0)).
				WithPerLayerModelProjectionScale(1.0 / math.Sqrt(float64(m.Config.HiddenSize)))
		}
	}

	torchDtype := m.Config.TorchDtype
	if torchDtype == "" {
		torchDtype = m.Config.DType
	}
	switch torchDtype {
	case "bfloat16":
		tm.WithDType(dtypes.BFloat16)
	case "float16":
		tm.WithDType(dtypes.Float16)
	default:
		tm.WithDType(dtypes.Float32)
	}

	if m.KVCache != nil {
		tm.KVCache = m.KVCache
	}

	if m.Config.GlobalHeadDim > 0 {
		tm.WithGlobalHeadDim(m.Config.GlobalHeadDim)
		tm.KVCache.WithGlobalHeadDim(m.Config.GlobalHeadDim)
	}

	return tm
}

// AllLayers takes the input tokens and creates the GoMLX forward graph for the transformer model,
// returning the last layer and all the intermediate layers.
//
// Inputs:
//
//   - tokens: shaped [batchSize, seqLen] with the tokens, including padding.
//   - mask: shaped [batchSize, seqLen] with the mask, where 1 means valid token and 0 means padding. It can be nil,
//     in which case not mask is set.
//
// It returns:
//
//   - lastLayer: the final hidden state of the last layer, shaped [batchSize, seqLen,hiddenSize].
//   - allLayers: the input to the first layer and the output of each layer.
//     It follows the HuggingFace convention, where the allLayers[0] is the input to the first attention layer,
//     and the following nodes in allLayers are the outputs of all NumHiddenLayers attention layers.
func (m *Model) AllLayers(scope *model.Scope, tokens, mask *graph.Node) (lastLayer *graph.Node, allLayers []*graph.Node) {
	// Sanity checking.
	if tokens.Rank() == 1 {
		// Add batch dimension if not present.
		tokens = graph.ExpandAxes(tokens, 0)
	} else if tokens.Rank() != 2 {
		exceptions.Panicf("tokens must be shaped [batchSize, seqLen] or [seqLen], got %v", tokens.Shape())
	}
	if mask != nil {
		if mask.Rank() == 1 {
			// Add batch dimension if not present.
			mask = graph.ExpandAxes(mask, 0)
		} else if mask.Rank() != 2 {
			exceptions.Panicf("mask must be shaped [batchSize, seqLen] or [seqLen], got %v", mask.Shape())
		}
		if !slices.Equal(tokens.Shape().Dimensions, mask.Shape().Dimensions) {
			exceptions.Panicf("if mask is set, its shape must match the tokens: got tokens.shape=%s, mask.shape=%s", tokens.Shape(), mask.Shape())
		}
	}

	// Create GoMLXModel and build the graph for all the layers.
	tm := m.CreateGoMLXModel(scope)
	lastLayer, allLayers, _ = tm.AllLayers(scope, tokens, nil, mask, nil)
	return lastLayer, allLayers
}

// Forward performs the forward pass of the model.
// It returns the logits of the vocabulary projection (typically shaped [batchSize, seqLen, vocabSize])
// and the updated KV Cache.
func (m *Model) Forward(
	scope *model.Scope,
	tokenIds *graph.Node,
	positionIds *graph.Node,
	attentionMask *graph.Node,
	cache mltransformer.KVCacheNodes,
) (logits *graph.Node, updatedCache mltransformer.KVCacheNodes) {
	tm := m.CreateGoMLXModel(scope)
	logits, updatedCache = tm.Forward(scope, tokenIds, positionIds, attentionMask, cache)
	return logits, updatedCache
}

