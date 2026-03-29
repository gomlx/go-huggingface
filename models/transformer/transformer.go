package transformer

import (
	"fmt"
	"slices"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention/pos"
	mltransformer "github.com/gomlx/gomlx/pkg/ml/model/transformer"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
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
func (m *Model) ForwardGraph(ctx *context.Context, tokens, mask *graph.Node) *graph.Node {
	if len(m.Modules) > 0 || m.PoolingConfig != nil {
		return m.SentenceEmbeddingGraph(ctx, tokens, mask)
	}

	// Default to just getting the final hidden state of all layers
	lastLayer, _ := m.AllLayers(ctx, tokens, mask)
	return lastLayer
}

// CreateGoMLXModel initializes the base ml_transformer.Model configuration using the loaded fields.
func (m *Model) CreateGoMLXModel() *mltransformer.Model {
	headDim := m.Config.HeadDim
	if headDim == 0 && m.Config.NumAttentionHeads > 0 {
		headDim = m.Config.HiddenSize / m.Config.NumAttentionHeads
	}
	ropeTheta := m.Config.RopeTheta
	if ropeTheta == 0 {
		ropeTheta = 10000.0
	}

	tm := mltransformer.New(
		m.Config.VocabSize,
		m.Config.HiddenSize,
		m.Config.NumHiddenLayers,
		m.Config.NumAttentionHeads,
		headDim,
	).
		WithFFNDim(m.Config.IntermediateSize).
		WithMaxPosEmbed(m.Config.MaxPositionEmbeddings).
		WithArchitecture(mltransformer.ArchitectureGemma3). // FIXME: Should use m.Config.Architectures to map Architecture
		WithTransposedWeights(true).
		WithNormalization(layers.NormalizationRMSNorm).
		WithNormEpsilon(m.Config.RMSNormEps).
		WithActivation(activations.FromName(m.Config.HiddenActivation)).
		WithNumKVHeads(m.Config.NumKeyValueHeads).
		WithBias(false).
		WithCausalMask(m.useCausalMask).
		WithPositionalEncoder(pos.NewRoPE(ropeTheta)).
		WithFinalNormalization(layers.NormalizationRMSNorm)

	switch m.Config.TorchDtype {
	case "bfloat16":
		tm.WithDType(dtypes.BFloat16)
	case "float16":
		tm.WithDType(dtypes.Float16)
	default:
		tm.WithDType(dtypes.Float32)
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
func (m *Model) AllLayers(ctx *context.Context, tokens, mask *graph.Node) (lastLayer *graph.Node, allLayers []*graph.Node) {
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
	tm := m.CreateGoMLXModel()
	return tm.AllLayers(ctx, tokens, mask, false, 0)
}

// SentenceEmbeddingGraph builds the equivalent of the sentence_transformers pipeline.
// It uses AllLayers for the base model, and then applies pooling and normalization
// layers sequentially according to the modules.json configuration.
//
//   - tokens: shaped [batchSize, seqLen] with the tokens, including padding.
//   - mask: shaped [batchSize, seqLen] with the mask, where 1 means valid token and 0 means padding. It can be nil,
//     in which case no mask is used and it's assumed all elements of the sentence are used.
//
// It returns the final pooled embedding (usually [batchSize, embedSize]) for the sentence.
func (m *Model) SentenceEmbeddingGraph(ctx *context.Context, tokens, mask *graph.Node) *graph.Node {
	var x *graph.Node

	for _, mod := range m.Modules {
		switch mod.Type {
		case "sentence_transformers.models.Transformer":
			// The base transformer output is the list of layer outputs.
			// The last item is the final hidden state.
			lastLayer, _ := m.AllLayers(ctx, tokens, mask)
			x = lastLayer

		case "sentence_transformers.models.Pooling":
			if x == nil {
				exceptions.Panicf("pooling module found before transformer module")
			}
			x = m.ApplySentencePooling(ctx, x, tokens, mask)

		case "sentence_transformers.models.Normalize":
			if x == nil {
				exceptions.Panicf("normalize module found before transformer module")
			}
			// Apply L2 normalization: x = x / max(norm(x), eps)
			// Compute norm along the final dimension (hidden_size)
			norm := graph.Sqrt(graph.ReduceSum(graph.Square(x), -1))
			eps := graph.Scalar(x.Graph(), norm.DType(), 1e-12)
			norm = graph.Max(norm, eps)

			// Expand norm back to [batch, 1] for broadcasting against [batch, hidden_size]
			norm = graph.ExpandAxes(norm, -1)
			x = graph.Div(x, norm)

		default:
			fmt.Printf("Warning: unknown module type %q in sentence transformer pipeline. Ignoring.\n", mod.Type)
		}
	}

	if x == nil {
		// Fallback if modules.json is not present or didn't contain sentence_transformers layers
		lastLayer, _ := m.AllLayers(ctx, tokens, mask)
		x = lastLayer
		// and apply default pooling if a pooling config exists
		if m.PoolingConfig != nil {
			x = m.ApplySentencePooling(ctx, x, tokens, mask)
		}
	}

	return x
}

// ApplySentencePooling applies the configured pooling function to the hidden states.
func (m *Model) ApplySentencePooling(ctx *context.Context, hiddenStates, tokens, mask *graph.Node) *graph.Node {
	if m.PoolingConfig == nil {
		exceptions.Panicf("no pooling config was loaded for this model")
	}
	cfg := m.PoolingConfig

	if cfg.PoolingModeLastToken {
		// In Hugging Face, sentence transformers typically use the attention mask to find the last valid token.
		// Since we don't handle padding explicitly through an attention mask yet, we take the physical
		// last token in the sequence. For left-padded or unpadded sequences, this is correct.
		seqLen := hiddenStates.Shape().Dimensions[1]
		if seqLen == -1 {
			// If sequence length is unknown at translation time, we could use graph.Shape(hiddenStates)
			// and graph.Slice dynamically. For simplicity assuming known seq length for now.
			exceptions.Panicf("PoolingModeLastToken requires sequence length to be known at trace time")
		}
		sliced := graph.Slice(hiddenStates, graph.AxisRange(), graph.AxisRange(seqLen-1, seqLen))
		return graph.Squeeze(sliced, 1) // [batch, 1, hidden] -> [batch, hidden]
	}

	if cfg.PoolingModeMeanTokens {
		// Plain mean pooling over the sequence tokens (assuming no padding for now).
		return graph.ReduceMean(hiddenStates, 1)
	}

	exceptions.Panicf("no supported pooling mode enabled in PoolingConfig: %+v", cfg)
	return nil
}
