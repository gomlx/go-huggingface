package transformer

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
)

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

// SingleSentenceEmbeddingExec returns a context.Exec that can be used to compute sentence embeddings.
// No padding, not bucketing, the exec takes as input a single sentence [seqLen] and returns the embedding [embedDim].
func (m *Model) SingleSentenceEmbeddingExec(backend backends.Backend, ctx *context.Context) (*context.Exec, error) {
	return context.NewExec(backend, ctx, func(ctx *context.Context, tokens *graph.Node) *graph.Node {
		return graph.ConvertDType(m.SentenceEmbeddingGraph(ctx, tokens, nil), dtypes.Float32)
	})
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
