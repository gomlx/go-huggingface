package transformer

import (
	"fmt"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/exceptions"
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
func (m *Model) SentenceEmbeddingGraph(ctx *model.Context, tokens, mask *graph.Node) *graph.Node {
	var x *graph.Node
	// Add a batch axis if not present.
	if tokens.Rank() == 1 {
		tokens = graph.ExpandAxes(tokens, 0) // [seqLen] -> [1, seqLen]
		if mask != nil {
			mask = graph.ExpandAxes(mask, 0) // [seqLen] -> [1, seqLen]
		}
	}
	if tokens.Rank() != 2 || (mask != nil && mask.Rank() != 2) {
		exceptions.Panicf("tokens must be [batchSize, seqLen] and mask must be [batchSize, seqLen] or nil, got %v and %v",
			tokens.Shape(), mask.Shape())
	}

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
			x = m.ApplySentencePooling(x, mask)

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
			x = m.ApplySentencePooling(x, mask)
		}
	}

	return x
}

// SingleSentenceEmbeddingExec returns a context.Exec that can be used to compute sentence embeddings.
// No padding, not bucketing, the exec takes as input a single sentence [seqLen] and returns the embedding [embedDim].
func (m *Model) SingleSentenceEmbeddingExec(backend compute.Backend, ctx *model.Context) (*model.Exec, error) {
	return model.NewExec(backend, ctx, func(ctx *model.Context, tokens *graph.Node) *graph.Node {
		output := graph.ConvertDType(m.SentenceEmbeddingGraph(ctx, tokens, nil), dtypes.Float32)
		if output.Rank() == 2 && output.Shape().Dimensions[0] == 1 {
			// Remove the batch dimension, since we are expecting a single sentence.
			return graph.Squeeze(output, 0)
		}
		return output
	})
}

// ApplySentencePooling applies the configured pooling function to the hidden states.
//
// - hiddenStates: [batchSize, seqLen, hiddenSize]
// - mask: nil or [batchSize, seqLen] of dtype Bool.
//
// Returns [batchSize, hiddenSize]
func (m *Model) ApplySentencePooling(hiddenStates, mask *graph.Node) *graph.Node {
	if m.PoolingConfig == nil {
		exceptions.Panicf("no pooling config was loaded for this model")
	}
	cfg := m.PoolingConfig
	g := hiddenStates.Graph()
	batchSize := hiddenStates.Shape().Dimensions[0]
	seqLen := hiddenStates.Shape().Dimensions[1]
	// hiddenDim := hiddenStates.Shape().Dimensions[2]

	switch {
	case cfg.PoolingModeLastToken:
		// In Hugging Face, sentence transformers typically use the "attention mask" (the 1D mask, here called simply
		// mask) to find the last valid token.

		var lastTokenIdx *graph.Node
		if mask == nil {
			// If no mask is provided, assume all tokens are valid and take the last one.
			lastTokenIdx = graph.Scalar(g, dtypes.Int32, seqLen-1)
			lastTokenIdx = graph.ExpandAxes(lastTokenIdx, 0)              // scalar -> [1]
			lastTokenIdx = graph.BroadcastPrefix(lastTokenIdx, batchSize) // -> [batchSize, 1]
		} else {
			// Find the last token index by finding the last non-zero element in the mask.
			// mask is [batchSize, seqLen]
			sequenceIndices := graph.Iota(g, shapes.Make(dtypes.Int32, batchSize, seqLen), 1)
			validIndices := graph.Where(mask, sequenceIndices, graph.Scalar(g, dtypes.Int32, -1))
			lastTokenIdx = graph.ReduceAndKeep(validIndices, graph.ReduceMax, 1) // [batchSize, 1]
		}
		// Gather the last token embeddings of each example.
		// Add the batch index to each lastTokenIdx:
		batchIndices := graph.Iota(g, lastTokenIdx.Shape(), 0)
		lastTokenIdx = graph.Concatenate([]*graph.Node{batchIndices, lastTokenIdx}, -1)                        // [batchSize, 2]
		lastTokenEmbeddings := graph.GatherSlices(hiddenStates, []int{0, 1}, lastTokenIdx, []int{1, 1}, false) // [batchSize, 1, 1, hiddenDim]
		lastTokenEmbeddings = graph.Squeeze(lastTokenEmbeddings, 1, 2)                                         // [batchSize, hiddenDim]
		return lastTokenEmbeddings

	case cfg.PoolingModeMeanTokens:
		// Plain mean pooling over the sequence tokens (assuming no padding for now).
		return graph.MaskedReduceMean(hiddenStates, mask, 1) // [batch, hidden]

	case cfg.PoolingModeClsToken:
		// CLS token is typically the first token (index 0).
		// hiddenStates: [batchSize, seqLen, hiddenSize]
		// We take the slice [batchSize, 0, hiddenSize]
		clsTokenEmbeddings := graph.Slice(hiddenStates, graph.AxisRange(), graph.AxisElem(0)) // [batchSize, 1, hiddenSize]
		clsTokenEmbeddings = graph.Squeeze(clsTokenEmbeddings, 1)                             // [batchSize, hiddenSize]
		return clsTokenEmbeddings
	}

	exceptions.Panicf("unsupported pooling mode in PoolingConfig, please add an issue in github.com/gomlx/go-huggingface to add support for it: %+v", cfg)
	return nil
}
