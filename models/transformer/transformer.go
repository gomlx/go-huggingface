package transformer

import (
	"fmt"
	"math"
	"strings"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	ml_transformer "github.com/gomlx/gomlx/pkg/ml/model/transformer"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
)

// WithCausalMask sets whether to use a causal mask in the attention layers.
//
// The KaLM paper says that the model is trained without a causal mask, but HuggingFace transformer
// leaves that on by default. We default to off, but we make it configurable.
// When porting to generic transformer package, causal mask handling depends on the model.
func (m *Model) WithCausalMask(useCausalMask bool) *Model {
	m.useCausalMask = useCausalMask
	return m
}

// BuildAllLayersGraph takes the input tokens and creates the GoMLX graph for the embedding layer.
// It returns an array of []*graph.Node containing [embeddings, layer0, (layer1, etc...)].
func (m *Model) BuildAllLayersGraph(ctx *context.Context, tokens *graph.Node) (outputs []*graph.Node) {
	g := tokens.Graph()

	// The embedding weight name mapped from the HuggingFace safetensors
	varName := "embeddings"

	// Check if the variable exists in the context
	weightVar := ctx.In("token_embed").InspectVariableInScope(varName)
	if weightVar == nil {
		exceptions.Panicf("failed to find embedding weights in context: token_embed/%s. did you call LoadContext?", varName)
	}

	weightNode := weightVar.ValueGraph(g)

	// Gather embeddings for the given tokens.
	// tokens shape: [batch, seq_len] -> expanded to [batch, seq_len, 1] for Gather
	// weightNode shape: [vocab_size, hidden_size]
	// Gather output shape: [batch, seq_len, hidden_size]
	embeddings := graph.Gather(weightNode, graph.ExpandAxes(tokens, -1))

	// Gemma3 scales the embeddings by sqrt(hidden_size).
	// We might need to make this configurable based on architecture (e.g. m.Config.Architectures).
	// For now we keep this scaling if the architecture implies it.
	scaleEmbeddings := false
	for _, arch := range m.Config.Architectures {
		if strings.Contains(arch, "Gemma3") || strings.Contains(arch, "Gemma2") || strings.Contains(arch, "GemmaForCausal") {
			scaleEmbeddings = true
			break
		}
	}

	if scaleEmbeddings {
		scale := math.Sqrt(float64(m.Config.HiddenSize))
		embeddings = graph.MulScalar(embeddings, scale)
	}

	// Output index 0 is token embeddings
	outputs = append(outputs, embeddings)

	// Initialize the base ml_transformer.Model configuration using the loaded fields.
	headDim := m.Config.HeadDim
	if headDim == 0 && m.Config.NumAttentionHeads > 0 {
		headDim = m.Config.HiddenSize / m.Config.NumAttentionHeads
	}
	tm := ml_transformer.New(
		m.Config.VocabSize,
		m.Config.HiddenSize,
		m.Config.NumHiddenLayers,
		m.Config.NumAttentionHeads,
		headDim,
	).
		WithFFNDim(m.Config.IntermediateSize).
		WithMaxPosEmbed(m.Config.MaxPositionEmbeddings).
		WithArchitecture(ml_transformer.ArchitectureGemma3). // FIXME: Should use m.Config.Architectures to map Architecture
		WithTransposedWeights(true).
		WithNormalization(layers.NormalizationRMSNorm).
		WithNormEpsilon(m.Config.RMSNormEps).
		WithActivation(activations.FromName(m.Config.HiddenActivation)).
		WithNumKVHeads(m.Config.NumKeyValueHeads).
		WithBias(false).
		WithCausalMask(m.useCausalMask)

	x := embeddings
	for layerIdx := 0; layerIdx < m.Config.NumHiddenLayers; layerIdx++ {
		layerCtx := ctx.In(fmt.Sprintf("layer_%d", layerIdx))
		x = tm.ForwardLayer(layerCtx, x, layerIdx, false, 0)

		// For the very last layer, huggingface returns the normalized hidden state in hidden_states[-1]
		if layerIdx == m.Config.NumHiddenLayers-1 {
			x = layers.RMSNorm(ctx.In("final_norm"), x).WithEpsilon(m.Config.RMSNormEps).WithScaleOffset(1.0).Done()
		}

		outputs = append(outputs, x)
	}

	return outputs
}

// SentenceEmbeddingGraph builds the equivalent of the sentence_transformers pipeline.
// It uses BuildAllLayersGraph for the base model, and then applies pooling and normalization
// layers sequentially according to the modules.json configuration.
func (m *Model) SentenceEmbeddingGraph(ctx *context.Context, tokens *graph.Node) *graph.Node {
	var x *graph.Node

	for _, mod := range m.Modules {
		switch mod.Type {
		case "sentence_transformers.models.Transformer":
			// The base transformer output is the list of layer outputs.
			// The last item is the final hidden state.
			outputs := m.BuildAllLayersGraph(ctx, tokens)
			x = outputs[len(outputs)-1]

		case "sentence_transformers.models.Pooling":
			if x == nil {
				exceptions.Panicf("pooling module found before transformer module")
			}
			x = m.ApplySentencePooling(ctx, x, tokens)

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
		outputs := m.BuildAllLayersGraph(ctx, tokens)
		x = outputs[len(outputs)-1]
		// and apply default pooling if a pooling config exists
		if m.PoolingConfig != nil {
			x = m.ApplySentencePooling(ctx, x, tokens)
		}
	}

	return x
}

// ApplySentencePooling applies the configured pooling function to the hidden states.
func (m *Model) ApplySentencePooling(ctx *context.Context, hiddenStates, tokens *graph.Node) *graph.Node {
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

// BuildGraph takes the input tokens and creates the GoMLX graph for the model.
// It returns the final sentence embeddings if appropriate, otherwise just final hidden states.
func (m *Model) BuildGraph(ctx *context.Context, tokens *graph.Node) *graph.Node {
	if len(m.Modules) > 0 || m.PoolingConfig != nil {
		return m.SentenceEmbeddingGraph(ctx, tokens)
	}
	
	// Default to just getting the final hidden state of all layers
	outputs := m.BuildAllLayersGraph(ctx, tokens)
	return outputs[len(outputs)-1]
}
