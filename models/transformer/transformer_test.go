package transformer

import (
	"testing"

	"github.com/gomlx/gomlx/ml/model"
	mltransformer "github.com/gomlx/gomlx/ml/zoo/transformer"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDetectCausalMask(t *testing.T) {
	tests := []struct {
		name       string
		cfg        Config
		pooling    *PoolingConfig
		wantCausal bool
	}{
		{
			name: "BERT by model_type",
			cfg: Config{
				ModelType: "bert",
			},
			wantCausal: false,
		},
		{
			name: "BERT by architecture",
			cfg: Config{
				Architectures: []string{"BertModel"},
			},
			wantCausal: false,
		},
		{
			name: "RoBERTa by model_type",
			cfg: Config{
				ModelType: "roberta",
			},
			wantCausal: false,
		},
		{
			name: "DeBERTa by architecture",
			cfg: Config{
				Architectures: []string{"DebertaForMaskedLM"},
			},
			wantCausal: false,
		},
		{
			name: "Gemma 3 text model",
			cfg: Config{
				ModelType:     "gemma3_text",
				Architectures: []string{"Gemma3TextModel"},
			},
			wantCausal: true,
		},
		{
			name: "Gemma 4 model",
			cfg: Config{
				ModelType:     "gemma4",
				Architectures: []string{"Gemma4ForConditionalGeneration"},
			},
			wantCausal: true,
		},
		{
			name: "LLaMA causal LM",
			cfg: Config{
				ModelType:     "llama",
				Architectures: []string{"LlamaForCausalLM"},
			},
			wantCausal: true,
		},
		{
			name: "GPT2 LM Head",
			cfg: Config{
				ModelType:     "gpt2",
				Architectures: []string{"GPT2LMHeadModel"},
			},
			wantCausal: true,
		},
		{
			name: "Unknown architecture with last-token pooling",
			cfg: Config{
				ModelType: "custom_embedding",
			},
			pooling: &PoolingConfig{
				PoolingModeLastToken: true,
			},
			wantCausal: true,
		},
		{
			name: "Explicit is_decoder true in Extra",
			cfg: Config{
				ModelType: "custom",
				Extra: map[string]any{
					"is_decoder": true,
				},
			},
			wantCausal: true,
		},
		{
			name: "Explicit is_decoder false in Extra",
			cfg: Config{
				ModelType: "custom",
				Extra: map[string]any{
					"is_decoder": false,
				},
			},
			wantCausal: false,
		},
		{
			name:       "Unknown fallback defaults to true",
			cfg:        Config{},
			wantCausal: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := &Model{
				Config:        tc.cfg,
				PoolingConfig: tc.pooling,
			}
			assert.Equal(t, tc.wantCausal, m.detectCausalMask())
		})
	}
}

func TestWithCausalMaskOverride(t *testing.T) {
	m := &Model{}
	assert.False(t, m.UseCausalMask()) // uninitialized zero-value

	m.WithCausalMask(true)
	assert.True(t, m.UseCausalMask())

	m.WithCausalMask(false)
	assert.False(t, m.UseCausalMask())
}

func TestCreateGoMLXModelPropagatesCausalMask(t *testing.T) {
	scope := model.NewStore().RootScope()

	// 1. Decoder model (Gemma) with causal mask true
	gemmaModel := (&Model{
		Config: Config{
			ModelType:         "gemma3_text",
			HiddenSize:        256,
			NumHiddenLayers:   2,
			NumAttentionHeads: 4,
			VocabSize:         1000,
		},
		useCausalMask: true,
	})

	tmGemma := gemmaModel.CreateGoMLXModel(scope.In("gemma"))
	require.NotNil(t, tmGemma)
	assert.True(t, tmGemma.UseCausalMask)

	// 2. BERT model with causal mask false
	bertModel := (&Model{
		Config: Config{
			ModelType:         "bert",
			HiddenSize:        256,
			NumHiddenLayers:   2,
			NumAttentionHeads: 4,
			VocabSize:         1000,
		},
		useCausalMask: false,
	})

	tmBert := bertModel.CreateGoMLXModel(scope.In("bert"))
	require.NotNil(t, tmBert)
	assert.False(t, tmBert.UseCausalMask)
	assert.Equal(t, mltransformer.ArchitectureStandard, tmBert.Architecture)

	// 3. User override: BERT with causal mask explicitly set to true
	bertModelCausal := (&Model{
		Config: Config{
			ModelType:         "bert",
			HiddenSize:        256,
			NumHiddenLayers:   2,
			NumAttentionHeads: 4,
			VocabSize:         1000,
		},
	}).WithCausalMask(true)

	tmBertCausal := bertModelCausal.CreateGoMLXModel(scope.In("bert_causal"))
	require.NotNil(t, tmBertCausal)
	assert.True(t, tmBertCausal.UseCausalMask)

	// 4. Gemma 2 model configuration (ArchitectureGemma, attn_logit_softcapping, query_pre_attn_scalar, alternating layer types)
	gemma2Model := (&Model{
		Config: Config{
			ModelType:             "gemma2",
			Architectures:         []string{"Gemma2ForCausalLM"},
			HiddenSize:            2304,
			NumHiddenLayers:       4,
			NumAttentionHeads:     8,
			NumKeyValueHeads:      4,
			HeadDim:               256,
			IntermediateSize:      9216,
			VocabSize:             256000,
			HiddenActivation:      "gelu_pytorch_tanh",
			AttnLogitSoftcapping:  50.0,
			FinalLogitSoftcapping: 30.0,
			QueryPreAttnScalar:    256.0,
			SlidingWindow:         4096,
		},
		useCausalMask: true,
	})
	tmGemma2 := gemma2Model.CreateGoMLXModel(scope.In("gemma2"))
	require.NotNil(t, tmGemma2)
	assert.Equal(t, mltransformer.ArchitectureGemma, tmGemma2.Architecture)
	assert.Equal(t, 50.0, tmGemma2.AttentionScoreSoftCap)
	assert.Equal(t, 30.0, tmGemma2.FinalLogitSoftCap)
	assert.InDelta(t, 1.0/16.0, tmGemma2.QueryKeyScale, 1e-9)
	assert.Equal(t, []mltransformer.LayerType{
		mltransformer.LocalLayer,
		mltransformer.GlobalLayer,
		mltransformer.LocalLayer,
		mltransformer.GlobalLayer,
	}, tmGemma2.LayerTypes)
}
