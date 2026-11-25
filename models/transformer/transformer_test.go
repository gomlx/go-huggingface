package transformer

import (
	"encoding/json"
	"testing"

	"github.com/gomlx/compute/dtypes"
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

func TestQwen35ConfigAndTensorMapping(t *testing.T) {
	rawConfig := `{
		"architectures": ["Qwen3_5ForConditionalGeneration"],
		"model_type": "qwen3_5",
		"tie_word_embeddings": true,
		"text_config": {
			"attention_bias": false,
			"attn_output_gate": true,
			"dtype": "bfloat16",
			"full_attention_interval": 4,
			"head_dim": 256,
			"hidden_act": "silu",
			"hidden_size": 2560,
			"intermediate_size": 9216,
			"layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
			"linear_conv_kernel_dim": 4,
			"linear_key_head_dim": 128,
			"linear_num_key_heads": 16,
			"linear_num_value_heads": 32,
			"linear_value_head_dim": 128,
			"max_position_embeddings": 262144,
			"model_type": "qwen3_5_text",
			"num_attention_heads": 16,
			"num_hidden_layers": 4,
			"num_key_value_heads": 4,
			"rms_norm_eps": 1e-06,
			"tie_word_embeddings": true,
			"vocab_size": 248320,
			"mamba_ssm_dtype": "float32",
			"rope_parameters": {
				"mrope_interleaved": true,
				"mrope_section": [11, 11, 10],
				"rope_type": "default",
				"rope_theta": 10000000,
				"partial_rotary_factor": 0.25
			}
		}
	}`

	var cfg Config
	require.NoError(t, json.Unmarshal([]byte(rawConfig), &cfg))
	assert.Equal(t, "qwen3_5_text", cfg.ModelType)
	assert.Equal(t, 2560, cfg.HiddenSize)
	assert.Equal(t, 4, cfg.NumHiddenLayers)
	assert.True(t, cfg.AttnOutputGate)
	assert.Equal(t, 4, cfg.LinearConvKernelDim)
	assert.Equal(t, 16, cfg.LinearNumKeyHeads)
	assert.Equal(t, 32, cfg.LinearNumValueHeads)
	require.NotNil(t, cfg.TieWordEmbeddings)
	assert.True(t, *cfg.TieWordEmbeddings)
	require.Contains(t, cfg.RoPEParameters, "default")
	assert.Equal(t, 10000000.0, cfg.RoPEParameters["default"].RopeTheta)
	assert.Equal(t, 0.25, cfg.RoPEParameters["default"].PartialRotaryFactor)

	m := &Model{Config: cfg}
	m.useCausalMask = m.detectCausalMask()
	assert.True(t, m.useCausalMask)

	scope := model.NewStore().RootScope()
	tm := m.CreateGoMLXModel(scope)
	assert.Equal(t, mltransformer.ArchitectureQwen35, tm.Architecture)
	assert.False(t, tm.ScaleTokenEmbeddings)
	assert.True(t, tm.AttnOutputGate)
	assert.Equal(t, 1.0, tm.RMSNormOffset)
	assert.Equal(t, dtypes.BFloat16, tm.DType)
	assert.Equal(t, dtypes.Float32, tm.LinearStateDType)
	assert.Equal(t, []mltransformer.LayerType{
		mltransformer.LinearLayer,
		mltransformer.LinearLayer,
		mltransformer.LinearLayer,
		mltransformer.GlobalLayer,
	}, tm.LayerTypes)

	// Verify Gemma 4 nested map rope_parameters still parses identically.
	gemma4Raw := `{
		"model_type": "gemma4_text",
		"rope_parameters": {
			"full_attention": {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25},
			"sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}
		}
	}`
	var g4Cfg Config
	require.NoError(t, json.Unmarshal([]byte(gemma4Raw), &g4Cfg))
	require.Contains(t, g4Cfg.RoPEParameters, "full_attention")
	require.Contains(t, g4Cfg.RoPEParameters, "sliding_attention")
	assert.Equal(t, 1000000.0, g4Cfg.RoPEParameters["full_attention"].RopeTheta)
	assert.Equal(t, "proportional", g4Cfg.RoPEParameters["full_attention"].RopeType)
	assert.Equal(t, 10000.0, g4Cfg.RoPEParameters["sliding_attention"].RopeTheta)

	// Verify tensor name mappings for Qwen3.5 linear_attn and ignored visual/mtp weights.
	mapped := map[string]struct {
		scope []string
		name  string
	}{
		"model.language_model.layers.0.linear_attn.in_proj_qkv.weight": {[]string{"layer_0", "linear_attn", "in_proj_qkv", "dense"}, "weights"},
		"model.language_model.layers.0.linear_attn.in_proj_z.weight":   {[]string{"layer_0", "linear_attn", "in_proj_z", "dense"}, "weights"},
		"model.language_model.layers.0.linear_attn.in_proj_b.weight":   {[]string{"layer_0", "linear_attn", "in_proj_b", "dense"}, "weights"},
		"model.language_model.layers.0.linear_attn.in_proj_a.weight":   {[]string{"layer_0", "linear_attn", "in_proj_a", "dense"}, "weights"},
		"model.language_model.layers.0.linear_attn.out_proj.weight":    {[]string{"layer_0", "linear_attn", "out_proj", "dense"}, "weights"},
		"model.language_model.layers.0.linear_attn.conv1d.weight":      {[]string{"layer_0", "linear_attn"}, "conv1d_weight"},
		"model.language_model.layers.0.linear_attn.A_log":              {[]string{"layer_0", "linear_attn"}, "A_log"},
		"model.language_model.layers.0.linear_attn.dt_bias":            {[]string{"layer_0", "linear_attn"}, "dt_bias"},
		"model.language_model.layers.0.linear_attn.norm.weight":        {[]string{"layer_0", "linear_attn", "norm"}, "weight"},
		"lm_head.weight":                                               {[]string{"output", "dense"}, "weights"},
	}
	for tensorName, want := range mapped {
		gotScope, gotName, ok := mapTensorName(tensorName)
		require.True(t, ok, "expected %s to map", tensorName)
		assert.Equal(t, want.scope, gotScope, tensorName)
		assert.Equal(t, want.name, gotName, tensorName)
	}
	for _, skipped := range []string{
		"model.visual.blocks.0.attn.qkv.weight",
		"mtp.layers.0.self_attn.q_proj.weight",
	} {
		_, _, ok := mapTensorName(skipped)
		assert.False(t, ok, "expected %s to be skipped", skipped)
	}
}
