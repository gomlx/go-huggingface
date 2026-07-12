// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sam2

import (
	"encoding/json"
	"fmt"
	"os"
)

// HieraDetConfig represents the backbone configuration.
type HieraDetConfig struct {
	HiddenSize                               int     `json:"hidden_size"`
	NumAttentionHeads                        int     `json:"num_attention_heads"`
	NumChannels                              int     `json:"num_channels"`
	ImageSize                                []int   `json:"image_size"`
	PatchKernelSize                          []int   `json:"patch_kernel_size"`
	PatchStride                              []int   `json:"patch_stride"`
	PatchPadding                             []int   `json:"patch_padding"`
	QueryStride                              []int   `json:"query_stride"`
	WindowPositionalEmbeddingBackgroundSize []int   `json:"window_positional_embedding_background_size"`
	NumQueryPoolStages                       int     `json:"num_query_pool_stages"`
	BlocksPerStage                           []int   `json:"blocks_per_stage"`
	EmbedDimPerStage                         []int   `json:"embed_dim_per_stage"`
	NumAttentionHeadsPerStage                []int   `json:"num_attention_heads_per_stage"`
	WindowSizePerStage                       []int   `json:"window_size_per_stage"`
	GlobalAttentionBlocks                    []int   `json:"global_attention_blocks"`
	MLPRatio                                 float64 `json:"mlp_ratio"`
	HiddenAct                                string  `json:"hidden_act"`
	LayerNormEps                             float64 `json:"layer_norm_eps"`
}

// VisionConfig represents the vision encoder configuration.
// It includes parameters for the Feature Pyramid Network (FPN) neck.
type VisionConfig struct {
	BackboneChannelList  []int          `json:"backbone_channel_list"`
	BackboneFeatureSizes [][]int        `json:"backbone_feature_sizes"`
	// FPNHiddenSize is the hidden dimension of the Feature Pyramid Network (FPN) neck.
	FPNHiddenSize        int            `json:"fpn_hidden_size"`
	// FPNKernelSize is the kernel size used in the Feature Pyramid Network (FPN) neck.
	FPNKernelSize        int            `json:"fpn_kernel_size"`
	// FPNStride is the stride used in the Feature Pyramid Network (FPN) neck.
	FPNStride            int            `json:"fpn_stride"`
	// FPNPadding is the padding used in the Feature Pyramid Network (FPN) neck.
	FPNPadding           int            `json:"fpn_padding"`
	// FPNTopDownLevels specifies the levels of the Feature Pyramid Network (FPN) involved in top-down connections.
	FPNTopDownLevels     []int          `json:"fpn_top_down_levels"`
	NumFeatureLevels     int            `json:"num_feature_levels"`
	HiddenAct            string         `json:"hidden_act"`
	LayerNormEps         float64        `json:"layer_norm_eps"`
	BackboneConfig       HieraDetConfig `json:"backbone_config"`
}

// PromptEncoderConfig represents the prompt encoder configuration.
type PromptEncoderConfig struct {
	HiddenSize         int     `json:"hidden_size"`
	ImageSize          int     `json:"image_size"`
	PatchSize          int     `json:"patch_size"`
	MaskInputChannels  int     `json:"mask_input_channels"`
	NumPointEmbeddings int     `json:"num_point_embeddings"`
	HiddenAct          string  `json:"hidden_act"`
	LayerNormEps       float64 `json:"layer_norm_eps"`
	Scale              float64 `json:"scale"`
}

// MaskDecoderConfig represents the mask decoder configuration.
type MaskDecoderConfig struct {
	HiddenSize                    int     `json:"hidden_size"`
	HiddenAct                     string  `json:"hidden_act"`
	MLPDim                        int     `json:"mlp_dim"`
	NumHiddenLayers               int     `json:"num_hidden_layers"`
	NumAttentionHeads             int     `json:"num_attention_heads"`
	AttentionDownsampleRate       int     `json:"attention_downsample_rate"`
	NumMultimaskOutputs           int     `json:"num_multimask_outputs"`
	IoUHeadDepth                  int     `json:"iou_head_depth"`
	IoUHeadHiddenDim              int     `json:"iou_head_hidden_dim"`
	DynamicMultimaskViaStability  bool    `json:"dynamic_multimask_via_stability"`
	DynamicMultimaskStabilityDelta float64 `json:"dynamic_multimask_stability_delta"`
	DynamicMultimaskStabilityThresh float64 `json:"dynamic_multimask_stability_thresh"`
}

// Config represents the top-level configuration for SAM2.
type Config struct {
	ModelType           string              `json:"model_type"`
	VisionConfig        VisionConfig        `json:"vision_config"`
	PromptEncoderConfig PromptEncoderConfig `json:"prompt_encoder_config"`
	MaskDecoderConfig   MaskDecoderConfig   `json:"mask_decoder_config"`
	TorchDtype          string              `json:"torch_dtype"`
	DType               string              `json:"dtype"`
}

// LoadConfig parses a config.json file.
func LoadConfig(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}
	return &c, nil
}
