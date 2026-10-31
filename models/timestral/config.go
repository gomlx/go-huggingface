// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package timestral

import (
	"encoding/json"
	"math"
	"os"
	"slices"

	"github.com/pkg/errors"
)

// Well-known Hugging Face Hub repository IDs for Mistral AI's Timestral checkpoints.
const (
	RepoTimestral5M   = "mistralai/Timestral-5M"
	RepoTimestral28M  = "mistralai/Timestral-28M"
	RepoTimestral123M = "mistralai/Timestral-123M"
	RepoTimestral209M = "mistralai/Timestral-209M"
)

// LayerKind specifies the sequence of attention sublayers in a TransformerBlock.
type LayerKind string

const (
	LayerTime        LayerKind = "time"
	LayerVariate     LayerKind = "variate"
	LayerTimeVariate LayerKind = "time_variate"
	LayerVariateTime LayerKind = "variate_time"
)

// Axes returns the ordered attention axes for this LayerKind (true = time axis, false = variate axis).
func (k LayerKind) Axes() ([]bool, error) {
	switch k {
	case LayerTime:
		return []bool{true}, nil
	case LayerVariate:
		return []bool{false}, nil
	case LayerTimeVariate:
		return []bool{true, false}, nil
	case LayerVariateTime:
		return []bool{false, true}, nil
	default:
		return nil, errors.Errorf("unknown layer kind %q; expected one of time, variate, time_variate, variate_time", k)
	}
}

// AttentionGate constants supported by Timestral attention layers.
const (
	GateNone        = "none"
	GateSDPASigmoid = "sdpa_sigmoid"
)

// Row role constants for packed (R, T) rows.
const (
	RoleTarget   = 0
	RolePastCov  = 1
	RoleKnownCov = 2
)

// Config holds the architecture and decode settings of a Timestral checkpoint (`config.json`).
type Config struct {
	PatchSize               int         `json:"patch_size"`
	DModel                  int         `json:"d_model"`
	NumHeads                int         `json:"num_heads"`
	MLPHiddenDim            int         `json:"mlp_hidden_dim"`
	BlockSchedule           []LayerKind `json:"block_schedule"`
	QuantileKnots           []float64   `json:"quantile_knots"`
	BlockSize               int         `json:"block_size"`
	ContextLength           int         `json:"context_length"`
	RoPETheta               float64     `json:"rope_theta"`
	RoPEPartialFactor       float64     `json:"rope_partial_factor"`
	NormEps                 float64     `json:"norm_eps"`
	AttnBias                bool        `json:"attn_bias"`
	MLPBias                 bool        `json:"mlp_bias"`
	TimeAttnGate            string      `json:"time_attn_gate"`
	VariateAttnGate         string      `json:"variate_attn_gate"`
	MinimumScale            float64     `json:"minimum_scale"`
	UseAsinh                bool        `json:"use_asinh"`
	StabilizeWithGlobal     bool        `json:"stabilize_with_global"`
	CovariateEnabled        bool        `json:"covariate_enabled"`
	SplitContextHorizonAttn bool        `json:"split_context_horizon_attn"`
	AttendMissingPatches    bool        `json:"attend_missing_patches"`
	QuantileWidthFactor     float64     `json:"quantile_width_factor"`
}

// DefaultConfig returns a Config pre-populated with Timestral's default fields.
func DefaultConfig() *Config {
	return &Config{
		PatchSize:               32,
		DModel:                  384,
		NumHeads:                6,
		MLPHiddenDim:            1024,
		BlockSchedule:           []LayerKind{LayerTime, LayerTime, LayerTime, LayerTimeVariate},
		QuantileKnots:           []float64{0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95},
		BlockSize:               8,
		ContextLength:           4096,
		RoPETheta:               10000.0,
		RoPEPartialFactor:       0.5,
		NormEps:                 1e-6,
		AttnBias:                false,
		MLPBias:                 false,
		TimeAttnGate:            GateSDPASigmoid,
		VariateAttnGate:         GateSDPASigmoid,
		MinimumScale:            1e-6,
		UseAsinh:                true,
		StabilizeWithGlobal:     false,
		CovariateEnabled:        true,
		SplitContextHorizonAttn: true,
		AttendMissingPatches:    true,
		QuantileWidthFactor:     2.0,
	}
}

// ParseConfig unmarshals and validates a Timestral Config from JSON bytes.
func ParseConfig(data []byte) (*Config, error) {
	cfg := DefaultConfig()
	if err := json.Unmarshal(data, cfg); err != nil {
		return nil, errors.Wrap(err, "failed to parse Timestral config.json")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

// LoadConfig reads, parses, and validates a Timestral Config from a local file path.
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read Timestral config file %q", path)
	}
	return ParseConfig(data)
}

// ApplyDefaults fills in zero-valued optional numerical/string fields with Timestral defaults.
func (c *Config) ApplyDefaults() {
	if c.RoPETheta == 0 {
		c.RoPETheta = 10000.0
	}
	if c.RoPEPartialFactor == 0 {
		c.RoPEPartialFactor = 0.5
	}
	if c.NormEps == 0 {
		c.NormEps = 1e-6
	}
	if c.TimeAttnGate == "" {
		c.TimeAttnGate = GateSDPASigmoid
	}
	if c.VariateAttnGate == "" {
		c.VariateAttnGate = GateSDPASigmoid
	}
	if c.MinimumScale == 0 {
		c.MinimumScale = 1e-6
	}
	if c.QuantileWidthFactor == 0 {
		c.QuantileWidthFactor = 2.0
	}
}

// Validate checks that all configuration parameters are valid and consistent.
func (c *Config) Validate() error {
	c.ApplyDefaults()
	if c.PatchSize <= 0 {
		return errors.Errorf("patch_size must be > 0, got %d", c.PatchSize)
	}
	if c.DModel <= 0 || c.NumHeads <= 0 || c.DModel%c.NumHeads != 0 {
		return errors.Errorf("d_model=%d is not divisible by num_heads=%d", c.DModel, c.NumHeads)
	}
	if c.MLPHiddenDim <= 0 {
		return errors.Errorf("mlp_hidden_dim must be > 0, got %d", c.MLPHiddenDim)
	}
	if c.BlockSize <= 0 {
		return errors.Errorf("block_size must be > 0, got %d", c.BlockSize)
	}
	if c.ContextLength <= 0 {
		return errors.Errorf("context_length must be > 0, got %d", c.ContextLength)
	}
	ropeDim := c.RoPEDim()
	if ropeDim <= 0 || ropeDim%2 != 0 || ropeDim > c.HeadDim() {
		return errors.Errorf("rope_partial_factor=%v gives an invalid rope_dim=%d for head_dim=%d",
			c.RoPEPartialFactor, ropeDim, c.HeadDim())
	}
	if len(c.BlockSchedule) == 0 {
		return errors.New("block_schedule must not be empty")
	}
	for _, kind := range c.BlockSchedule {
		if _, err := kind.Axes(); err != nil {
			return err
		}
	}
	for _, gate := range []string{c.TimeAttnGate, c.VariateAttnGate} {
		if gate != GateNone && gate != GateSDPASigmoid {
			return errors.Errorf("unknown attention gate %q; expected %q or %q", gate, GateNone, GateSDPASigmoid)
		}
	}
	if len(c.QuantileKnots) == 0 {
		return errors.New("quantile_knots must not be empty")
	}
	if c.MedianIndex() < 0 {
		return errors.Errorf("quantile_knots must contain 0.5, got %v", c.QuantileKnots)
	}
	for i := 1; i < len(c.QuantileKnots); i++ {
		if c.QuantileKnots[i-1] >= c.QuantileKnots[i] {
			return errors.Errorf("quantile_knots must be strictly increasing, got %v", c.QuantileKnots)
		}
	}
	if c.StabilizeWithGlobal {
		return errors.New("the advancing refine frame needs stabilize_with_global=false")
	}
	return nil
}

// HeadDim returns DModel / NumHeads.
func (c *Config) HeadDim() int {
	return c.DModel / c.NumHeads
}

// RoPEDim returns the number of head dimensions rotated by RoPE; the rest pass through unrotated.
func (c *Config) RoPEDim() int {
	return int(float64(c.HeadDim()) * c.RoPEPartialFactor)
}

// NumQuantiles returns len(QuantileKnots).
func (c *Config) NumQuantiles() int {
	return len(c.QuantileKnots)
}

// MedianIndex returns the index of 0.5 in QuantileKnots, or -1 if not found.
func (c *Config) MedianIndex() int {
	for i, q := range c.QuantileKnots {
		if math.Abs(q-0.5) < 1e-9 {
			return i
		}
	}
	return -1
}

// AreQuantilesSymmetric checks whether QuantileKnots are symmetric around 0.5 (q[i] + q[Q-1-i] == 1).
func (c *Config) AreQuantilesSymmetric() bool {
	n := len(c.QuantileKnots)
	for i := range n {
		if math.Abs(c.QuantileKnots[i]+c.QuantileKnots[n-1-i]-1.0) >= 1e-9 {
			return false
		}
	}
	return true
}

// SupportsKnownCovariates returns whether the layer schedule can route a known covariate's future to the targets.
// Some time op must be followed by at least one more time op and at least one variate op.
func (c *Config) SupportsKnownCovariates() bool {
	if !c.CovariateEnabled {
		return false
	}
	var ops []bool
	for _, kind := range c.BlockSchedule {
		axes, err := kind.Axes()
		if err != nil {
			return false
		}
		ops = append(ops, axes...)
	}
	for i, isTime := range ops {
		if !isTime {
			continue
		}
		rest := ops[i+1:]
		hasTime := slices.Contains(rest, true)
		hasVariate := slices.Contains(rest, false)
		if hasTime && hasVariate {
			return true
		}
	}
	return false
}
