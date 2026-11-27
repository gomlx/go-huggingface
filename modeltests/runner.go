// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package modeltests implements end-to-end model parity tests for go-huggingface.
//
// See docs/ModelTests.md for architecture, CLI flags, and how to add new model tests.
package modeltests

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/lipgloss/table"
	"github.com/gomlx/compute/support/humanize"
)

// SharedSentences holds canonical sentences loaded from testdata/reference_sentences.json.
type SharedSentences struct {
	Queries            []string `json:"queries"`
	Documents          []string `json:"documents"`
	TokenizerEdgeCases []string `json:"tokenizer_edge_cases"`
}

// SharedPrompt is a single canonical prompt from testdata/reference_prompts.json.
type SharedPrompt struct {
	Name   string `json:"name"`
	Prompt string `json:"prompt"`
	Raw    bool   `json:"raw"`
}

// SharedPrompts holds canonical prompts loaded from testdata/reference_prompts.json.
type SharedPrompts struct {
	Prompts []SharedPrompt `json:"prompts"`
}

// ModelSpec defines a full-model test suite stored in testdata/<model>/spec.json.
type ModelSpec struct {
	Name         string         `json:"name"`
	Family       string         `json:"family"` // "transformer", "timeseries"
	Format       string         `json:"format"` // "safetensors", "gguf"
	RepoID       string         `json:"repo_id"`
	Revision     string         `json:"revision,omitempty"`
	SizeTier     string         `json:"size_tier"` // "small", "medium", "large"
	Tags         []string       `json:"tags,omitempty"`
	Gated        bool           `json:"gated,omitempty"`
	ModelOptions ModelOptions   `json:"model_options,omitempty"`
	Provenance   Provenance     `json:"provenance,omitempty"`
	Tokenizer    *TokenizerSpec `json:"tokenizer,omitempty"`
	Variants     []VariantSpec  `json:"variants"`
}

// ModelOptions holds optional overrides applied when constructing the model.
type ModelOptions struct {
	CausalMask                *bool    `json:"causal_mask,omitempty"`
	ApproximateGelu           *bool    `json:"approximate_gelu,omitempty"`
	ForceFloat32              *bool    `json:"force_float32,omitempty"`
	UseMMap                   *bool    `json:"use_mmap,omitempty"`
	QuantizedMatmul           *bool    `json:"quantized_matmul,omitempty"`
	IgnoredCheckpointPrefixes []string `json:"ignored_checkpoint_prefixes,omitempty"`
	SkipBackends              []string `json:"skip_backends,omitempty"`
}

// Provenance records how the golden files were produced for reproducibility.
type Provenance struct {
	Generator      string `json:"generator,omitempty"`
	Command        string `json:"command,omitempty"`
	RuntimeVersion string `json:"runtime_version,omitempty"`
	DType          string `json:"dtype,omitempty"`
	Device         string `json:"device,omitempty"`
	GeneratedAt    string `json:"generated_at,omitempty"`
}

// TokenizerSpec holds reference tokenization and detokenization expectations.
type TokenizerSpec struct {
	SpecialTokens map[string]int  `json:"special_tokens,omitempty"`
	Cases         []TokenizerCase `json:"cases"`
}

// TokenizerCase holds expected token IDs and decoded text for a single input string.
type TokenizerCase struct {
	Text               string `json:"text"`
	TokenIDs           []int  `json:"token_ids"`
	Decoded            string `json:"decoded,omitempty"`
	DecodedSkipSpecial string `json:"decoded_skip_special,omitempty"`
	SkipDecodeCheck    bool   `json:"skip_decode_check,omitempty"`
}

// VariantSpec defines one testable checkpoint variant of a model (e.g., a specific GGUF quant or model size).
type VariantSpec struct {
	Name         string          `json:"name"`
	RepoID       string          `json:"repo_id,omitempty"`   // Optional override of ModelSpec.RepoID
	Revision     string          `json:"revision,omitempty"`  // Optional override of ModelSpec.Revision
	GGUFFile     string          `json:"gguf_file,omitempty"` // Specific .gguf filename in multi-file repos
	ModelOptions *ModelOptions   `json:"model_options,omitempty"`
	Tokenizer    *TokenizerSpec  `json:"tokenizer,omitempty"` // Optional variant-specific tokenizer expectations
	Embedding    *EmbeddingSpec  `json:"embedding,omitempty"`
	Generate     *GenerateSpec   `json:"generate,omitempty"`
	TimeSeries   *TimeSeriesSpec `json:"timeseries,omitempty"`
}

// EmbeddingSpec configures sentence embedding parity tests.
type EmbeddingSpec struct {
	UseSharedSentences bool                `json:"use_shared_sentences,omitempty"`
	QueryPrefix        string              `json:"query_prefix,omitempty"`
	DocumentPrefix     string              `json:"document_prefix,omitempty"`
	Sentences          []string            `json:"sentences,omitempty"`       // Overrides shared sentences if non-empty
	ExtraSentences     []string            `json:"extra_sentences,omitempty"` // Appended to shared sentences
	EmbeddingDim       int                 `json:"embedding_dim"`
	EmbeddingsFile     string              `json:"embeddings_file"` // Relative path to little-endian .f32 file
	SimilarityMatrix   [][]float32         `json:"similarity_matrix,omitempty"`
	SkipBatchTest      bool                `json:"skip_batch_test,omitempty"`
	Tolerances         EmbeddingTolerances `json:"tolerances,omitempty"`
}

// EmbeddingTolerances controls numerical thresholds for embedding comparison.
type EmbeddingTolerances struct {
	MinCosineSim       float64 `json:"min_cosine_sim,omitempty"`       // Default 0.999
	MaxAbsDiff         float64 `json:"max_abs_diff,omitempty"`         // Default 0.05
	MaxMeanAbsDiff     float64 `json:"max_mean_abs_diff,omitempty"`    // Default 0.01
	SimilarityMaxDelta float64 `json:"similarity_max_delta,omitempty"` // Default 0.01
}

// GenerateSpec configures causal/generative LLM parity tests (temperature = 0).
type GenerateSpec struct {
	ChatTemplatePrefix string         `json:"chat_template_prefix,omitempty"`
	ChatTemplateSuffix string         `json:"chat_template_suffix,omitempty"`
	StopTokens         []string       `json:"stop_tokens,omitempty"`
	MaxNewTokens       int            `json:"max_new_tokens,omitempty"`
	SkipKVCacheParity  bool           `json:"skip_kvcache_parity,omitempty"`
	UseNaiveOnly       bool           `json:"use_naive_only,omitempty"`
	Cases              []GenerateCase `json:"cases"`
}

// GenerateCase holds golden prefill top-K logits and greedy generation outputs for one prompt.
type GenerateCase struct {
	Name            string             `json:"name"`
	Prompt          string             `json:"prompt"`
	FormattedPrompt string             `json:"formatted_prompt,omitempty"`
	PromptTokenIDs  []int              `json:"prompt_token_ids"`
	PrefillTopK     []TopKLogit        `json:"prefill_top_k,omitempty"`
	GreedyTokenIDs  []int              `json:"greedy_token_ids,omitempty"`
	GreedyText      string             `json:"greedy_text,omitempty"`
	Tolerances      GenerateTolerances `json:"tolerances,omitempty"`
}

// TopKLogit records one top-K logit entry at the final prompt position.
type TopKLogit struct {
	TokenID int     `json:"token_id"`
	Text    string  `json:"text,omitempty"`
	Logit   float32 `json:"logit"`
}

// GenerateTolerances controls numerical thresholds for generative LLM checks.
type GenerateTolerances struct {
	LogitMaxAbsDiff      float64 `json:"logit_max_abs_diff,omitempty"`      // Default 0.5
	LogitMinCosineSim    float64 `json:"logit_min_cosine_sim,omitempty"`    // Default 0.995
	Top1MarginTolerance  float64 `json:"top1_margin_tolerance,omitempty"`   // Allow top-1 flip if ref top1-top2 margin < this (default 0.25)
	MinExactPrefixTokens int     `json:"min_exact_prefix_tokens,omitempty"` // Minimum matching greedy prefix tokens (default: all)
	KVCacheParityTokens  int     `json:"kvcache_parity_tokens,omitempty"`   // Number of tokens for KV-cache vs naive check (default 8)
}

// TimeSeriesSpec configures time-series model parity tests (e.g., Timestral).
type TimeSeriesSpec struct {
	Batch              int     `json:"batch"`
	ContextSteps       int     `json:"context_steps"`
	HorizonSteps       int     `json:"horizon_steps"`
	ContextFile        string  `json:"context_file"`
	ResultsFile        string  `json:"results_file"`
	SymmetricAveraging bool    `json:"symmetric_averaging"`
	ClipNonNegative    bool    `json:"clip_non_negative"`
	MaxAbsDiff         float64 `json:"max_abs_diff,omitempty"` // Default 1e-4
}

type loadedSpec struct {
	Dir  string
	Spec ModelSpec
}

func loadModelSpecs(testdataDir string) ([]loadedSpec, error) {
	entries, err := os.ReadDir(testdataDir)
	if err != nil {
		return nil, fmt.Errorf("reading testdata dir %q: %w", testdataDir, err)
	}
	var specs []loadedSpec
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dir := filepath.Join(testdataDir, e.Name())
		specPath := filepath.Join(dir, "spec.json")
		data, err := os.ReadFile(specPath)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, fmt.Errorf("reading %q: %w", specPath, err)
		}
		var spec ModelSpec
		if err := json.Unmarshal(data, &spec); err != nil {
			return nil, fmt.Errorf("parsing %q: %w", specPath, err)
		}
		if spec.Name == "" {
			spec.Name = e.Name()
		}
		if len(spec.Variants) == 0 {
			spec.Variants = []VariantSpec{{Name: "default"}}
		}
		specs = append(specs, loadedSpec{Dir: dir, Spec: spec})
	}
	slices.SortFunc(specs, func(a, b loadedSpec) int {
		return strings.Compare(a.Spec.Name, b.Spec.Name)
	})
	return specs, nil
}

func loadSharedSentences(testdataDir string) (*SharedSentences, error) {
	path := filepath.Join(testdataDir, "reference_sentences.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading %q: %w", path, err)
	}
	var s SharedSentences
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, fmt.Errorf("parsing %q: %w", path, err)
	}
	return &s, nil
}

func loadSharedPrompts(testdataDir string) (*SharedPrompts, error) {
	path := filepath.Join(testdataDir, "reference_prompts.json")
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading %q: %w", path, err)
	}
	var p SharedPrompts
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, fmt.Errorf("parsing %q: %w", path, err)
	}
	return &p, nil
}

// resolveSentences returns the list of input strings to embed for an EmbeddingSpec.
func resolveSentences(emb *EmbeddingSpec, shared *SharedSentences) []string {
	if len(emb.Sentences) > 0 {
		return emb.Sentences
	}
	var out []string
	if emb.UseSharedSentences && shared != nil {
		for _, q := range shared.Queries {
			out = append(out, emb.QueryPrefix+q)
		}
		for _, d := range shared.Documents {
			out = append(out, emb.DocumentPrefix+d)
		}
	}
	out = append(out, emb.ExtraSentences...)
	return out
}

// readF32File reads a raw little-endian float32 binary file.
func readF32File(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("%q has %d bytes, not a multiple of 4", path, len(data))
	}
	out := make([]float32, len(data)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return out, nil
}

type embeddingMetrics struct {
	CosineSim   float64
	MaxAbsDiff  float64
	MeanAbsDiff float64
	RMSE        float64
}

func computeEmbeddingMetrics(got, want []float32) embeddingMetrics {
	if len(got) == 0 || len(got) != len(want) {
		return embeddingMetrics{}
	}
	var dot, normGotSq, normWantSq, sumAbs, sumSq, maxAbs float64
	for i := range got {
		g := float64(got[i])
		w := float64(want[i])
		d := math.Abs(g - w)
		dot += g * w
		normGotSq += g * g
		normWantSq += w * w
		sumAbs += d
		sumSq += d * d
		if d > maxAbs {
			maxAbs = d
		}
	}
	n := float64(len(got))
	cosSim := 0.0
	if denom := math.Sqrt(normGotSq) * math.Sqrt(normWantSq); denom > 0 {
		cosSim = dot / denom
	}
	return embeddingMetrics{
		CosineSim:   cosSim,
		MaxAbsDiff:  maxAbs,
		MeanAbsDiff: sumAbs / n,
		RMSE:        math.Sqrt(sumSq / n),
	}
}

func tierRank(tier string) int {
	switch strings.ToLower(strings.TrimSpace(tier)) {
	case "small", "":
		return 1
	case "medium":
		return 2
	case "large":
		return 3
	default:
		return 99
	}
}

func mergeModelOptions(base ModelOptions, override *ModelOptions) ModelOptions {
	if override == nil {
		return base
	}
	merged := base
	if override.CausalMask != nil {
		merged.CausalMask = override.CausalMask
	}
	if override.ApproximateGelu != nil {
		merged.ApproximateGelu = override.ApproximateGelu
	}
	if override.ForceFloat32 != nil {
		merged.ForceFloat32 = override.ForceFloat32
	}
	if override.UseMMap != nil {
		merged.UseMMap = override.UseMMap
	}
	if override.QuantizedMatmul != nil {
		merged.QuantizedMatmul = override.QuantizedMatmul
	}
	if len(override.IgnoredCheckpointPrefixes) > 0 {
		merged.IgnoredCheckpointPrefixes = override.IgnoredCheckpointPrefixes
	}
	if len(override.SkipBackends) > 0 {
		merged.SkipBackends = override.SkipBackends
	}
	return merged
}

// runReportRow records timing and peak memory for one (model, variant, backend) execution.
type runReportRow struct {
	Model     string
	Variant   string
	Backend   string
	Status    string
	LoadTime  time.Duration
	InferTime time.Duration
	Mem       MemoryStats
	Notes     string
}

func printSummaryTable(rows []runReportRow) {
	if len(rows) == 0 {
		return
	}
	baseStyle := lipgloss.NewStyle().Padding(0, 1)
	tbl := table.New().
		Border(lipgloss.NormalBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(lipgloss.Color("238"))).
		Headers("Model", "Variant", "Backend", "Status", "Load", "Infer", "Peak RSS", "Delta RSS", "Peak Heap", "Notes").
		StyleFunc(func(row, col int) lipgloss.Style {
			st := baseStyle
			if row == table.HeaderRow {
				return st.Foreground(lipgloss.Color("252")).Bold(true)
			}
			if col >= 4 && col <= 8 {
				st = st.Align(lipgloss.Right)
			}
			return st
		})
	for _, r := range rows {
		loadStr := "-"
		if r.LoadTime > 0 {
			loadStr = humanize.Duration(r.LoadTime)
		}
		inferStr := "-"
		if r.InferTime > 0 {
			inferStr = humanize.Duration(r.InferTime)
		}
		peakRSS := "-"
		if r.Mem.PeakRSSBytes > 0 {
			peakRSS = humanize.Bytes(r.Mem.PeakRSSBytes)
		}
		deltaRSS := "-"
		if r.Mem.DeltaPeakRSSBytes > 0 {
			deltaRSS = "+" + humanize.Bytes(r.Mem.DeltaPeakRSSBytes)
		}
		peakHeap := "-"
		if r.Mem.PeakHeapAllocBytes > 0 {
			peakHeap = humanize.Bytes(r.Mem.PeakHeapAllocBytes)
		}
		tbl.Row(r.Model, r.Variant, r.Backend, r.Status, loadStr, inferStr, peakRSS, deltaRSS, peakHeap, r.Notes)
	}
	fmt.Printf("\n=== Model Tests Summary ===\n%s\n", tbl)
}
