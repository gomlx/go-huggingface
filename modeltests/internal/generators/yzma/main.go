// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Command yzma generates modeltests spec.json and binary .f32 reference files for GGUF
// models using llama.cpp via github.com/hybridgroup/yzma.
//
// It lives in its own Go module so hybridgroup/yzma does not enter go-huggingface's
// dependency graph. Tokenization ground truth is generated directly by llama.cpp
// (llama.Tokenize / llama.Detokenize) so go-huggingface's embedded GGUF tokenizer is
// tested against an independent implementation.
//
// Usage:
//
//	cd modeltests/internal/generators/yzma
//	go run . -model second-state/All-MiniLM-L6-v2-Embedding-GGUF -all_quants -out ../../../testdata/minilm-l6-v2-gguf
//	go run . -model ChristianAzinn/bge-base-en-v1.5-gguf -quant Q4_0 -out ../../../testdata/bge-base-en-v1.5-gguf
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/gguf"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/hybridgroup/yzma/pkg/download"
	"github.com/hybridgroup/yzma/pkg/llama"
	"k8s.io/klog/v2"
)

var (
	flagModel        = flag.String("model", "", "HuggingFace GGUF model repository ID (required)")
	flagGGUFFile     = flag.String("gguf_file", "", "Specific .gguf filename to export (mutually exclusive with -quant / -all_quants)")
	flagQuant        = flag.String("quant", "", "GGUF quantization tag to select (e.g. Q4_0, Q4_K_M, Q8_0, F16)")
	flagAllQuants    = flag.Bool("all_quants", false, "Export all non-mmproj .gguf files in the repository as variants")
	flagOutDir       = flag.String("out", "", "Output directory under modeltests/testdata/<name> (required)")
	flagName         = flag.String("name", "", "Model slug name (defaults to basename of -out)")
	flagTier         = flag.String("tier", "small", "Model size tier: small, medium, or large")
	flagQueryPrefix  = flag.String("query_prefix", "", "Optional prefix prepended to shared query sentences")
	flagDocPrefix    = flag.String("doc_prefix", "", "Optional prefix prepended to shared document sentences")
	flagRefSentences = flag.String("ref_sentences", "../../../testdata/reference_sentences.json", "Path to shared reference_sentences.json")
	flagMinCosine    = flag.Float64("min_cosine", 0.9995, "Minimum cosine similarity tolerance")
	flagMaxAbsDiff   = flag.Float64("max_abs_diff", 6e-3, "Maximum absolute element diff tolerance")
	flagMaxMeanDiff  = flag.Float64("max_mean_diff", 1.5e-3, "Maximum mean absolute diff tolerance")

	// llama.cpp / yzma options.
	flagLibDir    = flag.String("lib", defaultLibDir(), "Directory to install/load llama.cpp shared libraries")
	flagProcessor = flag.String("processor", "cpu", "llama.cpp processor library variant (cpu, cuda, vulkan, metal)")
	flagNGL       = flag.Int("ngl", 0, "Number of GPU layers to offload (0 = CPU only)")
	flagThreads   = flag.Int("threads", 0, "Number of CPU threads (0 = llama.cpp default)")
	flagVerbose   = flag.Bool("verbose_llama", false, "Enable llama.cpp internal logging")
)

func defaultLibDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "/tmp/yzma_lib"
	}
	return filepath.Join(home, ".cache", "yzma", "lib")
}

// Schema structs matching modeltests/runner.go.
type sharedSentences struct {
	Queries            []string `json:"queries"`
	Documents          []string `json:"documents"`
	TokenizerEdgeCases []string `json:"tokenizer_edge_cases"`
}

type modelSpec struct {
	Name         string         `json:"name"`
	Family       string         `json:"family"`
	Format       string         `json:"format"`
	RepoID       string         `json:"repo_id"`
	Revision     string         `json:"revision,omitempty"`
	SizeTier     string         `json:"size_tier"`
	Tags         []string       `json:"tags,omitempty"`
	ModelOptions modelOptions   `json:"model_options,omitempty"`
	Provenance   provenance     `json:"provenance,omitempty"`
	Tokenizer    *tokenizerSpec `json:"tokenizer,omitempty"`
	Variants     []variantSpec  `json:"variants"`
}

type modelOptions struct {
	CausalMask      *bool `json:"causal_mask,omitempty"`
	ApproximateGelu *bool `json:"approximate_gelu,omitempty"`
}

type provenance struct {
	Generator      string `json:"generator,omitempty"`
	Command        string `json:"command,omitempty"`
	RuntimeVersion string `json:"runtime_version,omitempty"`
	DType          string `json:"dtype,omitempty"`
	Device         string `json:"device,omitempty"`
	GeneratedAt    string `json:"generated_at,omitempty"`
}

type tokenizerSpec struct {
	SpecialTokens map[string]int  `json:"special_tokens,omitempty"`
	Cases         []tokenizerCase `json:"cases"`
}

type tokenizerCase struct {
	Text               string `json:"text"`
	TokenIDs           []int  `json:"token_ids"`
	Decoded            string `json:"decoded,omitempty"`
	DecodedSkipSpecial string `json:"decoded_skip_special,omitempty"`
	SkipDecodeCheck    bool   `json:"skip_decode_check,omitempty"`
}

type variantSpec struct {
	Name      string         `json:"name"`
	GGUFFile  string         `json:"gguf_file,omitempty"`
	Embedding *embeddingSpec `json:"embedding,omitempty"`
}

type embeddingSpec struct {
	UseSharedSentences bool                `json:"use_shared_sentences,omitempty"`
	QueryPrefix        string              `json:"query_prefix,omitempty"`
	DocumentPrefix     string              `json:"document_prefix,omitempty"`
	EmbeddingDim       int                 `json:"embedding_dim"`
	EmbeddingsFile     string              `json:"embeddings_file"`
	SimilarityMatrix   [][]float32         `json:"similarity_matrix,omitempty"`
	Tolerances         embeddingTolerances `json:"tolerances,omitempty"`
}

type embeddingTolerances struct {
	MinCosineSim       float64 `json:"min_cosine_sim,omitempty"`
	MaxAbsDiff         float64 `json:"max_abs_diff,omitempty"`
	MaxMeanAbsDiff     float64 `json:"max_mean_abs_diff,omitempty"`
	SimilarityMaxDelta float64 `json:"similarity_max_delta,omitempty"`
}

func ensureLlamaLoaded(libDir, processor string, verbose bool) {
	if err := os.MkdirAll(libDir, 0755); err != nil {
		klog.Fatalf("Failed to create lib dir %q: %v", libDir, err)
	}
	if !download.AlreadyInstalled(libDir) {
		fmt.Fprintf(os.Stderr, "Downloading llama.cpp (%s, %s) into %s...\n", download.DefaultVersion, processor, libDir)
		if err := download.Get(runtime.GOARCH, runtime.GOOS, processor, download.DefaultVersion, libDir); err != nil {
			klog.Fatalf("download.Get failed: %v", err)
		}
	}
	if err := llama.Load(libDir); err != nil {
		klog.Fatalf("llama.Load(%q) failed: %v", libDir, err)
	}
	if !verbose {
		llama.LogSet(llama.LogSilent())
	}
	llama.Init()
}

func poolingFromHFModel(m *transformer.Model) (llama.PoolingType, bool) {
	if m.PoolingConfig != nil {
		switch {
		case m.PoolingConfig.PoolingModeClsToken:
			return llama.PoolingTypeCLS, shouldNormalizeHFModel(m)
		case m.PoolingConfig.PoolingModeLastToken:
			return llama.PoolingTypeLast, shouldNormalizeHFModel(m)
		case m.PoolingConfig.PoolingModeMeanTokens:
			return llama.PoolingTypeMean, shouldNormalizeHFModel(m)
		}
	}
	if m.UseCausalMask() {
		return llama.PoolingTypeLast, shouldNormalizeHFModel(m)
	}
	return llama.PoolingTypeMean, shouldNormalizeHFModel(m)
}

func shouldNormalizeHFModel(m *transformer.Model) bool {
	if len(m.Modules) == 0 {
		return true
	}
	for _, mod := range m.Modules {
		if mod.Type == "sentence_transformers.models.Normalize" {
			return true
		}
	}
	return false
}

func variantNameFromFile(ggufFile string) string {
	base := filepath.Base(ggufFile)
	base = strings.TrimSuffix(base, filepath.Ext(base))
	// Extract standard quantization suffix if present.
	knownTags := []string{
		"Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
		"Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
		"Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
		"Q6_K", "Q8_0", "Q8_K", "IQ4_NL",
		"BF16", "F16", "F32",
	}
	upper := strings.ToUpper(base)
	for _, tag := range knownTags {
		if strings.HasSuffix(upper, "-"+tag) || strings.HasSuffix(upper, "_"+tag) || strings.HasSuffix(upper, "."+tag) || upper == tag {
			return tag
		}
	}
	return base
}

func selectFiles(files []string, ggufFile, quant string, allQuants bool) []string {
	var nonMMProj []string
	for _, f := range files {
		if !strings.HasPrefix(strings.ToLower(filepath.Base(f)), "mmproj") {
			nonMMProj = append(nonMMProj, f)
		}
	}
	if allQuants {
		slices.Sort(nonMMProj)
		return nonMMProj
	}
	if ggufFile != "" {
		return []string{ggufFile}
	}
	if quant != "" {
		tag := strings.ToUpper(strings.TrimSuffix(strings.TrimSpace(quant), ".gguf"))
		for _, f := range nonMMProj {
			if strings.EqualFold(variantNameFromFile(f), tag) {
				return []string{f}
			}
		}
		klog.Fatalf("No .gguf file matching quant %q (available: %v)", quant, nonMMProj)
	}
	return []string{gguf.SelectGGUFFile(nonMMProj)}
}

func embedWithLlama(llamaModel llama.Model, pooling llama.PoolingType, normalize bool, tokenIDs [][]llama.Token) ([][]float32, int) {
	nEmbd := int(llama.ModelNEmbd(llamaModel))
	hasEncoder := llama.ModelHasEncoder(llamaModel)
	hasDecoder := llama.ModelHasDecoder(llamaModel)

	maxLen := 512
	for _, toks := range tokenIDs {
		if len(toks) > maxLen {
			maxLen = len(toks)
		}
	}

	buildParams := func(p llama.PoolingType) llama.ContextParams {
		cp := llama.ContextDefaultParams()
		cp.NCtx = uint32(maxLen)
		cp.NBatch = uint32(maxLen)
		cp.NUbatch = uint32(maxLen)
		cp.NSeqMax = 1
		cp.PoolingType = p
		cp.Embeddings = 1
		if *flagThreads > 0 {
			cp.NThreads = int32(*flagThreads)
			cp.NThreadsBatch = int32(*flagThreads)
		}
		return cp
	}

	lctx, err := llama.InitFromModel(llamaModel, buildParams(pooling))
	if err != nil || lctx == 0 {
		klog.Fatalf("llama.InitFromModel failed: %v", err)
	}
	if llama.GetPoolingType(lctx) == llama.PoolingTypeNone {
		llama.Free(lctx)
		fallback := llama.PoolingTypeMean
		if hasDecoder && !hasEncoder {
			fallback = llama.PoolingTypeLast
		}
		lctx, err = llama.InitFromModel(llamaModel, buildParams(fallback))
		if err != nil || lctx == 0 {
			klog.Fatalf("llama.InitFromModel (fallback pooling) failed: %v", err)
		}
	}
	defer llama.Free(lctx)

	useEncode := (hasEncoder && !hasDecoder) || !llama.GetCausalAttn(lctx)
	mem, _ := llama.GetMemory(lctx)
	batch := llama.BatchInit(int32(maxLen), 0, 1)
	defer llama.BatchFree(batch)

	results := make([][]float32, len(tokenIDs))
	seqZero := []llama.SeqId{0}
	for i, toks := range tokenIDs {
		if mem != 0 {
			_ = llama.MemoryClear(mem, false)
		}
		_ = batch.Clear()
		for pos, tok := range toks {
			if err := batch.Add(tok, llama.Pos(pos), seqZero, true); err != nil {
				klog.Fatalf("batch.Add failed: %v", err)
			}
		}
		var ret int32
		if useEncode {
			ret, err = llama.Encode(lctx, batch)
		} else {
			ret, err = llama.Decode(lctx, batch)
		}
		if err != nil || ret != 0 {
			klog.Fatalf("llama forward failed (prompt %d): ret=%d err=%v", i, ret, err)
		}
		vec, err := llama.GetEmbeddingsSeq(lctx, 0, int32(nEmbd))
		if err != nil || len(vec) == 0 {
			klog.Fatalf("GetEmbeddingsSeq failed (prompt %d): %v", i, err)
		}
		out := make([]float32, nEmbd)
		copy(out, vec)
		if normalize {
			var normSq float64
			for _, v := range out {
				normSq += float64(v) * float64(v)
			}
			norm := float32(math.Sqrt(normSq))
			if norm > 1e-12 {
				for j := range out {
					out[j] /= norm
				}
			}
		}
		results[i] = out
	}
	return results, nEmbd
}

func extractLlamaTokenizerSpec(vocab llama.Vocab, texts []string) *tokenizerSpec {
	specials := make(map[string]int)
	if id := int(llama.VocabPAD(vocab)); id >= 0 {
		specials["pad"] = id
	}
	if id := int(llama.VocabBOS(vocab)); id >= 0 {
		specials["bos"] = id
	}
	if id := int(llama.VocabEOS(vocab)); id >= 0 {
		specials["eos"] = id
	}
	if id := int(llama.VocabMASK(vocab)); id >= 0 {
		specials["mask"] = id
	}

	seen := make(map[string]bool)
	var cases []tokenizerCase
	for _, text := range texts {
		if seen[text] {
			continue
		}
		seen[text] = true
		toks := llama.Tokenize(vocab, text, true, true)
		ids := make([]int, len(toks))
		for i, t := range toks {
			ids[i] = int(t)
		}
		dec := llama.Detokenize(vocab, toks, false, true)
		decSkip := llama.Detokenize(vocab, toks, true, false)
		cases = append(cases, tokenizerCase{
			Text:               text,
			TokenIDs:           ids,
			Decoded:            dec,
			DecodedSkipSpecial: decSkip,
			// llama.cpp Detokenize concatenates special tokens without spaces (e.g. "china?[SEP]"),
			// which differs from HuggingFace WordPiece Decode ("[CLS] ... china? [SEP]").
			SkipDecodeCheck: true,
		})
	}
	return &tokenizerSpec{
		SpecialTokens: specials,
		Cases:         cases,
	}
}

func writeF32(path string, vecs [][]float32) error {
	total := 0
	for _, v := range vecs {
		total += len(v)
	}
	buf := make([]byte, total*4)
	offset := 0
	for _, v := range vecs {
		for _, f := range v {
			binary.LittleEndian.PutUint32(buf[offset:offset+4], math.Float32bits(f))
			offset += 4
		}
	}
	return os.WriteFile(path, buf, 0644)
}

func similarityMatrix(vecs [][]float32) [][]float32 {
	n := len(vecs)
	mat := make([][]float32, n)
	for i := range n {
		mat[i] = make([]float32, n)
		for j := range n {
			var dot, na, nb float64
			for k := range vecs[i] {
				a, b := float64(vecs[i][k]), float64(vecs[j][k])
				dot += a * b
				na += a * a
				nb += b * b
			}
			sim := 0.0
			if denom := math.Sqrt(na) * math.Sqrt(nb); denom > 0 {
				sim = dot / denom
			}
			mat[i][j] = float32(math.Round(sim*1e6) / 1e6)
		}
	}
	return mat
}

func boolPtr(b bool) *bool { return &b }

func main() {
	klog.InitFlags(nil)
	flag.Parse()
	if *flagModel == "" || *flagOutDir == "" {
		klog.Fatalf("-model and -out are required")
	}

	if err := os.MkdirAll(*flagOutDir, 0755); err != nil {
		klog.Fatalf("Failed to create output dir %q: %v", *flagOutDir, err)
	}

	refBytes, err := os.ReadFile(*flagRefSentences)
	if err != nil {
		klog.Fatalf("Failed to read %q: %v", *flagRefSentences, err)
	}
	var shared sharedSentences
	if err := json.Unmarshal(refBytes, &shared); err != nil {
		klog.Fatalf("Failed to parse %q: %v", *flagRefSentences, err)
	}

	var sentences []string
	for _, q := range shared.Queries {
		sentences = append(sentences, *flagQueryPrefix+q)
	}
	for _, d := range shared.Documents {
		sentences = append(sentences, *flagDocPrefix+d)
	}
	tokTexts := append(append([]string(nil), sentences...), shared.TokenizerEdgeCases...)

	repo := hub.New(*flagModel)
	if err := repo.DownloadInfo(false); err != nil {
		klog.Fatalf("Failed to load repo info for %q: %v", *flagModel, err)
	}
	revision := ""
	if info := repo.Info(); info != nil {
		revision = info.CommitHash
	}

	hfModel, err := transformer.LoadModel(repo)
	if err != nil {
		klog.Fatalf("transformer.LoadModel(%q) failed: %v", *flagModel, err)
	}
	if !hfModel.IsGGUF() {
		klog.Fatalf("Repository %q does not contain GGUF files", *flagModel)
	}

	selectedFiles := selectFiles(hfModel.GGUFFiles(), *flagGGUFFile, *flagQuant, *flagAllQuants)
	fmt.Fprintf(os.Stderr, "Selected %d GGUF file(s) from %s: %v\n", len(selectedFiles), *flagModel, selectedFiles)

	ensureLlamaLoaded(*flagLibDir, *flagProcessor, *flagVerbose)
	defer llama.Close()

	var tokSpec *tokenizerSpec
	var variants []variantSpec
	var causalMask bool

	for idx, ggufFile := range selectedFiles {
		vName := variantNameFromFile(ggufFile)
		fmt.Fprintf(os.Stderr, "[%d/%d] Processing %s (variant %s)...\n", idx+1, len(selectedFiles), ggufFile, vName)

		hfModel.WithGGUFFile(ggufFile)
		pooling, normalize := poolingFromHFModel(hfModel)
		if idx == 0 {
			causalMask = hfModel.UseCausalMask()
		}

		localPath, err := repo.DownloadFile(ggufFile)
		if err != nil {
			klog.Fatalf("DownloadFile(%q) failed: %v", ggufFile, err)
		}

		mparams := llama.ModelDefaultParams()
		mparams.NGpuLayers = int32(*flagNGL)
		lm, err := llama.ModelLoadFromFile(localPath, mparams)
		if err != nil || lm == 0 {
			klog.Fatalf("ModelLoadFromFile(%q) failed: %v", localPath, err)
		}

		vocab := llama.ModelGetVocab(lm)
		if tokSpec == nil {
			tokSpec = extractLlamaTokenizerSpec(vocab, tokTexts)
		}

		promptToks := make([][]llama.Token, len(sentences))
		for i, s := range sentences {
			promptToks[i] = llama.Tokenize(vocab, s, true, true)
		}

		embs, nEmbd := embedWithLlama(lm, pooling, normalize, promptToks)
		llama.ModelFree(lm)

		embFile := fmt.Sprintf("embeddings_%s.f32", vName)
		if err := writeF32(filepath.Join(*flagOutDir, embFile), embs); err != nil {
			klog.Fatalf("Failed writing %q: %v", embFile, err)
		}

		variants = append(variants, variantSpec{
			Name:     vName,
			GGUFFile: ggufFile,
			Embedding: &embeddingSpec{
				UseSharedSentences: true,
				QueryPrefix:        *flagQueryPrefix,
				DocumentPrefix:     *flagDocPrefix,
				EmbeddingDim:       nEmbd,
				EmbeddingsFile:     embFile,
				SimilarityMatrix:   similarityMatrix(embs),
				Tolerances: embeddingTolerances{
					MinCosineSim:       *flagMinCosine,
					MaxAbsDiff:         *flagMaxAbsDiff,
					MaxMeanAbsDiff:     *flagMaxMeanDiff,
					SimilarityMaxDelta: 0.01,
				},
			},
		})
	}

	modelName := *flagName
	if modelName == "" {
		modelName = filepath.Base(*flagOutDir)
	}

	spec := modelSpec{
		Name:     modelName,
		Family:   "transformer",
		Format:   "gguf",
		RepoID:   *flagModel,
		Revision: revision,
		SizeTier: *flagTier,
		Tags:     []string{"gguf", "embed"},
		ModelOptions: modelOptions{
			CausalMask:      boolPtr(causalMask),
			ApproximateGelu: boolPtr(true), // llama.cpp uses tanh-approximate GELU
		},
		Provenance: provenance{
			Generator:      "modeltests/internal/generators/yzma",
			Command:        strings.Join(os.Args, " "),
			RuntimeVersion: fmt.Sprintf("yzma/%s (llama.cpp %s)", "v1.32.0", download.DefaultVersion),
			DType:          "fp32",
			Device:         *flagProcessor,
			GeneratedAt:    time.Now().Format("2006-01-02"),
		},
		Tokenizer: tokSpec,
		Variants:  variants,
	}

	specBytes, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		klog.Fatalf("json.MarshalIndent failed: %v", err)
	}
	specBytes = append(specBytes, '\n')
	specPath := filepath.Join(*flagOutDir, "spec.json")
	if err := os.WriteFile(specPath, specBytes, 0644); err != nil {
		klog.Fatalf("Failed writing %q: %v", specPath, err)
	}
	fmt.Fprintf(os.Stderr, "Wrote %s (%d variants)\n", specPath, len(variants))
}
