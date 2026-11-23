package main

import (
	"flag"
	"fmt"
	"os"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/support"
	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/datasets"
	"github.com/gomlx/go-huggingface/examples/dolly"
	"github.com/gomlx/go-huggingface/hub"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/go-huggingface/tokenizers"
	tapi "github.com/gomlx/go-huggingface/tokenizers/api"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/attention/kvcache"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/zoo/transformer/generate"
	"k8s.io/klog/v2"
)

var (
	flagNum          = flag.Int("n", 3, "Number of examples to print from the dataset.")
	flagModel        = flag.String("model", "", "HuggingFace model repository to use for generation (e.g. \"google/gemma-4-E2B-it\"). If empty, only browses the dataset.")
	flagPromptTokens = flag.Int("prompt_tokens", 1024, "Maximum number of tokens used for prompt+context.")
	flagGenTokens    = flag.Int("gen_tokens", 128, "Maximum number of tokens to generate.")
	flagDonateStore  = flag.Bool("donate_store", true, "Donate model weights to shared backend constants for pre-packing without duplicating memory.")
	flagCPUProf      = flag.String("cpuprof", "", "If set, writes a CPU profile of the generation loop to this file.")
)

type chatTemplate struct {
	prefix    string
	suffix    string
	stopToken string
}

func detectChatTemplate(tok tokenizers.Tokenizer) chatTemplate {
	ttid, ok := tok.(interface {
		TokenToID(token string) (int, bool)
	})
	if !ok {
		return chatTemplate{}
	}
	if _, hasStart := ttid.TokenToID("<|turn>"); hasStart {
		if _, hasEnd := ttid.TokenToID("<turn|>"); hasEnd {
			return chatTemplate{
				prefix:    "<bos><|turn>user\n",
				suffix:    "<turn|>\n<|turn>model\n",
				stopToken: "<turn|>",
			}
		}
	}
	if _, hasStart := ttid.TokenToID("<start_of_turn>"); hasStart {
		if _, hasEnd := ttid.TokenToID("<end_of_turn>"); hasEnd {
			return chatTemplate{
				prefix:    "<bos><start_of_turn>user\n",
				suffix:    "<end_of_turn>\n<start_of_turn>model\n",
				stopToken: "<end_of_turn>",
			}
		}
	}
	if _, hasStart := ttid.TokenToID("<|im_start|>"); hasStart {
		if _, hasEnd := ttid.TokenToID("<|im_end|>"); hasEnd {
			return chatTemplate{
				prefix:    "<|im_start|>user\n",
				suffix:    "<|im_end|>\n<|im_start|>assistant\n",
				stopToken: "<|im_end|>",
			}
		}
	}
	return chatTemplate{}
}

type generatorRunner struct {
	backend    compute.Backend
	store      *model.Store
	tokenizer  tokenizers.Tokenizer
	decoder    *generate.Generator
	template   chatTemplate
	prefixIDs  []int
	suffixIDs  []int
	stopTokens map[int]bool
}

func newGeneratorRunner(modelRepo string, maxPromptTokens, maxGenTokens int) (*generatorRunner, error) {
	backend, err := compute.New()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize backend: %w", err)
	}

	fmt.Printf("- Backend: %s\n", backend.Description())
	fmt.Printf("- Loading model repository %q... ", modelRepo)
	start := time.Now()
	repo := hub.New(modelRepo)
	if err := repo.DownloadInfo(false); err != nil {
		backend.Finalize()
		return nil, fmt.Errorf("failed to load repo %q: %w", modelRepo, err)
	}
	fmt.Printf("done (%s)\n", humanize.Duration(time.Since(start)))

	fmt.Printf("- Loading model configurations... ")
	start = time.Now()
	hfModel, err := transformer.LoadModel(repo)
	if err != nil {
		backend.Finalize()
		return nil, fmt.Errorf("failed to load model config: %w", err)
	}
	hfModel.WithCausalMask(true)
	fmt.Printf("done (%s)\n", humanize.Duration(time.Since(start)))

	descLines := strings.Split(hfModel.Description(), "\n")
	if len(descLines) > 2 {
		descLines = descLines[1 : len(descLines)-1]
	}
	fmt.Printf("- Model description:\n\t%s\n", strings.Join(descLines, "\n\t"))

	store := model.NewStore()
	store.SetParams(map[string]any{
		layers.ParamQuantizedMatmul: true,
	})
	fmt.Printf("- Loading model weights into store... ")
	start = time.Now()
	if err := hfModel.LoadStore(backend, store); err != nil {
		backend.Finalize()
		return nil, fmt.Errorf("failed to load weights into store: %w", err)
	}
	fmt.Printf("done (%s)\n", humanize.Duration(time.Since(start)))

	tokenizer, err := hfModel.GetTokenizer()
	if err != nil {
		backend.Finalize()
		return nil, fmt.Errorf("failed to get tokenizer: %w", err)
	}

	kvCacheConfig := hfModel.KVCacheConfig()
	tm := hfModel.CreateGoMLXModel(store.RootScope())
	tm.PopulateKVCacheConfigs()

	var incrementalModelFn generate.KVCacheModelFn = func(scope *model.Scope, newTokens *Node, position *Node, cache kvcache.KVCacheNodes) (*Node, kvcache.KVCacheNodes) {
		return hfModel.Forward(scope, newTokens, position, nil, nil, cache)
	}

	maxTotalLen := maxPromptTokens + maxGenTokens
	decoder := generate.New(incrementalModelFn).
		WithKVCache(kvCacheConfig, tm.NumKVHeads, tm.HeadDim, tm.DType).
		WithMaxLength(maxTotalLen).
		WithDonateStore(*flagDonateStore)
	decoder.Temperature = 0
	decoder.TopK = 0
	decoder.TopP = 0

	stopTokens := make(map[int]bool)
	if eosTokenID, err := tokenizer.SpecialTokenID(tapi.TokEndOfSentence); err == nil {
		decoder.WithEOS(eosTokenID)
		stopTokens[eosTokenID] = true
	}

	tmpl := detectChatTemplate(tokenizer)
	if tmpl.stopToken != "" {
		if ttid, ok := tokenizer.(interface {
			TokenToID(token string) (int, bool)
		}); ok {
			if stopID, ok := ttid.TokenToID(tmpl.stopToken); ok {
				decoder.WithStopTokens(stopID)
				stopTokens[stopID] = true
			}
		}
	}

	var prefixIDs, suffixIDs []int
	if tmpl.prefix != "" {
		prefixIDs = tokenizer.Encode(tmpl.prefix)
	}
	if tmpl.suffix != "" {
		suffixIDs = tokenizer.Encode(tmpl.suffix)
	}

	fmt.Println()
	return &generatorRunner{
		backend:    backend,
		store:      store,
		tokenizer:  tokenizer,
		decoder:    decoder,
		template:   tmpl,
		prefixIDs:  prefixIDs,
		suffixIDs:  suffixIDs,
		stopTokens: stopTokens,
	}, nil
}

func (r *generatorRunner) close() {
	if r.backend != nil {
		r.backend.Finalize()
	}
}

func (r *generatorRunner) encodePrompt(record dolly.Record, maxPromptTokens int) []int {
	var sb strings.Builder
	sb.WriteString(strings.TrimSpace(record.Instruction))
	if ctx := strings.TrimSpace(record.Context); ctx != "" {
		sb.WriteString("\n\nContext:\n")
		sb.WriteString(ctx)
	}
	bodyIDs := r.tokenizer.Encode(sb.String())

	if maxPromptTokens > 0 {
		overhead := len(r.prefixIDs) + len(r.suffixIDs)
		maxBody := max(maxPromptTokens-overhead, 1)
		if len(bodyIDs) > maxBody {
			bodyIDs = bodyIDs[:maxBody]
		}
	}

	promptIDs := make([]int, 0, len(r.prefixIDs)+len(bodyIDs)+len(r.suffixIDs))
	promptIDs = append(promptIDs, r.prefixIDs...)
	promptIDs = append(promptIDs, bodyIDs...)
	promptIDs = append(promptIDs, r.suffixIDs...)
	return promptIDs
}

type genResult struct {
	text          string
	promptTokens  int
	genTokens     int
	ttft          time.Duration
	decodeTime    time.Duration
	decodeCPUTime time.Duration
	totalTime     time.Duration
}

func (r *generatorRunner) generate(record dolly.Record, maxPromptTokens, maxGenTokens int) (genResult, error) {
	promptIDs := r.encodePrompt(record, maxPromptTokens)
	var allGenIDs []int
	var displayIDs []int

	start := time.Now()
	var firstTokenAt time.Time
	var firstTokenCPU time.Duration
	err := r.decoder.GenerateStreaming(r.backend, r.store.RootScope(), promptIDs, func(token int) bool {
		if len(allGenIDs) == 0 {
			firstTokenAt = time.Now()
			firstTokenCPU = support.CPUTime()
		}
		allGenIDs = append(allGenIDs, token)
		if !r.stopTokens[token] {
			displayIDs = append(displayIDs, token)
		}
		if maxGenTokens > 0 && len(allGenIDs) >= maxGenTokens {
			return false
		}
		return true
	})
	endCPU := support.CPUTime()
	totalTime := time.Since(start)
	if err != nil {
		return genResult{}, err
	}

	var ttft, decodeTime, decodeCPUTime time.Duration
	if !firstTokenAt.IsZero() {
		ttft = firstTokenAt.Sub(start)
		decodeTime = totalTime - ttft
		decodeCPUTime = endCPU - firstTokenCPU
	} else {
		ttft = totalTime
	}

	return genResult{
		text:          strings.TrimSpace(r.tokenizer.Decode(displayIDs)),
		promptTokens:  len(promptIDs),
		genTokens:     len(allGenIDs),
		ttft:          ttft,
		decodeTime:    decodeTime,
		decodeCPUTime: decodeCPUTime,
		totalTime:     totalTime,
	}, nil
}

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	var runner *generatorRunner
	if *flagModel != "" {
		if *flagPromptTokens <= 0 {
			klog.Fatalf("-prompt_tokens must be > 0, got %d", *flagPromptTokens)
		}
		if *flagGenTokens <= 0 {
			klog.Fatalf("-gen_tokens must be > 0, got %d", *flagGenTokens)
		}
		var err error
		runner, err = newGeneratorRunner(*flagModel, *flagPromptTokens, *flagGenTokens)
		if err != nil {
			fmt.Fprintf(os.Stderr, "\n%v\n", err)
			os.Exit(1)
		}
		defer runner.close()
	}

	ds := datasets.New(dolly.ID)
	count := 0
	var totalPromptTokens, totalGenTokens, totalDecodeTokens int
	var totalTTFT, totalDecodeTime, totalDecodeCPUTime, totalTime time.Duration

	if *flagCPUProf != "" {
		f, err := os.Create(*flagCPUProf)
		if err != nil {
			klog.Fatalf("Could not create CPU profile %q: %v", *flagCPUProf, err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			klog.Fatalf("Could not start CPU profile: %v", err)
		}
		defer pprof.StopCPUProfile()
	}

	for record, err := range datasets.IterParquetFromDataset[dolly.Record](ds, dolly.Config, dolly.TrainSplit) {
		if err != nil {
			klog.Fatalf("Failed to read record: %+v", err)
		}
		if *flagNum > 0 && count >= *flagNum {
			break
		}
		count++
		fmt.Printf("=== Example #%d [%s] ===\n", count, record.Category)
		fmt.Printf("Instruction:\n%s\n", strings.TrimSpace(record.Instruction))
		if ctx := strings.TrimSpace(record.Context); ctx != "" {
			fmt.Printf("\nContext:\n%s\n", ctx)
		}
		fmt.Printf("\nReference Response:\n%s\n", strings.TrimSpace(record.Response))

		if runner != nil {
			res, err := runner.generate(record, *flagPromptTokens, *flagGenTokens)
			if err != nil {
				klog.Fatalf("Failed to generate for example #%d: %+v", count, err)
			}
			decodeToks := max(res.genTokens-1, 0)
			totalPromptTokens += res.promptTokens
			totalGenTokens += res.genTokens
			totalDecodeTokens += decodeToks
			totalTTFT += res.ttft
			totalDecodeTime += res.decodeTime
			totalDecodeCPUTime += res.decodeCPUTime
			totalTime += res.totalTime

			var prefillTPS, decodeTPS, cpuCores float64
			var cpuPerTok time.Duration
			if res.ttft.Seconds() > 0 {
				prefillTPS = float64(res.promptTokens) / res.ttft.Seconds()
			}
			if decodeToks > 0 {
				if res.decodeTime.Seconds() > 0 {
					decodeTPS = float64(decodeToks) / res.decodeTime.Seconds()
					cpuCores = res.decodeCPUTime.Seconds() / res.decodeTime.Seconds()
				}
				cpuPerTok = res.decodeCPUTime / time.Duration(decodeToks)
			}
			fmt.Printf("\nGenerated Response:\n%s\n", res.text)
			fmt.Printf("\n[Stats: prompt=%d tok, gen=%d tok, TTFT=%s (%.2f tok/s), decode=%s (%.2f tok/s, CPUTime/tok=%s, avg_cores=%.2f), total=%s, MaxRSS=%s]\n",
				res.promptTokens, res.genTokens,
				humanize.Duration(res.ttft), prefillTPS,
				humanize.Duration(res.decodeTime), decodeTPS,
				humanize.Duration(cpuPerTok), cpuCores,
				humanize.Duration(res.totalTime),
				humanize.Bytes(support.MaxRSS()))
		}
		fmt.Println()
	}

	if runner != nil {
		if count > 1 {
			var avgPrefillTPS, avgDecodeTPS, avgCPUCores float64
			var avgCPUPerTok time.Duration
			if totalTTFT.Seconds() > 0 {
				avgPrefillTPS = float64(totalPromptTokens) / totalTTFT.Seconds()
			}
			if totalDecodeTokens > 0 {
				if totalDecodeTime.Seconds() > 0 {
					avgDecodeTPS = float64(totalDecodeTokens) / totalDecodeTime.Seconds()
					avgCPUCores = totalDecodeCPUTime.Seconds() / totalDecodeTime.Seconds()
				}
				avgCPUPerTok = totalDecodeCPUTime / time.Duration(totalDecodeTokens)
			}
			fmt.Printf("=== Summary (%d examples) ===\n", count)
			fmt.Printf("Prompt tokens: %d, Generated tokens: %d\n", totalPromptTokens, totalGenTokens)
			fmt.Printf("Prefill: %s (%.2f tok/s), Decode: %s (%.2f tok/s, CPUTime/tok=%s, avg_cores=%.2f), Total: %s\n",
				humanize.Duration(totalTTFT), avgPrefillTPS,
				humanize.Duration(totalDecodeTime), avgDecodeTPS,
				humanize.Duration(avgCPUPerTok), avgCPUCores,
				humanize.Duration(totalTime))
		}
		fmt.Printf("MaxRSS: %s\n", humanize.Bytes(support.MaxRSS()))
	}
}
