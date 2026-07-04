package main

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/support/humanize"
	gemma4e4bit "github.com/gomlx/go-huggingface/examples/gemma4-e4bit"
	"github.com/gomlx/go-huggingface/models/transformer"
	"github.com/gomlx/go-huggingface/tokenizers/api"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/attention/kvcache"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/zoo/transformer/generate"
)

var (
	flagPrompt      = flag.String("prompt", "Write a short poem about antigravity.", "Prompt to generate text from.")
	flagRaw         = flag.Bool("raw", false, "If set, do not automatically wrap prompt in Gemma-4 turn templates.")
	flagMaxLength   = flag.Int("max_len", 100, "Maximum sequence length to generate.")
	flagTemperature = flag.Float64("temperature", 0.0, "Temperature (0.0 for greedy).")
	flagTopK        = flag.Int("top_k", 0, "Top K selection.")
	flagTopP        = flag.Float64("top_p", 0.0, "Top P selection.")
	flagRepeat      = flag.Int("repeat", 1, "Number of times to repeat the generation (for benchmarking).")
	flagQuiet       = flag.Bool("quiet", false, "If set, disables all informational printing and only outputs the final generated text.")
)

func main() {
	flag.Parse()

	// 1. Initialize Backend
	backend, err := compute.New()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to initialize backend: %+v\n", err)
		os.Exit(1)
	}
	defer backend.Finalize()

	// 2. Load Model Repo info
	if !*flagQuiet {
		fmt.Printf("- Loading model repository %s... ", gemma4e4bit.Repository)
	}
	start := time.Now()
	repo, err := gemma4e4bit.LoadRepo()
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nFailed to load repo: %+v\n", err)
		os.Exit(1)
	}
	if !*flagQuiet {
		fmt.Printf("done (%s)\n", humanize.Duration(time.Since(start)))
	}

	// 3. Load Model Config
	if !*flagQuiet {
		fmt.Printf("- Loading model configurations... ")
	}
	start = time.Now()
	hfModel, err := transformer.LoadModel(repo)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nFailed to load model: %+v\n", err)
		os.Exit(1)
	}
	if !*flagQuiet {
		fmt.Printf("done (%s)\n", humanize.Duration(time.Since(start)))
	}

	// We are using a text generation model, so ensure we use causal mask
	hfModel.WithCausalMask(true)

	// Expose KVCache configuration so we can print or check it
	kvCacheConfig := hfModel.KVCacheConfig()
	if !*flagQuiet {
		descLines := strings.Split(hfModel.Description(), "\n")
		descLines = descLines[1 : len(descLines)-1]
		fmt.Printf("- Model description:\n\t%s\n", strings.Join(descLines, "\n\t"))
	}

	// 4. Load Weights into model store
	store := model.NewStore()
	if !*flagQuiet {
		fmt.Printf("- Loading model weights (safetensors) into store... ")
	}
	start = time.Now()
	err = hfModel.LoadStore(backend, store)
	if err != nil {
		fmt.Fprintf(os.Stderr, "\nFailed to load weights into store: %+v\n", err)
		os.Exit(1)
	}
	if !*flagQuiet {
		fmt.Printf("done (%s)\n", humanize.Duration(time.Since(start)))
	}

	// 5. Get tokenizer and encode prompt
	tokenizer, err := hfModel.GetTokenizer()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to get tokenizer: %+v\n", err)
		os.Exit(1)
	}

	prompt := *flagPrompt
	if !*flagRaw {
		prompt = "<bos><|turn>user\n" + prompt + "<turn|>\n<|turn>model\n"
	}
	promptIds := tokenizer.Encode(prompt)
	promptLen := len(promptIds)
	if !*flagQuiet {
		fmt.Printf("- Encoded prompt into %d tokens\n", promptLen)
	}

	// 6. Setup Decoder and Strategy

	// Retrieve vocabulary sizes and head details from Model Config
	runScope := store.RootScope()
	tm := hfModel.CreateGoMLXModel(runScope)
	tm.PopulateKVCacheConfigs()

	// Create decoder with IncrementalModelFn
	var incrementalModelFn generate.KVCacheModelFn = func(scope *model.Scope, newTokens *Node, position *Node, cache kvcache.KVCacheNodes) (*Node, kvcache.KVCacheNodes) {
		return hfModel.Forward(scope, newTokens, position, nil, nil, cache)
	}

	decoder := generate.New(incrementalModelFn).
		WithKVCache(kvCacheConfig, tm.NumKVHeads, tm.HeadDim, tm.DType).
		WithMaxLength(*flagMaxLength)

	if *flagTemperature != 0 {
		decoder.WithTemperature(float32(*flagTemperature))
	} else {
		decoder.Temperature = 0
	}
	if *flagTopK != 0 {
		decoder.WithTopK(*flagTopK)
	} else {
		decoder.TopK = 0
	}
	if *flagTopP != 0 {
		decoder.WithTopP(float32(*flagTopP))
	} else {
		decoder.TopP = 0
	}

	stopTokens := make(map[int]bool)
	if eosTokenId, err := tokenizer.SpecialTokenID(api.TokEndOfSentence); err == nil {
		decoder.WithEOS(eosTokenId)
		stopTokens[eosTokenId] = true
	}
	// Also stop early on "<turn|>" token if resolved
	if tok, ok := tokenizer.(interface {
		TokenToID(token string) (int, bool)
	}); ok {
		if turnTokenId, ok := tok.TokenToID("<turn|>"); ok {
			decoder.WithStopTokens(turnTokenId)
			stopTokens[turnTokenId] = true
		}
	}

	// 7. Run decoding
	repeats := max(*flagRepeat, 1)

	var outputTensor *tensors.Tensor
	var generationTime time.Duration
	for r := range repeats {
		if !*flagQuiet {
			if repeats > 1 {
				fmt.Printf("- Generating tokens (run %d/%d)... ", r+1, repeats)
			} else {
				fmt.Printf("- Generating tokens... ")
			}
		}
		start = time.Now()
		var err error
		outputTensor, err = decoder.Decode(backend, store.RootScope(), promptIds)
		if err != nil {
			fmt.Fprintf(os.Stderr, "\nFailed to generate tokens: %+v\n", err)
			os.Exit(1)
		}
		generationTime = time.Since(start)
		if !*flagQuiet {
			fmt.Printf("done (%s)\n", humanize.Duration(generationTime))

			if repeats > 1 {
				outputValues := outputTensor.Value().([][]int32)[0]
				generatedNewIds := outputValues[promptLen:]
				var tokensSec float64
				if generationTime.Seconds() > 0 {
					tokensSec = float64(len(generatedNewIds)) / generationTime.Seconds()
				}
				fmt.Printf("  Run %d: Generated %d new tokens in %v (%.2f tokens/s)\n",
					r+1,
					len(generatedNewIds),
					humanize.Duration(generationTime),
					tokensSec,
				)
			}
		}
	}

	// 8. Convert generated tokens to text and print
	outputValues := outputTensor.Value().([][]int32)[0]
	generatedPromptIds := outputValues[:promptLen]
	generatedNewIds := outputValues[promptLen:]

	promptInts := make([]int, len(generatedPromptIds))
	for i, v := range generatedPromptIds {
		promptInts[i] = int(v)
	}
	newInts := make([]int, len(generatedNewIds))
	for i, v := range generatedNewIds {
		newInts[i] = int(v)
	}

	// Slice off the last token if it matches a stop token (like EOS or "<turn|>")
	if len(newInts) > 0 {
		lastToken := newInts[len(newInts)-1]
		if stopTokens[lastToken] {
			newInts = newInts[:len(newInts)-1]
		}
	}

	if !*flagQuiet {
		fmt.Printf("\n--- Generation Result ---\n")
		fmt.Printf("Prompt IDs: %v\n", promptInts)
		fmt.Printf("Prompt: %s\n", tokenizer.Decode(promptInts))
		fmt.Printf("Generated IDs: %v\n", newInts)
		fmt.Printf("Generated text: %s\n", tokenizer.Decode(newInts))
		fmt.Printf("-------------------------\n")
		var tokensSec float64
		if generationTime.Seconds() > 0 {
			tokensSec = float64(len(generatedNewIds)) / generationTime.Seconds()
		}
		fmt.Printf("Generated %d new tokens in %v (%.2f tokens/s)\n",
			len(generatedNewIds),
			humanize.Duration(generationTime),
			tokensSec,
		)
	} else {
		fmt.Println(tokenizer.Decode(newInts))
	}
}
