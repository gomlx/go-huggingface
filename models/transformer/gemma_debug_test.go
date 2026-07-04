package transformer_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	gemma4e4bit "github.com/gomlx/go-huggingface/examples/gemma4-e4bit"
	"github.com/gomlx/go-huggingface/models/transformer"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	mltransformer "github.com/gomlx/gomlx/ml/zoo/transformer"
)

func TestGemma(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping Gemma tests in short mode")
	}

	t.Run("Debug", func(t *testing.T) {
		backend, err := compute.New()
		if err != nil {
			t.Fatalf("Failed to initialize backend: %v", err)
		}
		defer backend.Finalize()

		repo, err := gemma4e4bit.LoadRepo()
		if err != nil {
			t.Fatalf("Failed to load repo: %v", err)
		}

		hfModel, err := transformer.LoadModel(repo)
		if err != nil {
			t.Fatalf("Failed to load model: %v", err)
		}
		hfModel.WithCausalMask(true)

		store := model.NewStore()
		err = hfModel.LoadStore(backend, store)
		if err != nil {
			t.Fatalf("Failed to load store: %v", err)
		}

		fmt.Println("=== LAYER 0 VARIABLES ===")
		for variable := range store.IterVariables() {
			path := variable.Path()
			if strings.Contains(path, "layer_0") || strings.Contains(path, "token_embed") || strings.Contains(path, "final_norm") {
				fmt.Printf("VARIABLE: %s, shape=%s\n", path, variable.Shape())
			}
		}
		fmt.Println("=========================")

		tokenizer, err := hfModel.GetTokenizer()
		if err != nil {
			t.Fatalf("Failed to get tokenizer: %v", err)
		}

		promptIds := tokenizer.Encode("Antigravity is")
		fmt.Printf("Go Token IDs: %v\n", promptIds)

		// Build inputs
		inputIds := make([]int32, len(promptIds))
		for i, v := range promptIds {
			inputIds[i] = int32(v)
		}

		exec, err := model.NewExec(backend, store, func(scope *model.Scope, tokens *graph.Node) []*graph.Node {
			tm := hfModel.CreateGoMLXModel(scope)
			lastLayer, allLayers, _ := tm.AllLayers(tokens, mltransformer.CallOptions{})
			logits := tm.LogitsFromEmbeddings(lastLayer)
			return []*graph.Node{
				graph.ConvertDType(allLayers[0], dtypes.Float32),
				graph.ConvertDType(allLayers[1], dtypes.Float32),
				graph.ConvertDType(allLayers[2], dtypes.Float32),
				graph.ConvertDType(logits, dtypes.Float32),
			}
		})
		if err != nil {
			t.Fatalf("Failed to create execution: %v", err)
		}

		inputTensor := tensors.FromValue([][]int32{inputIds})
		outputs, err := exec.Exec(inputTensor)
		if err != nil {
			t.Fatalf("Failed to execute: %v", err)
		}

		embeddingsTensor := outputs[0]
		layer0Tensor := outputs[1]
		layer1Tensor := outputs[2]
		logitsTensor := outputs[3]

		fmt.Printf("Go Embeddings (allLayers[0])[0, 0, :5]: %v\n", embeddingsTensor.Value().([][][]float32)[0][0][:5])
		fmt.Printf("Go Layer 0 Output (allLayers[1])[0, 0, :5]: %v\n", layer0Tensor.Value().([][][]float32)[0][0][:5])
		fmt.Printf("Go Layer 1 Output (allLayers[2])[0, 0, :5]: %v\n", layer1Tensor.Value().([][][]float32)[0][0][:5])

		logitsVal := logitsTensor.Value().([][][]float32)
		lastTokenLogits := logitsVal[0][3]
		fmt.Printf("Go Logits [0, -1, :5]: %v\n", lastTokenLogits[:5])

		// Find top 5 values and indices
		type entry struct {
			index int
			val   float32
		}
		top := make([]entry, 0, 5)
		for idx, val := range lastTokenLogits {
			if len(top) < 5 {
				top = append(top, entry{idx, val})
				// Sort ascending by value
				for i := len(top) - 1; i > 0; i-- {
					if top[i].val > top[i-1].val {
						top[i], top[i-1] = top[i-1], top[i]
					}
				}
			} else if val > top[4].val {
				top[4] = entry{idx, val}
				for i := 4; i > 0; i-- {
					if top[i].val > top[i-1].val {
						top[i], top[i-1] = top[i-1], top[i]
					}
				}
			}
		}

		fmt.Printf("Go top 5 token IDs: ")
		for _, e := range top {
			fmt.Printf("%d ", e.index)
		}
		fmt.Println()

		fmt.Printf("Go top 5 values: ")
		for _, e := range top {
			fmt.Printf("%f ", e.val)
		}
		fmt.Println()

		fmt.Printf("Go top 5 tokens decoded: ")
		for _, e := range top {
			fmt.Printf("%q ", tokenizer.Decode([]int{e.index}))
		}
		fmt.Println()
	})

	t.Run("TokenizerNewlines", func(t *testing.T) {
		repo, err := gemma4e4bit.LoadRepo()
		if err != nil {
			t.Fatalf("Failed to load repo: %v", err)
		}

		hfModel, err := transformer.LoadModel(repo)
		if err != nil {
			t.Fatalf("Failed to load model: %v", err)
		}

		tokenizer, err := hfModel.GetTokenizer()
		if err != nil {
			t.Fatalf("Failed to get tokenizer: %v", err)
		}

		prompt := "<bos><|turn>user\nWrite a short poem about antigravity.<turn|>\n<|turn>model\n"
		promptIds := tokenizer.Encode(prompt)
		fmt.Printf("Go Prompt IDs with newlines: %v\n", promptIds)
	})

	t.Run("IncrementalDebug", func(t *testing.T) {
		backend, err := compute.New()
		if err != nil {
			t.Fatalf("Failed to initialize backend: %v", err)
		}
		defer backend.Finalize()

		repo, err := gemma4e4bit.LoadRepo()
		if err != nil {
			t.Fatalf("Failed to load repo: %v", err)
		}

		hfModel, err := transformer.LoadModel(repo)
		if err != nil {
			t.Fatalf("Failed to load model: %v", err)
		}
		hfModel.WithCausalMask(true)

		store := model.NewStore()
		err = hfModel.LoadStore(backend, store)
		if err != nil {
			t.Fatalf("Failed to load store: %v", err)
		}

		tokenizer, err := hfModel.GetTokenizer()
		if err != nil {
			t.Fatalf("Failed to get tokenizer: %v", err)
		}

		prompt := "<bos><|turn>user\nWrite a short poem about antigravity.<turn|>\n<|turn>model\n"
		promptIds := tokenizer.Encode(prompt)
		fmt.Printf("Prompt IDs (%d tokens): %v\n", len(promptIds), promptIds)

		// We'll run prompt step
		runScope := store.RootScope()
		tm := hfModel.CreateGoMLXModel(runScope)
		tm.PopulateKVCacheConfigs()

		// Step 1: Prompt Execution
		promptIds32 := make([]int32, len(promptIds))
		for i, v := range promptIds {
			promptIds32[i] = int32(v)
		}
		promptTensor := tensors.FromValue([][]int32{promptIds32})

		batchSize := 1
		promptLen := len(promptIds)

		kvCacheTensors := tm.KVCache.InitializeTensors(batchSize, tm.NumKVHeads, tm.HeadDim, tm.DType, promptLen)
		kvCacheTensorsSerialized, err := tm.KVCache.SerializeTensors(kvCacheTensors)
		if err != nil {
			t.Fatal(err)
		}

		// Exec for prompt
		promptExec, err := model.NewExec(backend, store, func(scope *model.Scope, inputs []*graph.Node) []*graph.Node {
			tokens := inputs[0]
			cacheNodes := inputs[1:]
			cache := tm.KVCache.DeserializeNodes(cacheNodes)
			g := tokens.Graph()
			positionNode := graph.Const(g, int32(0))
			logits, updatedCache := hfModel.Forward(scope, tokens, positionNode, nil, nil, cache)
			serializedUpdatedCache, err := tm.KVCache.SerializeNodes(updatedCache)
			if err != nil {
				panic(err)
			}
			// Squeeze logits seq dimension at last index
			lastLogits := graph.Slice(logits, graph.AxisRange(), graph.AxisElem(-1), graph.AxisRange())
			lastLogits = graph.Squeeze(lastLogits, 1)

			res := make([]*graph.Node, 1+len(serializedUpdatedCache))
			res[0] = graph.ConvertDType(lastLogits, dtypes.Float32)
			for i, n := range serializedUpdatedCache {
				res[i+1] = n
			}
			return res
		})
		if err != nil {
			t.Fatal(err)
		}

		promptInputs := make([]any, 1+len(kvCacheTensorsSerialized))
		promptInputs[0] = promptTensor
		for i, ts := range kvCacheTensorsSerialized {
			promptInputs[i+1] = ts
		}

		outputs, err := promptExec.Exec(promptInputs...)
		if err != nil {
			t.Fatal(err)
		}

		logitsTensor := outputs[0]
		kvCacheTensors = tm.KVCache.DeserializeTensors(outputs[1:])

		// Print Step 1 top 5
		printTop5(tokenizer, logitsTensor.Value().([][]float32)[0], "Step 1 (Prompt)")

		// Step 2: Feed token 236776 at pos 17
		kvCacheTensors, err = tm.KVCache.PadTensors(kvCacheTensors, 17)
		if err != nil {
			t.Fatal(err)
		}
		kvCacheTensorsSerialized, err = tm.KVCache.SerializeTensors(kvCacheTensors)
		if err != nil {
			t.Fatal(err)
		}

		step2Exec, err := model.NewExec(backend, store, func(scope *model.Scope, inputs []*graph.Node) []*graph.Node {
			token := inputs[0]
			positionNode := inputs[1]
			cacheNodes := inputs[2:]
			tokenReshaped := graph.ExpandDims(token, -1)
			cache := tm.KVCache.DeserializeNodes(cacheNodes)
			logits, updatedCache := hfModel.Forward(scope, tokenReshaped, positionNode, nil, nil, cache)
			serializedUpdatedCache, err := tm.KVCache.SerializeNodes(updatedCache)
			if err != nil {
				panic(err)
			}
			lastLogits := graph.Squeeze(logits, 1)
			res := make([]*graph.Node, 1+len(serializedUpdatedCache))
			res[0] = graph.ConvertDType(lastLogits, dtypes.Float32)
			for i, n := range serializedUpdatedCache {
				res[i+1] = n
			}
			return res
		})
		if err != nil {
			t.Fatal(err)
		}

		step2Inputs := make([]any, 2+len(kvCacheTensorsSerialized))
		step2Inputs[0] = tensors.FromValue([]int32{236776})
		step2Inputs[1] = tensors.FromValue(int32(17))
		for i, ts := range kvCacheTensorsSerialized {
			step2Inputs[i+2] = ts
		}

		outputs2, err := step2Exec.Exec(step2Inputs...)
		if err != nil {
			t.Fatal(err)
		}

		logitsTensor2 := outputs2[0]
		kvCacheTensors = tm.KVCache.DeserializeTensors(outputs2[1:])
		printTop5(tokenizer, logitsTensor2.Value().([][]float32)[0], "Step 2 (after 236776)")

		// Step 3: Feed token 89830 at pos 18
		kvCacheTensors, err = tm.KVCache.PadTensors(kvCacheTensors, 18)
		if err != nil {
			t.Fatal(err)
		}
		kvCacheTensorsSerialized, err = tm.KVCache.SerializeTensors(kvCacheTensors)
		if err != nil {
			t.Fatal(err)
		}

		step3Exec, err := model.NewExec(backend, store, func(scope *model.Scope, inputs []*graph.Node) []*graph.Node {
			token := inputs[0]
			positionNode := inputs[1]
			cacheNodes := inputs[2:]
			tokenReshaped := graph.ExpandDims(token, -1)
			cache := tm.KVCache.DeserializeNodes(cacheNodes)
			logits, updatedCache := hfModel.Forward(scope, tokenReshaped, positionNode, nil, nil, cache)
			serializedUpdatedCache, err := tm.KVCache.SerializeNodes(updatedCache)
			if err != nil {
				panic(err)
			}
			lastLogits := graph.Squeeze(logits, 1)
			res := make([]*graph.Node, 1+len(serializedUpdatedCache))
			res[0] = graph.ConvertDType(lastLogits, dtypes.Float32)
			for i, n := range serializedUpdatedCache {
				res[i+1] = n
			}
			return res
		})
		if err != nil {
			t.Fatal(err)
		}

		step3Inputs := make([]any, 2+len(kvCacheTensorsSerialized))
		step3Inputs[0] = tensors.FromValue([]int32{89830})
		step3Inputs[1] = tensors.FromValue(int32(18))
		for i, ts := range kvCacheTensorsSerialized {
			step3Inputs[i+2] = ts
		}

		outputs3, err := step3Exec.Exec(step3Inputs...)
		if err != nil {
			t.Fatal(err)
		}

		logitsTensor3 := outputs3[0]
		printTop5(tokenizer, logitsTensor3.Value().([][]float32)[0], "Step 3 (after 89830)")
	})
}

func printTop5(tokenizer any, logits []float32, label string) {
	type entry struct {
		index int
		val   float32
	}
	top := make([]entry, 0, 5)
	for idx, val := range logits {
		if len(top) < 5 {
			top = append(top, entry{idx, val})
			for i := len(top) - 1; i > 0; i-- {
				if top[i].val > top[i-1].val {
					top[i], top[i-1] = top[i-1], top[i]
				}
			}
		} else if val > top[4].val {
			top[4] = entry{idx, val}
			for i := 4; i > 0; i-- {
				if top[i].val > top[i-1].val {
					top[i], top[i-1] = top[i-1], top[i]
				}
			}
		}
	}
	fmt.Printf("\n--- Go %s ---\n", label)
	fmt.Printf("Top 5 IDs: ")
	for _, e := range top {
		fmt.Printf("%d ", e.index)
	}
	fmt.Println()
	fmt.Printf("Top 5 values: ")
	for _, e := range top {
		fmt.Printf("%f ", e.val)
	}
	fmt.Println()
	tok := tokenizer.(interface{ Decode(ids []int) string })
	fmt.Printf("Decoded: ")
	for _, e := range top {
		fmt.Printf("%q ", tok.Decode([]int{e.index}))
	}
	fmt.Println()
}
