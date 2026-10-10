// Package transformer provides loading and mapping of HuggingFace transformer models into
// GoMLX computation graphs (using github.com/gomlx/gomlx/ml/zoo/transformer).
//
// It loads configuration files (config.json, config_sentence_transformers.json, modules.json,
// and 1_Pooling/config.json), reads safetensors weights into a GoMLX model.Store, and constructs
// computation graphs for text embeddings, transformer representations, or sentence-transformer pipelines.
//
// Models are loaded using [LoadModel]:
//
//	repo := hub.New("google/gemma-3-1b-pt")
//	model, err := transformer.LoadModel(repo)
//	if err != nil {
//		...
//	}
//
// By default, [LoadModel] automatically inspects the loaded model configuration to detect whether
// attention layers should use a causal mask (true for autoregressive decoder models like Gemma, LLaMA,
// or KaLM; false for bidirectional encoder models like BERT). This behavior can be inspected with
// [Model.UseCausalMask] and overridden via [Model.WithCausalMask].
package transformer
