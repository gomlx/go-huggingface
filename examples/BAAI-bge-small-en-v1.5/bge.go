package bge

const (
	// Repository for BAAI/bge-small-en-v1.5 [1]
	//
	// This is an embedding model, not a language model. It is trained to produce
	// embeddings for sentences, not to generate text.
	//
	// [1] https://huggingface.co/BAAI/bge-small-en-v1.5
	Repository = "BAAI/bge-small-en-v1.5"

	// Embedding dimension, per token, or pooled for sentence.
	EmbeddingDim = 384

	// Query instruction: prepend this to queries.
	//
	// Unfortunately, this is not automatically included in the `config_sentence_transformers.json` file.
	//
	// (Translation: "Generate a representation for this sentence to be used in retrieving related articles:")
	QueryInstruction = "为这个句子生成表示以用于检索相关文章："
)
