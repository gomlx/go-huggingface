package minilmgguf

const (
	// Repository for second-state/All-MiniLM-L6-v2-Embedding-GGUF [1]
	//
	// GGUF-quantized BERT sentence-embedding model (all-MiniLM-L6-v2).
	//
	// [1] https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF
	Repository = "second-state/All-MiniLM-L6-v2-Embedding-GGUF"

	// DefaultGGUFFile is the default quantization file selected by transformer.LoadModel.
	DefaultGGUFFile = "all-MiniLM-L6-v2-Q4_K_M.gguf"

	// EmbeddingDim is the pooled sentence embedding dimension.
	EmbeddingDim = 384
)
