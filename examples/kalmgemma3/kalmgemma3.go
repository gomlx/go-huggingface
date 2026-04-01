// kalmgemma3 package loads the KaLM-Gemma3 model configuration and weights.
//
// See KaLM embedding site [1] and HuggingFace model repo [2]. You'll find
// links to the papers in either these locations.
//
// [1] https://kalm-embedding.github.io/
// [2] https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511
package kalmgemma3

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/pkg/errors"
)

const (
	// Repository for KaLM-Embedding-Gemma3-12B-2511.
	//
	// This is an embedding model, not a language model. It is trained to produce
	// embeddings for sentences, not to generate text.
	//
	// Remember to set Model.WithCausalMask(false) when using this model.
	// (The default is set to use causal mask, but the model shouldn't use it)
	Repository = "tencent/KaLM-Embedding-Gemma3-12B-2511"

	// Embedding dimension, per token, or pooled for sentence.
	EmbeddingDim = 3840
)

// LoadRepo creates a hub.Repo that can be used to download tokenizer, configuration files and model files.
func LoadRepo() (*hub.Repo, error) {
	repo := hub.New(Repository)
	if err := repo.DownloadInfo(false); err != nil {
		return nil, fmt.Errorf("failed to get repo info: %w", err)
	}
	return repo, nil
}

// TaskPrompts is a map of task codes to their prompts, loaded from the model's
// `task_prompts.json` file. Load it with LoadTaskPrompts.
type TaskPrompts map[string]string

// LoadTaskPrompts loads the task prompts specific to KaLM-Gemma3 ("task_prompts.json").
func LoadTaskPrompts(repo *hub.Repo) (TaskPrompts, error) {
	path, err := repo.DownloadFile("task_prompts.json")
	if err != nil {
		return nil, errors.Wrapf(err, "failed to download task_prompts.json")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read task_prompts.json")
	}
	var prompts map[string]any
	if err := json.Unmarshal(b, &prompts); err != nil {
		return nil, errors.Wrapf(err, "failed to unmarshal task_prompts.json")
	}
	taskPrompts := make(TaskPrompts, len(prompts))
	for code, promptAny := range prompts {
		prompt, ok := promptAny.(string)
		if ok {
			taskPrompts[code] = prompt
		}
	}
	return taskPrompts, nil
}

// BuildQueryPrompt builds a query prompt for the given task code.
// If the taskCode is empty or is not found, it uses the default query prompt.
func (t TaskPrompts) BuildQueryPrompt(query, taskCode string) string {
	prompt := t[taskCode]
	if prompt == "" {
		return query
	}
	return fmt.Sprintf("Instruct: %s\nQuery: %s", prompt, query)
}
