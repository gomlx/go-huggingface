// Package gemma4e4bit loads the Gemma 4 E4B model configuration and weights.
package gemma4e4bit

import (
	"fmt"

	"github.com/gomlx/go-huggingface/hub"
)

const (
	// Repository for google/gemma-4-E4B-it.
	Repository = "google/gemma-4-E4B-it"
)

// LoadRepo creates a hub.Repo that can be used to download tokenizer, configuration files and model files.
func LoadRepo() (*hub.Repo, error) {
	repo := hub.New(Repository)
	if err := repo.DownloadInfo(false); err != nil {
		return nil, fmt.Errorf("failed to get repo info: %w", err)
	}
	return repo, nil
}
