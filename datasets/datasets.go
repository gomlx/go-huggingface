// Package datasets provides an interface for interacting with the HuggingFace datasets server
// and downloading dataset files, functioning similarly to the `hub` package.
package datasets

import (
	"github.com/gomlx/go-huggingface/hub"
)

// Dataset from which one wants to download files or get info.
// It embeds a *hub.Repo to reuse its downloading, caching, and file URL resolving functionality.
type Dataset struct {
	*hub.Repo
	info *Info
}

// New creates a reference to a HuggingFace dataset given its id.
// The id typically includes owner/dataset name. E.g.: "microsoft/ms_marco".
func New(id string) *Dataset {
	return &Dataset{
		Repo: hub.New(id).WithType(hub.RepoTypeDataset),
	}
}

// WithAuth sets the authentication token to use during downloads and info requests.
// Setting it to empty ("") is the same as resetting and not using authentication.
func (d *Dataset) WithAuth(authToken string) *Dataset {
	d.Repo.WithAuth(authToken)
	return d
}

// WithEndpoint sets the HuggingFace endpoint to use.
// Default is "https://huggingface.co" or, if set, the environment variable HF_ENDPOINT.
func (d *Dataset) WithEndpoint(endpoint string) *Dataset {
	d.Repo.WithEndpoint(endpoint)
	return d
}

// WithRevision sets the revision to use for this Dataset, defaults to "main", but can be set to a commit-hash value.
func (d *Dataset) WithRevision(revision string) *Dataset {
	d.Repo.WithRevision(revision)
	return d
}

// WithCacheDir sets the cache directory for downloaded files to the given directory.
func (d *Dataset) WithCacheDir(cacheDir string) *Dataset {
	d.Repo.WithCacheDir(cacheDir)
	return d
}
