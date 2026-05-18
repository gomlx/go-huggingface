// Package datasets provides an interface for interacting with the HuggingFace datasets server
// and downloading dataset files, functioning similarly to the `hub` package.
package datasets

import (
	"bytes"
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/hub"
)

// Dataset from which one wants to download files or get info.
// It embeds a *hub.Repo to reuse its downloading, caching, and file URL resolving functionality.
type Dataset struct {
	*hub.Repo
	info *Info

	// Files slice contains a list of backing ParquetFile entries for the dataset.
	// It is lazily populated via GetParquetFiles().
	Files []ParquetFile
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

// String returns a formatted, human-readable summary of the dataset.
func (d *Dataset) String() string {
	info, err := d.Info()
	if err != nil {
		return fmt.Sprintf("Dataset ID: %s -- failed to retrieve info: %v", d.ID, err)
	}
	if info == nil || len(info.DatasetInfo) == 0 {
		return fmt.Sprintf("Dataset ID: %s -- no info available", d.ID)
	}

	var buf bytes.Buffer
	buf.WriteString(fmt.Sprintf("Dataset ID: %s\n", d.ID))

	err = d.DownloadParquetFilesInfo(context.Background(), false)
	if err != nil {
		buf.WriteString(fmt.Sprintf(" - Failed to retrieve parquet files info: %+v", err))
	}

	var configs []string
	for k := range info.DatasetInfo {
		configs = append(configs, k)
	}
	sort.Strings(configs)

	for _, config := range configs {
		c := info.DatasetInfo[config]

		var features []string
		for featName, feat := range c.Features {
			if feat.Type == "Sequence" {
				features = append(features, featName+"[*]")
			} else {
				features = append(features, featName)
			}
		}
		sort.Strings(features)

		buf.WriteString(fmt.Sprintf("\nConfig: %s\n", config))
		buf.WriteString(fmt.Sprintf("  Features: %s\n", strings.Join(features, ", ")))

		var splits []string
		var splitKeys []string
		for s := range c.Splits {
			splitKeys = append(splitKeys, s)
		}
		sort.Strings(splitKeys)

		for _, s := range splitKeys {
			splitInfo := c.Splits[s]
			fileCount := 0
			if d.Files != nil {
				for _, f := range d.Files {
					if f.Config == config && f.Split == s {
						fileCount++
					}
				}
			}
			fileString := ""
			if fileCount > 0 {
				fileString = fmt.Sprintf(", %d files", fileCount)
			}
			splits = append(splits, fmt.Sprintf("%s (%s records, %s%s)",
				s,
				humanize.Count(splitInfo.NumExamples),
				humanize.Bytes(splitInfo.NumBytes),
				fileString,
			))
		}
		buf.WriteString(fmt.Sprintf("  Splits: %s\n", strings.Join(splits, ", ")))
	}

	return buf.String()
}
