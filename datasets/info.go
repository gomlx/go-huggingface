package datasets

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path"

	"github.com/gomlx/go-huggingface/internal/files"
	"github.com/pkg/errors"
)

// Info holds information about a dataset returned by the datasets-server.
// See the API at: https://datasets-server.huggingface.co/info?dataset=<id>
type Info struct {
	// DatasetInfo is keyed on the "config" name.
	// Some datasets use "config" names to denote versions, e.g. "v1.1".
	DatasetInfo map[string]ConfigInfo `json:"dataset_info"`
}

type ParquetFileResponse struct {
	ParquetFiles []ParquetFile `json:"parquet_files"`
}

type ParquetFile struct {
	Dataset  string `json:"dataset"`
	Config   string `json:"config"`
	Split    string `json:"split"`
	URL      string `json:"url"`
	Filename string `json:"filename"`
	Size     int64  `json:"size"`
}

// ConfigInfo holds the info for a specific configuration of a dataset.
type ConfigInfo struct {
	Description  string             `json:"description"`
	Citation     string             `json:"citation"`
	Homepage     string             `json:"homepage"`
	License      string             `json:"license"`
	Features     map[string]Feature `json:"features"`
	BuilderName  string             `json:"builder_name"`
	ConfigName   string             `json:"config_name"`
	Version      map[string]any     `json:"version"`
	Splits       map[string]Split   `json:"splits"`
	DownloadSize int64              `json:"download_size"`
	DatasetSize  int64              `json:"dataset_size"`
}

// Feature represents a dataset feature definition (column schema).
type Feature struct {
	Type  string `json:"_type,omitempty"`
	DType string `json:"dtype,omitempty"`

	// SubFeature is used when Type is "Sequence" or "Array". It will be a map[string]Feature.
	// If the feature is a single value, it will be stored under the empty string key ("").
	SubFeature map[string]Feature `json:"-"`

	// Used by features like ClassLabel
	Names []string `json:"names,omitempty"`
}

// Split holds information about a dataset split (e.g. "train", "test", "validation").
type Split struct {
	DatasetName string `json:"dataset_name,omitempty"`
	Name        string `json:"name"`
	NumBytes    int64  `json:"num_bytes"`
	NumExamples int64  `json:"num_examples"`
}

// UnmarshalJSON implements custom JSON unmarshaling to handle dynamic "feature" values.
func (f *Feature) UnmarshalJSON(b []byte) error {
	type Alias Feature
	aux := &struct {
		*Alias
		FeatureRaw json.RawMessage `json:"feature"`
	}{
		Alias: (*Alias)(f),
	}

	if err := json.Unmarshal(b, &aux); err != nil {
		return err
	}

	if aux.FeatureRaw != nil {
		var generic map[string]any
		if err := json.Unmarshal(aux.FeatureRaw, &generic); err == nil {
			if _, hasType := generic["_type"]; hasType {
				var single Feature
				if err := json.Unmarshal(aux.FeatureRaw, &single); err == nil {
					f.SubFeature = map[string]Feature{"": single}
				}
			} else {
				var multi map[string]Feature
				if err := json.Unmarshal(aux.FeatureRaw, &multi); err == nil {
					f.SubFeature = multi
				}
			}
		}
	}
	return nil
}

// MarshalJSON implements custom JSON marshaling to flatten single SubFeatures.
func (f Feature) MarshalJSON() ([]byte, error) {
	type Alias Feature
	aux := &struct {
		*Alias
		FeatureRaw any `json:"feature,omitempty"`
	}{
		Alias: (*Alias)(&f),
	}

	if f.SubFeature != nil {
		if single, ok := f.SubFeature[""]; ok && len(f.SubFeature) == 1 {
			aux.FeatureRaw = single
		} else {
			aux.FeatureRaw = f.SubFeature
		}
	}

	return json.Marshal(aux)
}

// String implements fmt.Stringer to pretty-print the Info.
func (i *Info) String() string {
	b, err := json.MarshalIndent(i, "", "  ")
	if err != nil {
		return fmt.Sprintf("Info(error: %v)", err)
	}
	return string(b)
}

// String implements fmt.Stringer to pretty-print the ConfigInfo.
func (c *ConfigInfo) String() string {
	b, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Sprintf("ConfigInfo(error: %v)", err)
	}
	return string(b)
}

// Info returns the Info structure about the dataset.
// If it hasn't been downloaded or loaded from the cache yet, it loads it first.
// It may return nil if there was an issue with the downloading of the Info json from HuggingFace.
// Try DownloadDatasetInfo to get an error.
func (d *Dataset) Info() (*Info, error) {
	if d.info == nil {
		err := d.DownloadDatasetInfo(model.Background(), false)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to download info for dataset %q", d.ID)
		}
	}
	return d.info, nil
}

// datasetServerURL for the API that returns the info about a dataset.
func (d *Dataset) datasetServerURL() string {
	// Use datasets-server.huggingface.co
	return fmt.Sprintf("https://datasets-server.huggingface.co/info?dataset=%s", d.ID)
}

// DownloadDatasetInfo downloads the dataset info using the datasets-server.
//
// If forceDownload is set to true, it ignores the cached one.
func (d *Dataset) DownloadDatasetInfo(ctx model.Context, forceDownload bool) error {
	if d.info != nil && !forceDownload {
		return nil
	}

	// Create directory and file path for the info file.
	infoFilePath, err := d.Repo.CacheDir()
	if err != nil {
		return err
	}
	infoFilePath = path.Join(infoFilePath, "dataset_info.json")

	// Download info file if needed.
	if !files.Exists(infoFilePath) || forceDownload {
		err := d.Repo.GetDownloadManager().LockedDownload(ctx, d.datasetServerURL(), infoFilePath, forceDownload, nil)
		if err != nil {
			return errors.WithMessagef(err, "failed to download dataset info")
		}
	}

	// Read dataset_info.json from disk.
	infoJson, err := os.ReadFile(infoFilePath)
	if err != nil {
		return errors.Wrapf(err, "failed to read info for dataset from disk in %q -- remove the file if you want to have it re-downloaded", infoFilePath)
	}

	decoder := json.NewDecoder(bytes.NewReader(infoJson))
	newInfo := &Info{}
	if err = decoder.Decode(newInfo); err != nil {
		return errors.Wrapf(err, "failed to parse info for dataset in %q (downloaded from %q)", infoFilePath, d.datasetServerURL())
	}
	d.info = newInfo
	return nil
}

// datasetServerParquetURL returns the URL to query parquet files info.
func (d *Dataset) datasetServerParquetURL() string {
	return fmt.Sprintf("https://datasets-server.huggingface.co/parquet?dataset=%s", d.ID)
}

// GetParquetFiles returns the list of parquet files for the dataset.
// If it hasn't been downloaded or loaded from the cache yet, it loads it first.
func (d *Dataset) GetParquetFiles() ([]ParquetFile, error) {
	if d.Files == nil {
		err := d.DownloadParquetFilesInfo(model.Background(), false)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to download parquet files info for dataset %q", d.ID)
		}
	}
	return d.Files, nil
}

// DownloadParquetFilesInfo downloads the parquet info using the datasets-server.
//
// If forceDownload is set to true, it ignores the cached one.
func (d *Dataset) DownloadParquetFilesInfo(ctx model.Context, forceDownload bool) error {
	if d.Files != nil && !forceDownload {
		return nil
	}

	infoFilePath, err := d.Repo.CacheDir()
	if err != nil {
		return err
	}
	infoFilePath = path.Join(infoFilePath, "dataset_parquet.json")

	// Download info file if needed.
	if !files.Exists(infoFilePath) || forceDownload {
		err := d.Repo.GetDownloadManager().LockedDownload(ctx, d.datasetServerParquetURL(), infoFilePath, forceDownload, nil)
		if err != nil {
			return errors.WithMessagef(err, "failed to download dataset parquet files info")
		}
	}

	infoJson, err := os.ReadFile(infoFilePath)
	if err != nil {
		return errors.Wrapf(err, "failed to read parquet info for dataset from disk in %q -- remove the file if you want to have it re-downloaded", infoFilePath)
	}

	decoder := json.NewDecoder(bytes.NewReader(infoJson))
	var response ParquetFileResponse
	if err = decoder.Decode(&response); err != nil {
		return errors.Wrapf(err, "failed to parse parquet info for dataset in %q (downloaded from %q)", infoFilePath, d.datasetServerParquetURL())
	}
	d.Files = response.ParquetFiles
	return nil
}
