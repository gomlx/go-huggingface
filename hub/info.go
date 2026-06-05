package hub

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path"
	"time"

	"github.com/gomlx/go-huggingface/internal/files"
	"github.com/pkg/errors"
)

// RepoInfo holds information about a HuggingFace repo, it is the json served when hitting the URL
// https://huggingface.co/api/<repo_type>/<model_id>?blobs=true
type RepoInfo struct {
	InternalID   string            `json:"_id"`
	ID           string            `json:"id"`
	ModelID      string            `json:"modelId"` // Also support model_id via custom UnmarshalJSON
	Author       string            `json:"author"`
	CommitHash   string            `json:"sha"`
	PipelineTag  string            `json:"pipeline_tag"`
	Tags         []string          `json:"tags"`
	Siblings     []*FileInfo       `json:"siblings"`
	SafeTensors  SafeTensorsInfo   `json:"safetensors"`
	Private      bool              `json:"private"`
	LibraryName  string            `json:"library_name"`
	Downloads    int64             `json:"downloads"`
	Likes        int64             `json:"likes"`
	LastModified time.Time         `json:"lastModified"`
	Gated        any               `json:"gated"`
	Disabled     bool              `json:"disabled"`
	WidgetData   []map[string]any  `json:"widgetData"`
	WidgetInfo   map[string]any    `json:"widgetInfo"`
	ModelIndex   any               `json:"model-index"`
	Config       *RepoConfig       `json:"config"`
	CardData     *CardData         `json:"cardData"`
	Transformers *TransformersInfo `json:"transformersInfo"`
	Spaces       []string          `json:"spaces"`
	CreatedAt    time.Time         `json:"createdAt"`
	UsedStorage  int64             `json:"usedStorage"`
}

// UnmarshalJSON customizes unmarshaling for RepoInfo to support both "modelId" and "model_id".
func (r *RepoInfo) UnmarshalJSON(data []byte) error {
	type Alias RepoInfo
	aux := &struct {
		ModelIDUnderscore string `json:"model_id"`
		*Alias
	}{
		Alias: (*Alias)(r),
	}
	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}
	if r.ModelID == "" && aux.ModelIDUnderscore != "" {
		r.ModelID = aux.ModelIDUnderscore
	}
	return nil
}

// RepoConfig contains configuration info about the repository/model.
type RepoConfig struct {
	Architectures   []string       `json:"architectures"`
	ModelType       string         `json:"model_type"`
	TokenizerConfig map[string]any `json:"tokenizer_config"`
}

// CardData contains metadata about the model card.
type CardData struct {
	LibraryName string `json:"library_name"`
	License     string `json:"license"`
	LicenseLink string `json:"license_link"`
	PipelineTag string `json:"pipeline_tag"`
	BaseModel   any    `json:"base_model"` // Can be a string, or slice of strings, or nil.
}

// TransformersInfo contains information related to the Hugging Face Transformers integration.
type TransformersInfo struct {
	AutoModel   string `json:"auto_model"`
	PipelineTag string `json:"pipeline_tag"`
	Processor   string `json:"processor"`
}

// FileInfo represents one of the model files, in the Info structure.
//
// The Hub is built on top of Git. Because standard Git is highly inefficient at handling large files (like machine
// learning model weights or massive datasets), Hugging Face heavily utilizes Git LFS (Large File Storage).
type FileInfo struct {
	Name string `json:"rfilename"`

	// Size is the file size in bytes.
	//
	// It may be left as 0 if not retrieved (if set WithExtraBlobInfo(false)).
	Size int64 `json:"size"`

	// BlobID is This is the standard Git Object ID (a SHA-1 hash) representing the file in the Git tree.
	//
	// It may be left as "" if not retrieved (if set WithExtraBlobsInfo(false)).
	BlobID string `json:"blobId"`

	// LFS holds Git LFS details if the file is tracked by LFS.
	//
	// It is not retrieved if WithExtraBlobsInfo(false).
	LFS *LFSInfo `json:"lfs"`
}

// LFSInfo holds Git LFS details for a file tracked by LFS.
type LFSInfo struct {
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`

	// PointerSize the size of the tiny Git LFS pointer file that lives inside the Git tree, in bytes. Because it only
	// contains a few lines of text (the LFS version, the SHA-256 hash of the real file, and the real file's size), this
	// value is almost always between 130 and 135 bytes. If a file is a standard text file and not tracked by LFS, this
	// concept doesn't apply (it may be omitted or set to 0).
	PointerSize int64 `json:"pointerSize"`
}

// SafeTensorsInfo holds counts on number of parameters of various types.
type SafeTensorsInfo struct {
	Total      int64            `json:"total"`
	Parameters map[string]int64 `json:"parameters"`
}

// Info returns the RepoInfo structure about the model.
// Most users don't need to call this directly, instead use the various iterators.
//
// If it hasn't been downloaded or loaded from the cache yet, it loads it first.
//
// It may return nil if there was an issue with the downloading of the RepoInfo json from HuggingFace.
// Try DownloadInfo to get an error.
func (r *Repo) Info() *RepoInfo {
	if r.info == nil {
		err := r.DownloadInfo(false)
		if err != nil {
			log.Printf("Error while downloading info about Repo: %+v", err)
		}
	}
	return r.info
}

// infoURL for the API that returns the info about a repository.
func (r *Repo) infoURL() string {
	var blobs string
	if r.extraBlobsInfo {
		blobs = "?blobs=true"
	}
	return fmt.Sprintf("%s/api/%s/%s/revision/%s%s", r.hfEndpoint, r.repoType, r.ID, r.revision, blobs)
}

// DownloadInfo about the model, if it hasn't yet.
//
// It will attempt to use the "_info_.json" file in the cache directory first.
//
// If forceDownload is set to true, it ignores the current info or the cached one, and download it again from HuggingFace.
//
// See Repo.Info to access the Info directory.
// Most users don't need to call this directly, instead use the various iterators.
func (r *Repo) DownloadInfo(forceDownload bool) error {
	if r.info != nil && !forceDownload {
		return nil
	}

	// Create directory and file path for the info file.
	infoFilePath, err := r.repoCacheDir()
	if err != nil {
		return err
	}
	infoFilePath = path.Join(infoFilePath, "info")
	if err = os.MkdirAll(infoFilePath, DefaultDirCreationPerm); err != nil {
		return errors.Wrapf(err, "while creating info directory %q", infoFilePath)
	}
	infoFilePath = path.Join(infoFilePath, r.revision)

	// Download info file if needed.
	if !files.Exists(infoFilePath) || forceDownload {
		err := r.GetDownloadManager().LockedDownload(context.Background(), r.infoURL(), infoFilePath, forceDownload, nil)
		if err != nil {
			return errors.WithMessagef(err, "failed to download repository info")
		}
	}

	// Read _info_.json from disk.
	infoJson, err := os.ReadFile(infoFilePath)
	if err != nil {
		return errors.Wrapf(err, "failed to read info for model from disk in %q -- remove the file if you want to have it re-downloaded",
			infoFilePath)
	}

	decoder := json.NewDecoder(bytes.NewReader(infoJson))
	newInfo := &RepoInfo{}
	if err = decoder.Decode(newInfo); err != nil {
		return errors.Wrapf(err, "failed to parse info for model in %q (downloaded from %q)",
			infoFilePath, r.infoURL())
	}
	r.info = newInfo
	return nil
}
