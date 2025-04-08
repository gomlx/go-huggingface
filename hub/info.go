package hub

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path"

	"github.com/gomlx/go-huggingface/internal/files"
	"github.com/pkg/errors"
)

// RepoInfo holds information about a HuggingFace repo, it is the json served when hitting the URL
// https://huggingface.co/api/<repo_type>/<model_id>
//
// TODO: Not complete, only holding the fields used so far by the library.
type RepoInfo struct {
	ID          string          `json:"id"`
	ModelID     string          `json:"model_id"`
	Author      string          `json:"author"`
	CommitHash  string          `json:"sha"`
	Tags        []string        `json:"tags"`
	Siblings    []*FileInfo     `json:"siblings"`
	SafeTensors SafeTensorsInfo `json:"safetensors"`
}

// FileInfo represents one of the model file, in the Info structure.
type FileInfo struct {
	Name string `json:"rfilename"`
}

// SafeTensorsInfo holds counts on number of parameters of various types.
type SafeTensorsInfo struct {
	Total int

	// Parameters: maps dtype name to int.
	Parameters map[string]int
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
	return fmt.Sprintf("%s/api/%s/%s/revision/%s", r.hfEndpoint, r.repoType, r.ID, r.revision)
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
		err := r.lockedDownload(context.Background(), r.infoURL(), infoFilePath, forceDownload, nil)
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
