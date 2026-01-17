// Package hub can be used to download and cache files from HuggingFace Hub, which may
// be models, tokenizers or anything.
//
// It is meant to be a port of huggingFace_hub python library to Go, and be able to share the same
// cache structure (usually under "~/.cache/huggingface/hub").
//
// It is also safe to be used concurrently by multiple programs -- it uses file system lock to control concurrency.
//
// Typical usage will be something like:
//
//	repo := hub.New(modelID).WithAuth(hfAuthToken)
//	var fileNames []string
//	for fileName, err := range repo.IterFileNames() {
//		if err != nil { panic(err) }
//		fmt.Printf("\t%s\n", fileName)
//		fileNames = append(fileNames, fileName)
//	}
//	downloadedFiles, err := repo.DownloadFiles(fileNames...)
//	if err != nil { ... }
//
// From here, downloadedFiles will point to files in the local cache that one can read.
//
// Environment variables:
//
// - HF_ENDPOINT: Where to connect to huggingface, default is https://huggingface.co
// - XDG_CACHE_HOME: Cache directory, defaults to ${HOME}/.cache
package hub

import (
	"fmt"
	"os"
	"path"
	"runtime"
	"strings"

	"github.com/gomlx/go-huggingface"
	"github.com/google/uuid"
	"github.com/pkg/errors"
)

// SessionId is unique and always created anew at the start of the program, and used during the life of the program.
var SessionId string

// panicf generates an error message and panics with it, in one function.
func panicf(format string, args ...any) {
	err := errors.Errorf(format, args...)
	panic(err)
}

func init() {
	sessionUUID, err := uuid.NewRandom()
	if err != nil {
		panicf("failed generating UUID for SessionId: %v", err)
	}
	SessionId = strings.Replace(sessionUUID.String(), "-", "", -1)
}

var (
	// DefaultDirCreationPerm is used when creating new cache subdirectories.
	DefaultDirCreationPerm = os.FileMode(0755)

	// DefaultFileCreationPerm is used when creating files inside the cache subdirectories.
	DefaultFileCreationPerm = os.FileMode(0644)
)

const (
	HeaderXRepoCommit = "X-Repo-Commit"
	HeaderXLinkedETag = "X-Linked-Etag"
	HeaderXLinkedSize = "X-Linked-Size"
)

func getEnvOr(key, defaultValue string) string {
	v := os.Getenv(key)
	if v == "" {
		return defaultValue
	}
	return v
}

// DefaultCacheDir for HuggingFace Hub, same used by the python library.
//
// Its prefix is either `${XDG_CACHE_HOME}` if set, or `~/.cache` otherwise. Followed by `/huggingface/hub/`.
// So typically: `~/.cache/huggingface/hub/`.
func DefaultCacheDir() string {
	cacheDir := getEnvOr("XDG_CACHE_HOME", path.Join(os.Getenv("HOME"), ".cache"))
	cacheDir = path.Join(cacheDir, "huggingface", "hub")
	return cacheDir
}

// DefaultHttpUserAgent returns a user agent to use with HuggingFace Hub API.
func DefaultHttpUserAgent() string {
	return fmt.Sprintf("go-huggingface/%v; golang/%s; session_id/%s",
		huggingface.Version, runtime.Version(), SessionId)
}

// RepoIdSeparator is used to separate repository/model names parts when mapping to file names.
// Likely only for internal use.
const RepoIdSeparator = "--"

// RepoType supported by HuggingFace-Hub
type RepoType string

const (
	RepoTypeDataset RepoType = "datasets"
	RepoTypeSpace   RepoType = "spaces"
	RepoTypeModel   RepoType = "models"
)
