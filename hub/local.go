package hub

import (
	"io/fs"
	"os"
	"path/filepath"
	"sort"

	"github.com/gomlx/go-huggingface/internal/files"
	"github.com/pkg/errors"
)

// LocalCommitHash is used as the synthetic RepoInfo.CommitHash for repositories in local-directory mode.
const LocalCommitHash = "local"

// localDirsToSkip lists directory names, at any depth, that are never scanned as part of the repository's
// file listing in local-directory mode (see NewLocal). These are VCS/tooling directories that are never
// part of a HuggingFace repo's own file listing.
var localDirsToSkip = map[string]bool{
	".git":               true,
	".cache":             true,
	".huggingface":       true,
	".ipynb_checkpoints": true,
}

// NewLocal creates a Repo that reads files directly from a local directory dir, instead of downloading them
// from HuggingFace Hub.
//
// dir is expected to be a plain directory containing the repository files (e.g. config.json, *.safetensors,
// tokenizer.json, ...), such as what one gets from `git clone` or `huggingface-cli download --local-dir`. It
// can also point directly at a snapshot directory inside an existing HuggingFace cache
// (".../snapshots/<commit-hash>").
//
// In local-directory mode, no network access is ever made: Repo.DownloadInfo scans dir for files instead of
// querying the HuggingFace API, and Repo.DownloadFile(s) simply resolve to paths inside dir. Options that only
// make sense for remote repositories (WithAuth, WithEndpoint, WithRevision, WithCacheDir,
// WithExtraBlobsInfo, WithProgressBar, MaxParallelDownload) are ignored. Repo.FileURL returns an error, since
// there is no remote URL.
//
// The returned Repo's ID defaults to the base name of dir; set Repo.ID explicitly (e.g. "BAAI/bge-small-zh-v1.5")
// if you want a more descriptive identifier to show up in logs and error messages.
func NewLocal(dir string) *Repo {
	r := New("")
	return r.WithLocalDir(dir)
}

// WithLocalDir switches r to local-directory mode, reading files directly from dir instead of downloading them
// from HuggingFace Hub. See NewLocal for the semantics of local-directory mode.
//
// Passing "" switches r back to normal (remote) mode.
func (r *Repo) WithLocalDir(dir string) *Repo {
	if dir == "" {
		r.localDir = ""
		return r
	}
	resolved, err := files.ReplaceTildeInDir(dir)
	if err != nil {
		resolved = dir
	}
	r.localDir = filepath.Clean(resolved)
	// Local mode changes what DownloadInfo means, so invalidate anything cached from a possible previous
	// (remote) configuration.
	r.info = nil
	r.revisionHashRefreshed = false
	if r.ID == "" {
		r.ID = filepath.Base(r.localDir)
	}
	return r
}

// IsLocal returns whether r is in local-directory mode. See NewLocal.
func (r *Repo) IsLocal() bool {
	return r.localDir != ""
}

// LocalDir returns the local directory used by r, if it is in local-directory mode (see NewLocal).
// It returns "" otherwise.
func (r *Repo) LocalDir() string {
	return r.localDir
}

// scanLocalInfo builds a RepoInfo by walking r.localDir, for local-directory mode. It is the local-mode
// counterpart of the network-based Repo.DownloadInfo.
func (r *Repo) scanLocalInfo(forceRescan bool) error {
	if r.info != nil && !forceRescan {
		return nil
	}
	st, err := os.Stat(r.localDir)
	if err != nil {
		return errors.Wrapf(err, "local model directory %q is not accessible", r.localDir)
	}
	if !st.IsDir() {
		return errors.Errorf("local model path %q is not a directory", r.localDir)
	}

	info := &RepoInfo{
		InternalID: r.ID,
		ID:         r.ID,
		ModelID:    r.ID,
		CommitHash: LocalCommitHash,
	}
	err = filepath.WalkDir(r.localDir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			// Unreadable entry (e.g. permissions): skip it rather than failing the whole scan.
			if d != nil && d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if p == r.localDir {
			return nil
		}
		if d.IsDir() {
			if localDirsToSkip[d.Name()] {
				return fs.SkipDir
			}
			return nil
		}

		rel, err := filepath.Rel(r.localDir, p)
		if err != nil {
			return nil
		}
		name := filepath.ToSlash(rel)

		// Use os.Stat (not d.Info()) so that symlinks (common in HuggingFace cache snapshot directories)
		// are resolved to the real file size.
		fi, statErr := os.Stat(p)
		if statErr != nil {
			// Broken symlink or similar: skip this file rather than failing the whole scan.
			return nil
		}
		if fi.IsDir() {
			return nil
		}

		info.Siblings = append(info.Siblings, &FileInfo{Name: name, Size: fi.Size()})
		return nil
	})
	if err != nil {
		return errors.Wrapf(err, "while scanning local model directory %q", r.localDir)
	}
	sort.Slice(info.Siblings, func(i, j int) bool { return info.Siblings[i].Name < info.Siblings[j].Name })
	r.info = info
	return nil
}

// localFiles resolves repoFiles (repository-relative paths, using "/" as separator) to paths inside
// r.localDir, for local-directory mode. It is the local-mode counterpart of the network-based
// Repo.DownloadFilesCtx.
func (r *Repo) localFiles(repoFiles ...string) ([]string, error) {
	paths := make([]string, len(repoFiles))
	for i, name := range repoFiles {
		rel := cleanRelativeFilePath(name)
		if rel == "." {
			return nil, errors.Errorf("invalid file name %q", name)
		}
		p := filepath.Join(r.localDir, rel)
		if !files.Exists(p) {
			return nil, errors.Errorf("file %q not found in local model directory %q", name, r.localDir)
		}
		paths[i] = p
	}
	return paths, nil
}
