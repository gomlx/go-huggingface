package hub

import (
	"context"
	"io"
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
// tokenizer.json, ...), such as what one gets from `git clone` or using `hubinfo -save <dir> <repo>` (or
// if not installed, `go run github.com/gomlx/go-huggingface/cmd/hubinfo`) or `huggingface-cli download --local-dir`.
// It can also point directly at a snapshot directory inside an existing HuggingFace cache
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

// Save downloads/resolves all files in the repository and copies (or hard-links) them into dirPath,
// creating a local copy of the repository that can be loaded using NewLocal(dirPath).
//
// If linkOnly is true, hard links (os.Link) are created instead of copying the files. If dirPath is on a
// different filesystem/device than the HuggingFace cache, hard linking will return an error (cross-device link).
// Existing files in dirPath will be overwritten.
//
// Save fails if r is in local-directory mode (r.IsLocal() is true).
func (r *Repo) Save(dirPath string, linkOnly bool) error {
	return r.SaveCtx(context.Background(), dirPath, linkOnly)
}

// SaveCtx is like Save, but takes a context to allow cancellation during downloading/saving files.
func (r *Repo) SaveCtx(ctx context.Context, dirPath string, linkOnly bool) error {
	if r.IsLocal() {
		return errors.Errorf("cannot Save local repository %q (local dir %q)", r.ID, r.localDir)
	}
	if r.IsEmbed() {
		return errors.Errorf("cannot Save embedded repository %q", r.ID)
	}

	resolvedDir, err := files.ReplaceTildeInDir(dirPath)
	if err != nil {
		resolvedDir = dirPath
	}
	resolvedDir = filepath.Clean(resolvedDir)

	fileNames := make([]string, 0)
	for fileName, err := range r.IterFileNames() {
		if err != nil {
			return errors.Wrapf(err, "failed to get file list for repository %q", r.ID)
		}
		fileNames = append(fileNames, fileName)
	}

	cachedPaths, err := r.DownloadFilesCtx(ctx, fileNames...)
	if err != nil {
		return errors.Wrapf(err, "failed to download files for repository %q to save to %q", r.ID, resolvedDir)
	}

	for i, relPath := range fileNames {
		src := cachedPaths[i]
		rel := cleanRelativeFilePath(relPath)
		dst := filepath.Join(resolvedDir, rel)

		if err := os.MkdirAll(filepath.Dir(dst), DefaultDirCreationPerm); err != nil {
			return errors.Wrapf(err, "failed to create directory for %q", dst)
		}

		if files.Exists(dst) {
			if err := os.Remove(dst); err != nil {
				return errors.Wrapf(err, "failed to remove existing file %q before overwriting", dst)
			}
		}

		if linkOnly {
			if err := os.Link(src, dst); err != nil {
				return errors.Wrapf(err, "failed to create hard link from %q to %q", src, dst)
			}
		} else {
			if err := copyFile(src, dst); err != nil {
				return errors.Wrapf(err, "failed to copy file from %q to %q", src, dst)
			}
		}
	}
	return nil
}

// copyFile copies a single file from src to dst.
func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	if _, err := io.Copy(out, in); err != nil {
		return err
	}
	return out.Close()
}
