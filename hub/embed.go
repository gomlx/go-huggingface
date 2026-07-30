package hub

import (
	"context"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/pkg/errors"
)

// EmbedCommitHash is used as the synthetic RepoInfo.CommitHash for repositories in embedded-filesystem mode.
const EmbedCommitHash = "embed"

// NewEmbed creates a Repo that reads files directly from an in-memory or embedded filesystem (fs.FS),
// such as one created via Go's `//go:embed` directive.
//
// fsys is the filesystem (e.g. an `embed.FS` instance). subDir specifies the relative directory inside
// fsys where repository files reside; use "" or "." if the files are at the root of fsys.
//
// In embedded-filesystem mode, no network access is ever made: Repo.DownloadInfo scans fsys for files instead
// of querying HuggingFace, and Repo.Open / Repo.ReadFile stream file data directly from fsys without disk I/O.
//
// Potential Side-Effects & Semantics:
//   - Calling Repo.Open(fileName) or Repo.ReadFile(fileName) reads directly from memory (no disk usage).
//   - If legacy methods that require an OS file path (e.g. Repo.DownloadFile or Repo.DownloadFiles) are called,
//     the requested files will be extracted to a temporary directory under `os.TempDir()` and the local path returned.
//   - Repo.Save and Repo.DeleteCache will fail when called on an embedded repository.
//
// The returned Repo's ID defaults to subDir (or "embed" if subDir is empty/root); set Repo.ID explicitly if desired.
func NewEmbed(fsys fs.FS, subDir string) *Repo {
	r := New("")
	return r.WithEmbedFS(fsys, subDir)
}

// WithEmbedFS switches r to embedded-filesystem mode using fsys and subDir. See NewEmbed for semantics.
//
// Passing nil for fsys switches r back to normal (remote) mode.
func (r *Repo) WithEmbedFS(fsys fs.FS, subDir string) *Repo {
	if fsys == nil {
		r.embedFS = nil
		r.embedSubDir = ""
		return r
	}
	r.embedFS = fsys
	subDir = strings.TrimPrefix(filepath.ToSlash(subDir), "/")
	if subDir == "." {
		subDir = ""
	}
	r.embedSubDir = subDir

	// Invalidate cached info from previous configuration.
	r.info = nil
	r.revisionHashRefreshed = false
	if r.ID == "" {
		if subDir != "" {
			r.ID = path.Base(subDir)
		} else {
			r.ID = "embed"
		}
	}
	return r
}

// IsEmbed returns whether r is in embedded-filesystem mode. See NewEmbed.
func (r *Repo) IsEmbed() bool {
	return r.embedFS != nil
}

// EmbedFS returns the embedded filesystem and subdirectory used by r if r.IsEmbed() is true.
// Otherwise it returns nil, "".
func (r *Repo) EmbedFS() (fs.FS, string) {
	return r.embedFS, r.embedSubDir
}

// scanFSInfo builds a RepoInfo by walking r.embedFS, for embedded-filesystem mode.
func (r *Repo) scanFSInfo(forceRescan bool) error {
	if r.info != nil && !forceRescan {
		return nil
	}
	if r.embedFS == nil {
		return errors.Errorf("embedded filesystem is nil")
	}

	rootPath := "."
	if r.embedSubDir != "" {
		rootPath = r.embedSubDir
	}

	info := &RepoInfo{
		InternalID: r.ID,
		ID:         r.ID,
		ModelID:    r.ID,
		CommitHash: EmbedCommitHash,
	}

	err := fs.WalkDir(r.embedFS, rootPath, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if d != nil && d.IsDir() {
				return fs.SkipDir
			}
			return nil
		}
		if p == rootPath {
			return nil
		}
		if d.IsDir() {
			if localDirsToSkip[d.Name()] {
				return fs.SkipDir
			}
			return nil
		}

		rel := p
		if rootPath != "." {
			var relErr error
			rel, relErr = filepath.Rel(rootPath, p)
			if relErr != nil {
				return nil
			}
		}
		name := filepath.ToSlash(rel)

		fi, statErr := d.Info()
		if statErr != nil || fi.IsDir() {
			return nil
		}

		info.Siblings = append(info.Siblings, &FileInfo{Name: name, Size: fi.Size()})
		return nil
	})
	if err != nil {
		return errors.Wrapf(err, "while scanning embedded repository filesystem under %q", rootPath)
	}
	sort.Slice(info.Siblings, func(i, j int) bool { return info.Siblings[i].Name < info.Siblings[j].Name })
	r.info = info
	return nil
}

// fsOpenPath returns the full path inside r.embedFS for a given repository relative file name.
func (r *Repo) fsOpenPath(name string) string {
	rel := cleanRelativeFilePath(name)
	if r.embedSubDir != "" {
		return path.Join(r.embedSubDir, rel)
	}
	return rel
}

// Open opens the named file for reading. It returns an fs.File interface.
//
// Works across all repository modes:
//   - Embedded mode (IsEmbed): opens directly from memory/fs.FS without disk I/O.
//   - Local mode (IsLocal): opens the file directly from LocalDir().
//   - Remote mode: ensures the file is downloaded to cache and opens it.
func (r *Repo) Open(name string) (fs.File, error) {
	return r.OpenCtx(context.Background(), name)
}

// OpenCtx is like Open, but accepts a context for cancellation support when downloading remote files.
func (r *Repo) OpenCtx(ctx context.Context, name string) (fs.File, error) {
	if r.IsEmbed() {
		p := r.fsOpenPath(name)
		f, err := r.embedFS.Open(p)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to open embedded file %q (path %q)", name, p)
		}
		return f, nil
	}

	diskPath, err := r.DownloadFileCtx(ctx, name)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(diskPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open cached file %q at %q", name, diskPath)
	}
	return f, nil
}

// ReadFile reads the named file and returns its complete contents as a byte slice.
//
// Works across all repository modes without requiring temporary disk extraction in embedded mode.
func (r *Repo) ReadFile(name string) ([]byte, error) {
	return r.ReadFileCtx(context.Background(), name)
}

// ReadFileCtx is like ReadFile, but accepts a context for cancellation support when downloading remote files.
func (r *Repo) ReadFileCtx(ctx context.Context, name string) ([]byte, error) {
	f, err := r.OpenCtx(ctx, name)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read contents of %q from repository %q", name, r.ID)
	}
	return data, nil
}

// extractEmbedFiles extracts repository files from r.embedFS to a temporary directory under os.TempDir()
// and returns their local disk paths. Used as a fallback when DownloadFile(s) is called on an embedded Repo.
func (r *Repo) extractEmbedFiles(repoFiles ...string) ([]string, error) {
	tmpDir := filepath.Join(os.TempDir(), "go-huggingface-embed", r.ID)
	if err := os.MkdirAll(tmpDir, DefaultDirCreationPerm); err != nil {
		return nil, errors.Wrapf(err, "failed to create temporary extraction directory %q", tmpDir)
	}

	paths := make([]string, len(repoFiles))
	for i, name := range repoFiles {
		rel := cleanRelativeFilePath(name)
		if rel == "." {
			return nil, errors.Errorf("invalid file name %q", name)
		}

		dstPath := filepath.Join(tmpDir, rel)
		paths[i] = dstPath

		// If file already extracted in temp directory, check if size matches.
		fsPath := r.fsOpenPath(name)
		srcFile, err := r.embedFS.Open(fsPath)
		if err != nil {
			return nil, errors.Wrapf(err, "embedded file %q (path %q) not found", name, fsPath)
		}

		fi, err := srcFile.Stat()
		if err == nil && osStatMatches(dstPath, fi.Size()) {
			srcFile.Close()
			continue
		}

		if err := os.MkdirAll(filepath.Dir(dstPath), DefaultDirCreationPerm); err != nil {
			srcFile.Close()
			return nil, errors.Wrapf(err, "failed to create directory for %q", dstPath)
		}

		dstFile, err := os.Create(dstPath)
		if err != nil {
			srcFile.Close()
			return nil, errors.Wrapf(err, "failed to create temporary file %q", dstPath)
		}

		_, copyErr := io.Copy(dstFile, srcFile)
		srcFile.Close()
		dstFile.Close()

		if copyErr != nil {
			return nil, errors.Wrapf(err, "failed to extract embedded file %q to %q", name, dstPath)
		}
	}
	return paths, nil
}

// osStatMatches checks if a file at path exists and has size bytes.
func osStatMatches(path string, size int64) bool {
	st, err := os.Stat(path)
	return err == nil && st.Size() == size
}

// verifyEmbedFilesExist checks that all requested files exist in r.embedFS without extracting them to disk.
func (r *Repo) verifyEmbedFilesExist(repoFiles ...string) error {
	for _, name := range repoFiles {
		fsPath := r.fsOpenPath(name)
		f, err := r.embedFS.Open(fsPath)
		if err != nil {
			return errors.Wrapf(err, "embedded file %q (path %q) not found", name, fsPath)
		}
		f.Close()
	}
	return nil
}
