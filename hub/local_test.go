package hub

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// writeTestFile creates dir/relPath (creating parent directories as needed) with the given contents.
func writeTestFile(t *testing.T, dir, relPath, contents string) {
	t.Helper()
	full := filepath.Join(dir, filepath.FromSlash(relPath))
	require.NoError(t, os.MkdirAll(filepath.Dir(full), DefaultDirCreationPerm))
	require.NoError(t, os.WriteFile(full, []byte(contents), DefaultFileCreationPerm))
}

// snapshotDirEntries lists every path under dir, recursively, relative to dir.
func snapshotDirEntries(t *testing.T, dir string) []string {
	t.Helper()
	var got []string
	require.NoError(t, filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		require.NoError(t, err)
		if p == dir {
			return nil
		}
		rel, err := filepath.Rel(dir, p)
		require.NoError(t, err)
		got = append(got, filepath.ToSlash(rel))
		return nil
	}))
	return got
}

func newTestLocalRepo(t *testing.T) (dir string, repo *Repo) {
	t.Helper()
	dir = t.TempDir()
	writeTestFile(t, dir, "config.json", `{"model_type":"bert"}`)
	writeTestFile(t, dir, "tokenizer.json", `{}`)
	writeTestFile(t, dir, "tokenizer_config.json", `{}`)
	writeTestFile(t, dir, "1_Pooling/config.json", `{"pooling_mode_mean_tokens": true}`)
	writeTestFile(t, dir, ".git/objects/xx", "should be ignored")
	return dir, NewLocal(dir)
}

func TestNewLocal_IDDefaultsToBaseName(t *testing.T) {
	dir := t.TempDir()
	repo := NewLocal(dir)
	assert.Equal(t, filepath.Base(dir), repo.ID)
	assert.True(t, repo.IsLocal())
	assert.Equal(t, filepath.Clean(dir), repo.LocalDir())
}

func TestWithLocalDir_EmptyStringRevertsToRemoteMode(t *testing.T) {
	dir := t.TempDir()
	repo := New("some/model").WithLocalDir(dir)
	require.True(t, repo.IsLocal())
	repo = repo.WithLocalDir("")
	assert.False(t, repo.IsLocal())
	assert.Equal(t, "", repo.LocalDir())
}

func TestLocalRepo_IterFileNames_UsesForwardSlashesAndSkipsGit(t *testing.T) {
	_, repo := newTestLocalRepo(t)

	var names []string
	for name, err := range repo.IterFileNames() {
		require.NoError(t, err)
		names = append(names, name)
	}

	assert.ElementsMatch(t, []string{
		"config.json",
		"tokenizer.json",
		"tokenizer_config.json",
		"1_Pooling/config.json",
	}, names)
	for _, name := range names {
		assert.NotContains(t, name, `\`)
		assert.NotContains(t, name, ".git")
	}
}

func TestLocalRepo_IterFileInfos_SizesMatch(t *testing.T) {
	_, repo := newTestLocalRepo(t)

	sizes := make(map[string]int64)
	for fi, err := range repo.IterFileInfos() {
		require.NoError(t, err)
		sizes[fi.Name] = fi.Size
	}

	assert.Equal(t, int64(len(`{"model_type":"bert"}`)), sizes["config.json"])
	assert.Equal(t, int64(len(`{"pooling_mode_mean_tokens": true}`)), sizes["1_Pooling/config.json"])
}

func TestLocalRepo_HasFile(t *testing.T) {
	_, repo := newTestLocalRepo(t)

	assert.True(t, repo.HasFile("1_Pooling/config.json"))
	assert.True(t, repo.HasFile("config.json"))
	assert.False(t, repo.HasFile("nope.json"))
}

func TestLocalRepo_DownloadFile_Success(t *testing.T) {
	dir, repo := newTestLocalRepo(t)

	p, err := repo.DownloadFile("1_Pooling/config.json")
	require.NoError(t, err)
	assert.Equal(t, filepath.Join(dir, "1_Pooling", "config.json"), p)
	contents, err := os.ReadFile(p)
	require.NoError(t, err)
	assert.Equal(t, `{"pooling_mode_mean_tokens": true}`, string(contents))
}

func TestLocalRepo_DownloadFile_MissingFileReturnsError(t *testing.T) {
	dir, repo := newTestLocalRepo(t)

	_, err := repo.DownloadFile("missing.json")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "missing.json")
	assert.Contains(t, err.Error(), filepath.Base(dir))
}

func TestLocalRepo_DownloadFile_CannotEscapeLocalDir(t *testing.T) {
	dir, repo := newTestLocalRepo(t)
	// Sibling directory to dir, with a file that must not become visible through path traversal.
	parent := filepath.Dir(dir)
	writeTestFile(t, parent, "secret.txt", "should not be reachable")

	_, err := repo.DownloadFile("../secret.txt")
	require.Error(t, err)

	paths, err := repo.DownloadFiles("../../../../../../../etc/passwd")
	require.Error(t, err)
	assert.Nil(t, paths)
}

func TestLocalRepo_FileURL_ReturnsError(t *testing.T) {
	_, repo := newTestLocalRepo(t)

	_, err := repo.FileURL("config.json")
	require.Error(t, err)
}

func TestLocalRepo_DownloadInfo_MissingDirectoryReturnsError(t *testing.T) {
	repo := NewLocal(filepath.Join(t.TempDir(), "does-not-exist"))
	err := repo.DownloadInfo(false)
	require.Error(t, err)
}

func TestLocalRepo_ScanDoesNotMutateDirectory(t *testing.T) {
	dir, repo := newTestLocalRepo(t)
	before := snapshotDirEntries(t, dir)

	require.NoError(t, repo.DownloadInfo(false))
	_, err := repo.DownloadFile("config.json")
	require.NoError(t, err)
	_, err = repo.CacheDir()
	require.NoError(t, err)

	after := snapshotDirEntries(t, dir)
	assert.ElementsMatch(t, before, after)
}
