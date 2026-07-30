package hub

import (
	"embed"
	"io"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

//go:embed testdata/embed_repo/*
var testEmbedFS embed.FS

func TestNewEmbed(t *testing.T) {
	repo := NewEmbed(testEmbedFS, "testdata/embed_repo")
	assert.True(t, repo.IsEmbed())
	assert.False(t, repo.IsLocal())
	assert.Equal(t, "embed_repo", repo.ID)

	fsys, subDir := repo.EmbedFS()
	assert.NotNil(t, fsys)
	assert.Equal(t, "testdata/embed_repo", subDir)
}

func TestEmbedRepo_IterFileNamesAndInfos(t *testing.T) {
	repo := NewEmbed(testEmbedFS, "testdata/embed_repo")

	var names []string
	for name, err := range repo.IterFileNames() {
		require.NoError(t, err)
		names = append(names, name)
	}
	assert.ElementsMatch(t, []string{"config.json", "tokenizer.json", "subfolder/config.json"}, names)

	var infoNames []string
	var totalSize int64
	for info, err := range repo.IterFileInfos() {
		require.NoError(t, err)
		infoNames = append(infoNames, info.Name)
		totalSize += info.Size
	}
	assert.ElementsMatch(t, []string{"config.json", "tokenizer.json", "subfolder/config.json"}, infoNames)
	assert.Greater(t, totalSize, int64(0))
}

func TestEmbedRepo_OpenAndReadFile(t *testing.T) {
	repo := NewEmbed(testEmbedFS, "testdata/embed_repo")

	// Test ReadFile
	data, err := repo.ReadFile("config.json")
	require.NoError(t, err)
	assert.Contains(t, string(data), "bert")

	// Test Open
	f, err := repo.Open("subfolder/config.json")
	require.NoError(t, err)
	defer f.Close()

	buf, err := io.ReadAll(f)
	require.NoError(t, err)
	assert.Contains(t, string(buf), "subfolder_config")
}

func TestEmbedRepo_DownloadFileExtractsToTemp(t *testing.T) {
	repo := NewEmbed(testEmbedFS, "testdata/embed_repo")

	diskPath, err := repo.DownloadFile("config.json")
	require.NoError(t, err)
	assert.Contains(t, filepath.ToSlash(diskPath), "go-huggingface-embed")

	data, err := repo.ReadFile("config.json")
	require.NoError(t, err)
	assert.Contains(t, string(data), "bert")
}

func TestEmbedRepo_SaveAndDeleteCacheFail(t *testing.T) {
	repo := NewEmbed(testEmbedFS, "testdata/embed_repo")

	err := repo.Save(t.TempDir(), false)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot Save embedded repository")

	err = repo.DeleteCache()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot DeleteCache of an embedded repository")
}

func TestEmbedRepo_FetchFilesDoesNotExtractToDisk(t *testing.T) {
	repo := NewEmbed(testEmbedFS, "testdata/embed_repo")

	// FetchFiles should verify files exist without failing or extracting to disk
	err := repo.FetchFiles("config.json", "subfolder/config.json")
	require.NoError(t, err)

	err = repo.FetchFiles("non_existent_file.json")
	require.Error(t, err)
}

