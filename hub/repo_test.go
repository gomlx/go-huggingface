package hub

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReadCommitHashForRevision_ExactHash(t *testing.T) {
	exactHash := "c1899de289a04d12100db370d81485cdf75e47ca"
	repo := New("Qwen/Qwen3-0.6B").
		WithRevision(exactHash).
		WithEndpoint("http://invalid.unreachable.endpoint")

	hash, err := repo.readCommitHashForRevision()
	require.NoError(t, err)
	assert.Equal(t, exactHash, hash)
}

func TestReadCommitHashForRevision_CachedInfoWithoutNetwork(t *testing.T) {
	// Create a temporary cache directory with cached info for Qwen/Qwen3-0.6B revision "main".
	tempCacheDir, err := os.MkdirTemp("", "hf_cache_test")
	require.NoError(t, err)
	defer os.RemoveAll(tempCacheDir)

	repo := New("Qwen/Qwen3-0.6B").
		WithCacheDir(tempCacheDir).
		WithEndpoint("http://invalid.unreachable.endpoint")

	cacheDir, err := repo.repoCacheDir()
	require.NoError(t, err)

	infoDir := filepath.Join(cacheDir, "info")
	require.NoError(t, os.MkdirAll(infoDir, 0755))

	fixtureData, err := os.ReadFile("repo_test.json")
	require.NoError(t, err)

	infoFile := filepath.Join(infoDir, "main")
	require.NoError(t, os.WriteFile(infoFile, fixtureData, 0644))

	// readCommitHashForRevision should read from cache without issuing network requests.
	hash, err := repo.readCommitHashForRevision()
	require.NoError(t, err)
	assert.Equal(t, "c1899de289a04d12100db370d81485cdf75e47ca", hash)

	// repo.Info() should also be populated from the cached file.
	assert.NotNil(t, repo.info)
	assert.Equal(t, "c1899de289a04d12100db370d81485cdf75e47ca", repo.Info().CommitHash)
}

func TestDownloadInfo_ForceDownload_FailurePreservesCache(t *testing.T) {
	tempCacheDir, err := os.MkdirTemp("", "hf_cache_test")
	require.NoError(t, err)
	defer os.RemoveAll(tempCacheDir)

	repo := New("Qwen/Qwen3-0.6B").
		WithCacheDir(tempCacheDir).
		WithEndpoint("http://invalid.unreachable.endpoint")

	cacheDir, err := repo.repoCacheDir()
	require.NoError(t, err)

	infoDir := filepath.Join(cacheDir, "info")
	require.NoError(t, os.MkdirAll(infoDir, 0755))

	fixtureData, err := os.ReadFile("repo_test.json")
	require.NoError(t, err)

	infoFile := filepath.Join(infoDir, "main")
	require.NoError(t, os.WriteFile(infoFile, fixtureData, 0644))

	// Attempting a force-download with an invalid endpoint should fail, but must NOT delete the cached file.
	err = repo.DownloadInfo(true)
	assert.Error(t, err)

	assert.FileExists(t, infoFile)
	cachedData, err := os.ReadFile(infoFile)
	require.NoError(t, err)
	assert.Equal(t, string(fixtureData), string(cachedData))
}
