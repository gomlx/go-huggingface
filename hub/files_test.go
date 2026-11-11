package hub

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCleanRelativeFilePath(t *testing.T) {
	testCases := []struct {
		input    string
		expected string
	}{
		{"foo/bar", "foo/bar"},
		{"foo/../bar", "bar"},
		{"foo/./bar", "foo/bar"},
		{"/foo/bar", "foo/bar"},
		{"foo//bar", "foo/bar"},
		{"foo/bar/..", "foo"},
		{"../foo/bar", "foo/bar"},
		{"foo/../../../..", "."},
		{"foo/../../../bar", "bar"},
		{"", "."},
		{".", "."},
		{"..", "."},
	}

	for _, tc := range testCases {
		expected := filepath.FromSlash(tc.expected)
		got := cleanRelativeFilePath(tc.input)
		fmt.Printf("\tcleanRelativeFilePath(%q) = %q\n", tc.input, got)
		assert.Equal(t, expected, got)
	}
}

func newMockHubServer(t *testing.T, repoID, commitSHA string, fileContents map[string][]byte, reqCount *atomic.Int32) *httptest.Server {
	t.Helper()
	var siblings []*FileInfo
	for name, data := range fileContents {
		siblings = append(siblings, &FileInfo{
			Name: name,
			Size: int64(len(data)),
		})
	}
	infoPayload, err := json.Marshal(&RepoInfo{
		ID:         repoID,
		ModelID:    repoID,
		CommitHash: commitSHA,
		Siblings:   siblings,
	})
	require.NoError(t, err)

	infoPath := fmt.Sprintf("/api/models/%s/revision/main", repoID)
	resolvePrefix := fmt.Sprintf("/%s/resolve/%s/", repoID, commitSHA)

	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if reqCount != nil {
			reqCount.Add(1)
		}
		if r.URL.Path == infoPath {
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Content-Length", strconv.Itoa(len(infoPayload)))
			w.WriteHeader(http.StatusOK)
			if r.Method != http.MethodHead {
				_, _ = w.Write(infoPayload)
			}
			return
		}
		if strings.HasPrefix(r.URL.Path, resolvePrefix) {
			fileName := strings.TrimPrefix(r.URL.Path, resolvePrefix)
			data, ok := fileContents[fileName]
			if !ok {
				http.NotFound(w, r)
				return
			}
			etag := fmt.Sprintf("\"etag-%s\"", strings.ReplaceAll(fileName, "/", "_"))
			w.Header().Set("ETag", etag)
			w.Header().Set(HeaderXLinkedSize, strconv.Itoa(len(data)))
			w.Header().Set("Content-Length", strconv.Itoa(len(data)))
			w.WriteHeader(http.StatusOK)
			if r.Method != http.MethodHead {
				_, _ = w.Write(data)
			}
			return
		}
		http.NotFound(w, r)
	}))
}

func TestDownloadFilesProgressAndCaching(t *testing.T) {
	const (
		repoID    = "owner/test-model"
		commitSHA = "0123456789abcdef0123456789abcdef01234567"
	)
	shardNames := []string{
		"model-00001-of-00003.safetensors",
		"model-00002-of-00003.safetensors",
		"model-00003-of-00003.safetensors",
	}
	fileContents := map[string][]byte{
		shardNames[0]: bytes.Repeat([]byte("a"), 1024),
		shardNames[1]: bytes.Repeat([]byte("b"), 2048),
		shardNames[2]: bytes.Repeat([]byte("c"), 1024),
	}

	var reqCount atomic.Int32
	srv := newMockHubServer(t, repoID, commitSHA, fileContents, &reqCount)
	defer srv.Close()

	cacheDir := t.TempDir()
	var progressBuf bytes.Buffer
	repo := New(repoID).WithEndpoint(srv.URL).WithCacheDir(cacheDir).WithAuth("")
	repo.progressWriter = &progressBuf

	paths, err := repo.DownloadFiles(shardNames...)
	require.NoError(t, err)
	require.Len(t, paths, 3)
	for i, p := range paths {
		got, readErr := os.ReadFile(p)
		require.NoError(t, readErr)
		assert.Equal(t, fileContents[shardNames[i]], got)
	}

	out := progressBuf.String()
	assert.True(t, strings.HasPrefix(out, "\nHugging Face repo \"owner/test-model\":\n"), "unexpected progress output: %q", out)
	assert.Contains(t, out, "- Downloading file 1/3")
	assert.Contains(t, out, "/ 4.0 KiB)")
	assert.Contains(t, out, "- Downloaded 3/3 files (4.0 KiB)")
	assert.True(t, strings.HasSuffix(out, "\n"))
	assert.NotContains(t, out, "1/1")
	assert.NotContains(t, out, "0/1")

	// Second call when cached should make zero HTTP requests and print nothing.
	progressBuf.Reset()
	reqCount.Store(0)
	paths2, err := repo.DownloadFiles(shardNames...)
	require.NoError(t, err)
	assert.Equal(t, paths, paths2)
	assert.Equal(t, int32(0), reqCount.Load())
	assert.Empty(t, progressBuf.String())
}

func TestDownloadFilesWithProgressBarDisabled(t *testing.T) {
	const (
		repoID    = "owner/quiet-model"
		commitSHA = "abcdef0123456789abcdef0123456789abcdef01"
	)
	fileContents := map[string][]byte{
		"config.json": []byte(`{"model_type":"gemma2"}`),
	}
	srv := newMockHubServer(t, repoID, commitSHA, fileContents, nil)
	defer srv.Close()

	var progressBuf bytes.Buffer
	repo := New(repoID).WithEndpoint(srv.URL).WithCacheDir(t.TempDir()).WithAuth("").WithProgressBar(false)
	repo.progressWriter = &progressBuf

	paths, err := repo.DownloadFiles("config.json")
	require.NoError(t, err)
	require.Len(t, paths, 1)
	assert.Empty(t, progressBuf.String())
}

func TestDownloadFile404NoEmptySubdir(t *testing.T) {
	const (
		repoID    = "owner/no-pooling"
		commitSHA = "fedcba9876543210fedcba9876543210fedcba98"
	)
	fileContents := map[string][]byte{
		"config.json": []byte(`{"model_type":"gemma2"}`),
	}
	srv := newMockHubServer(t, repoID, commitSHA, fileContents, nil)
	defer srv.Close()

	cacheDir := t.TempDir()
	var progressBuf bytes.Buffer
	repo := New(repoID).WithEndpoint(srv.URL).WithCacheDir(cacheDir).WithAuth("")
	repo.progressWriter = &progressBuf

	_, err := repo.DownloadFile("1_Pooling/config.json")
	require.Error(t, err)
	assert.Empty(t, progressBuf.String())

	snapshotDir, snapErr := repo.repoSnapshotsDir()
	require.NoError(t, snapErr)
	_, statErr := os.Stat(filepath.Join(snapshotDir, "1_Pooling"))
	assert.True(t, os.IsNotExist(statErr), "failed 404 download should not leave empty subdirectory in snapshot")
}
