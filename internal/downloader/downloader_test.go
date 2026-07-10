package downloader

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDownload_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("hello world"))
	}))
	defer server.Close()

	tempDir, err := os.MkdirTemp("", "downloader_test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	targetFile := filepath.Join(tempDir, "testfile.txt")
	manager := New()

	err = manager.Download(context.Background(), server.URL, targetFile, nil)
	require.NoError(t, err)

	// Check if the final file exists
	assert.FileExists(t, targetFile)

	// Check if content matches
	content, err := os.ReadFile(targetFile)
	require.NoError(t, err)
	assert.Equal(t, "hello world", string(content))

	// Check that the temporary .part file does not exist
	assert.NoFileExists(t, targetFile+"."+Part)
}

func TestDownload_Interrupted(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.(http.Flusher).Flush()
		time.Sleep(100 * time.Millisecond)
		_, _ = w.Write([]byte("some data"))
	}))
	defer server.Close()

	tempDir, err := os.MkdirTemp("", "downloader_test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	targetFile := filepath.Join(tempDir, "testfile.txt")
	manager := New()

	ctx, cancel := context.WithCancel(context.Background())
	// Cancel the context concurrently
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	err = manager.Download(ctx, server.URL, targetFile, nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "cancelled")

	// Final file should NOT exist
	assert.NoFileExists(t, targetFile)

	// Temporary .part file should NOT exist because it got cleaned up
	assert.NoFileExists(t, targetFile+"."+Part)
}
