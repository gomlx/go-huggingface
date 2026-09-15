package downloader

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLockedDownload_ForceDownload_FailurePreservesExistingFile(t *testing.T) {
	// Server returns 500 error simulating failure
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	tempDir, err := os.MkdirTemp("", "locked_downloader_test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	targetFile := filepath.Join(tempDir, "testfile.txt")
	err = os.WriteFile(targetFile, []byte("initial content"), 0644)
	require.NoError(t, err)

	manager := New()
	err = manager.LockedDownload(context.Background(), server.URL, targetFile, true, nil)
	assert.Error(t, err)

	// Existing file must NOT have been deleted upon failed download
	assert.FileExists(t, targetFile)
	content, err := os.ReadFile(targetFile)
	require.NoError(t, err)
	assert.Equal(t, "initial content", string(content))
}

func TestLockedDownload_ForceDownload_SuccessUpdatesExistingFile(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("updated content"))
	}))
	defer server.Close()

	tempDir, err := os.MkdirTemp("", "locked_downloader_test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	targetFile := filepath.Join(tempDir, "testfile.txt")
	err = os.WriteFile(targetFile, []byte("initial content"), 0644)
	require.NoError(t, err)

	manager := New()
	err = manager.LockedDownload(context.Background(), server.URL, targetFile, true, nil)
	require.NoError(t, err)

	assert.FileExists(t, targetFile)
	content, err := os.ReadFile(targetFile)
	require.NoError(t, err)
	assert.Equal(t, "updated content", string(content))
}

func TestLockedDownload_NoForceDownload_SkipsIfFileExists(t *testing.T) {
	// Server URL is invalid / closed, which would fail if called
	tempDir, err := os.MkdirTemp("", "locked_downloader_test")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	targetFile := filepath.Join(tempDir, "testfile.txt")
	err = os.WriteFile(targetFile, []byte("cached content"), 0644)
	require.NoError(t, err)

	manager := New()
	// Should return nil immediately without attempting network request
	err = manager.LockedDownload(context.Background(), "http://invalid.local/nonexistent", targetFile, false, nil)
	require.NoError(t, err)

	content, err := os.ReadFile(targetFile)
	require.NoError(t, err)
	assert.Equal(t, "cached content", string(content))
}
