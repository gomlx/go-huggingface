package hub

import (
	"github.com/gomlx/go-huggingface/internal/downloader"
)

// Generic download utilities.

// GetDownloadManager returns current downloader.Manager, or creates a new one for this Repo.
//
// Internal use only.
func (r *Repo) GetDownloadManager() *downloader.Manager {
	if r.downloadManager == nil {
		r.downloadManager = downloader.New().MaxParallel(r.MaxParallelDownload).WithAuthToken(r.authToken)
	}
	return r.downloadManager
}
