package downloader

import (
	"context"
	"log"
	"os"
	"path"

	"github.com/gomlx/go-huggingface/internal/files"
	"github.com/pkg/errors"
)

// LockedDownload downloads url to the given filePath using a lock file to coordinate parallel downloads.
//
// If filePath exits and forceDownload is false, it is assumed to already have been correctly downloaded, and it will return immediately.
//
// It downloads the file to filePath+".tmp" and then atomically move it to filePath.
//
// It uses a temporary filePath+".lock" to coordinate multiple processes/programs trying to download the same file at the same time.
func (m *Manager) LockedDownload(ctx context.Context, url, filePath string, forceDownload bool, progressCallback ProgressCallback) error {
	if files.Exists(filePath) {
		if !forceDownload {
			return nil
		}
		err := os.Remove(filePath)
		if err != nil {
			return errors.Wrapf(err, "failed to remove %q while force-downloading %q", filePath, url)
		}
	}

	// Checks whether context has already been cancelled, and exit immediately.
	if err := ctx.Err(); err != nil {
		return err
	}

	// Create a directory for the file.
	if err := os.MkdirAll(path.Dir(filePath), 0755); err != nil {
		return errors.Wrapf(err, "failed to create directory for file %q", filePath)
	}

	// Lock file to avoid parallel downloads.
	lockPath := filePath + ".lock"
	var mainErr error
	errLock := files.ExecOnFileLock(lockPath, func() {
		if files.Exists(filePath) {
			// Some concurrent other process (or goroutine) already downloaded the file.
			return
		}

		// Create tmpFile where to download.
		var tmpFileClosed bool
		tmpPath := filePath + ".downloading"
		tmpFile, err := os.Create(tmpPath)
		if err != nil {
			mainErr = errors.Wrapf(err, "creating temporary file for download in %q", tmpPath)
			return
		}
		defer func() {
			// If we exit with an error, make sure to close and remove unfinished temporary file.
			if !tmpFileClosed {
				err := tmpFile.Close()
				if err != nil {
					log.Printf("Failed closing temporary file %q: %v", tmpPath, err)
				}
				err = os.Remove(tmpPath)
				if err != nil {
					log.Printf("Failed removing temporary file %q: %v", tmpPath, err)
				}
			}
		}()

		mainErr = m.Download(ctx, url, tmpPath, progressCallback)
		if mainErr != nil {
			mainErr = errors.WithMessagef(mainErr, "while downloading %q to %q", url, tmpPath)
			return
		}

		// Download succeeded, move to our target location.
		tmpFileClosed = true
		if err := tmpFile.Close(); err != nil {
			mainErr = errors.Wrapf(err, "failed to close temporary download file %q", tmpPath)
			return
		}
		if err := os.Rename(tmpPath, filePath); err != nil {
			mainErr = errors.Wrapf(err, "failed to move downloaded file %q to %q", tmpPath, filePath)
			return
		}

		// File already exists, so we no longer need the lock file.
		err = os.Remove(lockPath)
		if err != nil {
			log.Printf("Warning: error removing lock file %q: %+v", lockPath, err)
		}
	})
	if mainErr != nil {
		return mainErr
	}
	if errLock != nil {
		return errors.WithMessagef(errLock, "while locking %q to download %q", lockPath, url)
	}
	return nil
}
