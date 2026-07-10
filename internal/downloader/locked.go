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

		// Ensure the lock file is always cleaned up on exit (success or failure)
		defer func() {
			err := os.Remove(lockPath)
			if err != nil && !os.IsNotExist(err) {
				log.Printf("Warning: error removing lock file %q: %+v", lockPath, err)
			}
		}()

		mainErr = m.Download(ctx, url, filePath, progressCallback)
		if mainErr != nil {
			mainErr = errors.WithMessagef(mainErr, "while downloading %q to %q", url, filePath)
			return
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
