package hub

import (
	"context"
	"github.com/gomlx/go-huggingface/internal/downloader"
	"github.com/gomlx/go-huggingface/internal/files"
	"github.com/pkg/errors"
	"log"
	"math/rand"
	"os"
	"path"
	"github.com/gofrs/flock"
	"time"
)

// Generic download utilities.

// getDownloadManager returns current downloader.Manager, or creates a new one for this Repo.
func (r *Repo) getDownloadManager() *downloader.Manager {
	if r.downloadManager == nil {
		r.downloadManager = downloader.New().MaxParallel(r.MaxParallelDownload).WithAuthToken(r.authToken)
	}
	return r.downloadManager
}

// lockedDownload url to the given filePath.
//
// If filePath exits and forceDownload is false, it is assumed to already have been correctly downloaded, and it will return immediately.
//
// It downloads the file to filePath+".tmp" and then atomically move it to filePath.
//
// It uses a temporary filePath+".lock" to coordinate multiple processes/programs trying to download the same file at the same time.
func (r *Repo) lockedDownload(ctx context.Context, url, filePath string, forceDownload bool, progressCallback downloader.ProgressCallback) error {
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

	// Create directory for file.
	if err := os.MkdirAll(path.Dir(filePath), DefaultDirCreationPerm); err != nil {
		return errors.Wrapf(err, "failed to create directory for file %q", filePath)
	}

	// Lock file to avoid parallel downloads.
	lockPath := filePath + ".lock"
	var mainErr error
	errLock := execOnFileLock(lockPath, func() {
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

		downloadManager := r.getDownloadManager()
		mainErr = downloadManager.Download(ctx, url, tmpPath, progressCallback)
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

// execOnFileLock opens the lockPath file (or creates if it doesn't yet exist), locks it, and executes the function.
// If the lockPath is already locked, it polls with a 1 to 2 seconds period (randomly), until it acquires the lock.
//
// The lockPath is not removed. It's safe to remove it from the given fn, if one knows that no new calls to
// execOnFileLock with the same lockPath is going to be made.
func execOnFileLock(lockPath string, fn func()) (err error) {
	// Create a new flock instance directly using gofrs/flock
	fileLock := flock.New(lockPath)

	// Acquire lock with retry logic
	for {
		// Try to acquire the lock
		locked, err := fileLock.TryLock()
		if err != nil {
			return errors.Wrapf(err, "while trying to lock %q", lockPath)
		}

		// If we got the lock, break out of the retry loop
		if locked {
			break
		}

		// Wait from 1 to 2 seconds.
		time.Sleep(time.Millisecond * time.Duration(1000+rand.Intn(1000)))
	}

	// Setup clean up in a deferred function, so it happens even if `fn()` panics.
	defer func() {
		unlockErr := fileLock.Unlock()
		if unlockErr != nil {
			// If we already have an error, don't overwrite it
			if err == nil {
				err = errors.Wrapf(unlockErr, "unlocking file %q", lockPath)
			} else {
				log.Printf("Error unlocking file %q: %v", lockPath, unlockErr)
		}
		}
	}()

	// We got the lock, run the function.
	fn()

	return
}
