package files

import (
	"log"
	"math/rand"
	"time"

	"github.com/gofrs/flock"
	"github.com/pkg/errors"
)

// ExecOnFileLock opens the lockPath file (or creates if it doesn't yet exist), locks it, and executes the function.
// If the lockPath is already locked, it polls with a 1 to 2 seconds period (randomly), until it acquires the lock.
//
// The lockPath is not removed. It's safe to remove it from the given fn, if one knows that no new calls to
// ExecOnFileLock with the same lockPath is going to be made.
func ExecOnFileLock(lockPath string, fn func()) (err error) {
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
