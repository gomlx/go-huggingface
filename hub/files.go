package hub

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/dustin/go-humanize"
	"github.com/gomlx/go-huggingface/internal/files"
	"github.com/pkg/errors"
)

// IterFileNames iterate over the file names stored in the repo.
// It doesn't trigger the downloading of the repo, only of the repo info.
func (r *Repo) IterFileNames() iter.Seq2[string, error] {
	// Download info and files.
	err := r.DownloadInfo(false)
	if err != nil {
		// Error downloading: yield error only.
		return func(yield func(string, error) bool) {
			yield("", err)
		}
	}
	return func(yield func(string, error) bool) {
		for _, si := range r.info.Siblings {
			fileName := si.Name
			if path.IsAbs(fileName) || strings.Contains(fileName, "..") {
				yield("", errors.Errorf("model %q contains illegal file name %q -- it cannot be an absolute path, nor contain \"..\"",
					r.ID, fileName))
				return
			}
			if !yield(fileName, nil) {
				return
			}
		}
	}
}

// HasFile returns whether the repo has given fileName.
// Notice fileName is relative to the repository, not in local disk.
//
// If the Repo hasn't downloaded its info yet, it attempts to download it here.
// If it fails, it simply return false.
// Call Repo.DownloadInfo to handle errors downloading the info.
func (r *Repo) HasFile(fileName string) bool {
	if r.DownloadInfo(false) != nil {
		return false
	}
	for _, si := range r.info.Siblings {
		if si.Name == fileName {
			return true
		}
	}
	return false
}

// cleanRelativeFilePath sanitizes a file path by removing empty segments
// and parent directory references ("..") for security reasons.
func cleanRelativeFilePath(repoFileName string) string {
	// Convert to forward slashes and clean the path
	normalized := filepath.ToSlash(repoFileName)

	// Remove leading slash if present
	normalized = strings.TrimPrefix(normalized, "/")

	// Split into path components
	parts := strings.Split(normalized, "/")

	// Process parts to handle ".." components
	var stack []string
	for _, part := range parts {
		if part == "" || part == "." {
			continue
		}
		if part == ".." {
			if len(stack) > 0 {
				// Remove last element if we have something to pop
				stack = stack[:len(stack)-1]
			}
			continue
		}
		stack = append(stack, part)
	}

	if len(stack) == 0 {
		return "."
	}

	// Join with platform-specific separator
	return filepath.FromSlash(strings.Join(stack, "/"))
}

// DownloadFiles downloads the repository files (the names returned by repo.IterFileNames), and return the path to the
// downloaded files in the cache structure.
//
// The returned downloadPaths can be read, but shouldn't be modified, since there may be other programs using the same
// files.
func (r *Repo) DownloadFiles(repoFiles ...string) (downloadedPaths []string, err error) {
	if len(repoFiles) == 0 {
		return nil, nil
	}

	// Create download manager, if one hasn't been created yet.
	downloadManager := r.getDownloadManager()

	// Get/create repoCacheDir.
	var repoCacheDir string
	repoCacheDir, err = r.repoCacheDir()
	if err != nil {
		return nil, err
	}
	_ = repoCacheDir

	// Get snapshot dir:
	snapshotDir, err := r.repoSnapshotsDir()
	if err != nil {
		return nil, err
	}

	// Create context to stop any downloading of files if any error occur.
	// The deferred cancel both cleans up the context, and also stops any pending/ongoing
	// transfer that may be happening if an error occurs and the function exits.
	ctx, cancelFn := context.WithCancel(context.Background())
	defer cancelFn()

	// Store results.
	downloadedPaths = make([]string, len(repoFiles))

	// Information about download progress, and firstError to report back if needed.
	var downloadingMu sync.Mutex
	var firstError error
	var requireDownload int // number of files that require download (and are not in cache yet).
	perFileDownloaded := make([]uint64, len(repoFiles))
	var allFilesDownloaded uint64
	var numDownloadedFiles int
	busyLoop := `-\|/`
	busyLoopPos := 0
	lastPrintTime := time.Now()

	// Print downloading progress.
	ratePrintFn := func() {
		if firstError == nil {
			fmt.Printf("\rDownloaded %d/%d files %c %s downloaded    ",
				numDownloadedFiles, requireDownload, busyLoop[busyLoopPos], humanize.Bytes(allFilesDownloaded))
		} else {
			fmt.Printf("\rDownloaded %d/%d files, %s downloaded: error - %v     ",
				numDownloadedFiles, requireDownload, humanize.Bytes(allFilesDownloaded),
				firstError)
		}
		busyLoopPos = (busyLoopPos + 1) % len(busyLoop)
		lastPrintTime = time.Now()
	}

	// Report error for a download, and interrupt everyone.
	reportErrorFn := func(err error) {
		downloadingMu.Lock()
		if firstError == nil {
			firstError = err
		}
		cancelFn()
		downloadingMu.Unlock()
	}

	// Loop over each file to download.
	var wg sync.WaitGroup
	for idxFile, repoFileName := range repoFiles {
		fileURL, err := r.FileURL(repoFileName)
		if err != nil {
			return nil, err
		}

		// Join the path parts of fileName using the current OS separator.
		relativeFilePath := cleanRelativeFilePath(repoFileName)
		if relativeFilePath == "." {
			return nil, errors.Errorf("invalid file name %q", repoFileName)
		}
		snapshotPath := path.Join(snapshotDir, relativeFilePath)
		downloadedPaths[idxFile] = snapshotPath // This is the file pointer we are returning.
		if files.Exists(snapshotPath) {
			// File already downloaded, skip.
			continue
		}

		// Create directory for this individual file.
		dir, _ := path.Split(snapshotPath)
		if err = os.MkdirAll(dir, DefaultDirCreationPerm); err != nil {
			return nil, errors.Wrapf(err, "while creating directory to download %q", snapshotPath)
		}

		// Start downloading in a separate goroutine.
		wg.Add(1)
		go func() {
			defer wg.Done()

			// Download header of file for safety checks, and so we can find the blobPath.
			header, contentLength, err := downloadManager.FetchHeader(ctx, fileURL)
			if err != nil {
				reportErrorFn(err)
				return
			}
			metadata := extractFileMetadata(header, fileURL, contentLength)
			etag := metadata.ETag
			if etag == "" {
				reportErrorFn(errors.Errorf("resource %q for %q doesn't have an ETag, not able to ensure reproduceability",
					repoFileName, r.ID))
				return
			}
			if metadata.Location != fileURL {
				// In the case of a redirect, remove authorization header when downloading blob
				reportErrorFn(errors.Errorf("resource %q for %q has a redirect from %q to %q: this can be unsafe if we send our authorization token to the new URL",
					repoFileName, r.ID, fileURL, metadata.Location))
				return
			}

			// blobPath: download only if it has already been downloaded.
			blobPath := path.Join(repoCacheDir, "blobs", etag)
			if !files.Exists(blobPath) {
				requireDownload++ // This file require download.
				err := r.lockedDownload(ctx, fileURL, blobPath, false, func(downloadedBytes, totalBytes int64) {
					// Execute at every report of download.
					downloadingMu.Lock()
					defer downloadingMu.Unlock()
					lastReportedBytes := perFileDownloaded[idxFile]
					newDownloaded := uint64(downloadedBytes) - lastReportedBytes
					allFilesDownloaded += newDownloaded
					perFileDownloaded[idxFile] = uint64(downloadedBytes)
					if r.Verbosity > 0 && time.Since(lastPrintTime) > time.Second {
						ratePrintFn()
					}
				})
				if err != nil {
					reportErrorFn(err)
					return
				}

				// Done, print out progress.
				numDownloadedFiles++
				if r.Verbosity > 0 {
					ratePrintFn()
				}
			}

			// Link blob file to snapshot.
			err = createSymLink(snapshotPath, blobPath)
			if err != nil {
				reportErrorFn(errors.WithMessagef(err, "while downloading %q from repository %q", repoFileName, r.ID))
			}
		}()
	}
	wg.Wait()
	if requireDownload > 0 {
		if r.Verbosity > 0 {
			if firstError != nil {
				fmt.Println()
			} else {
				fmt.Printf("\rDownloaded %d/%d files, %s downloaded         \n",
					numDownloadedFiles, requireDownload, humanize.Bytes(allFilesDownloaded))
			}
		}
	}
	if firstError != nil {
		return nil, firstError
	}
	return downloadedPaths, nil
}

// DownloadFile is a shortcut to DownloadFiles with only one file.
func (r *Repo) DownloadFile(file string) (downloadedPath string, err error) {
	res, err := r.DownloadFiles(file)
	if err != nil {
		return "", err
	}
	return res[0], nil
}

// fileMetadata used by HuggingFace Hub.
type fileMetadata struct {
	CommitHash, ETag, Location string
	Size                       int
}

func extractFileMetadata(header http.Header, url string, contentLength int64) (metadata fileMetadata) {
	metadata.CommitHash = header.Get(HeaderXRepoCommit)
	metadata.ETag = header.Get(HeaderXLinkedETag)
	if metadata.ETag == "" {
		metadata.ETag = header.Get("ETag")
	}
	metadata.ETag = removeQuotes(metadata.ETag)
	metadata.Location = header.Get("Location")
	if metadata.Location == "" {
		metadata.Location = url
	}

	if sizeStr := header.Get(HeaderXLinkedSize); sizeStr != "" {
		var err error
		metadata.Size, err = strconv.Atoi(sizeStr)
		if err != nil {
			metadata.Size = 0
		}
	}
	if metadata.Size == 0 {
		metadata.Size = int(contentLength)
	}
	return
}

func removeQuotes(str string) string {
	return strings.TrimRight(strings.TrimLeft(str, "\""), "\"")
}

// createSymlink creates a symbolic link named dst pointing to src, using a relative path if possible.
// It removes previous link/file if it already exists.
//
// We use relative paths because:
// * It's what `huggingface_hub` library does, and we want to keep things compatible.
// * If the cache folder is moved or backed up, links won't break.
// * Relative paths seem better handled on Windows -- although Windows is not yet fully supported for this package.
//
// Example layout:
//
//	└── [ 128]  snapshots
//	  ├── [ 128]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f
//	  │   ├── [  52]  README.md -> ../../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
//	  │   └── [  76]  pytorch_model.bin -> ../../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
func createSymLink(dst, src string) error {
	relLink, err := filepath.Rel(path.Dir(dst), src)
	if err != nil {
		relLink = src // Take the absolute path instead.
	}

	// Remove link/file if it already exists.
	err = os.Remove(dst)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return errors.Wrapf(err, "failed to remove dst=%q before linking it to %q", dst, relLink)
	}

	if err = os.Symlink(relLink, dst); err != nil {
		return errors.Wrapf(err, "while symlink'ing %q to %q using %q", src, dst, relLink)
	}
	return nil
}
