package datasets

import (
	"context"
	"fmt"
	"os"
	"path"
	"sync"
	"time"

	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/internal/files"
	"github.com/pkg/errors"
)

// ListFiles returns the parquet files that match the given config and split.
// If config or split are empty, they are not used for filtering.
//
// They are returned in the order they are listed in the dataset info.
func (d *Dataset) ListFiles(config, split string) ([]ParquetFile, error) {
	allFiles, err := d.GetParquetFiles()
	if err != nil {
		return nil, err
	}
	var res []ParquetFile
	for _, f := range allFiles {
		if (config == "" || f.Config == config) && (split == "" || f.Split == split) {
			res = append(res, f)
		}
	}
	return res, nil
}

// Download downloads the given parquet files.
// It returns the list of absolute paths where the files were downloaded.
func (d *Dataset) Download(files ...ParquetFile) ([]string, error) {
	return d.DownloadCtx(model.Background(), files...)
}

// DownloadCtx is like Download but accepts a context for cancellation support.
func (d *Dataset) DownloadCtx(ctx model.Context, parquetFiles ...ParquetFile) (downloadedPaths []string, err error) {
	if len(parquetFiles) == 0 {
		return nil, nil
	}

	downloadManager := d.GetDownloadManager()

	repoCacheDir, err := d.CacheDir()
	if err != nil {
		return nil, err
	}

	ctx, cancelFn := model.WithCancel(ctx)
	defer cancelFn()

	downloadedPaths = make([]string, len(parquetFiles))

	var downloadingMu sync.Mutex
	var firstError error
	var requireDownload int
	perFileDownloaded := make([]uint64, len(parquetFiles))
	var allFilesDownloaded uint64
	var numDownloadedFiles int
	busyLoop := `-\|/`
	busyLoopPos := 0
	lastPrintTime := time.Now()

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

	reportErrorFn := func(err error) {
		downloadingMu.Lock()
		if firstError == nil {
			firstError = err
		}
		cancelFn()
		downloadingMu.Unlock()
	}

	var wg sync.WaitGroup
	for idxFile, pf := range parquetFiles {
		// Calculate the destination path.
		// e.g. <cacheDir>/parquet/<config>/<split>/<filename>
		destPath := path.Join(repoCacheDir, "parquet", pf.Config, pf.Split, pf.Filename)
		downloadedPaths[idxFile] = destPath

		if files.Exists(destPath) {
			continue
		}

		dir, _ := path.Split(destPath)
		if err = os.MkdirAll(dir, 0777); err != nil {
			return nil, errors.Wrapf(err, "while creating directory to download %q", destPath)
		}

		wg.Add(1)
		go func(idx int, pf ParquetFile, destPath string) {
			defer wg.Done()

			requireDownload++
			err := downloadManager.LockedDownload(ctx, pf.URL, destPath, false, func(downloadedBytes, totalBytes int64) {
				downloadingMu.Lock()
				defer downloadingMu.Unlock()
				lastReportedBytes := perFileDownloaded[idx]
				newDownloaded := uint64(downloadedBytes) - lastReportedBytes
				allFilesDownloaded += newDownloaded
				perFileDownloaded[idx] = uint64(downloadedBytes)
				if d.Verbosity > 0 && time.Since(lastPrintTime) > time.Second {
					ratePrintFn()
				}
			})

			if err != nil {
				reportErrorFn(errors.WithMessagef(err, "while downloading %q", pf.URL))
				return
			}

			downloadingMu.Lock()
			numDownloadedFiles++
			if d.Verbosity > 0 {
				ratePrintFn()
			}
			downloadingMu.Unlock()
		}(idxFile, pf, destPath)
	}
	wg.Wait()

	if requireDownload > 0 {
		if d.Verbosity > 0 {
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
