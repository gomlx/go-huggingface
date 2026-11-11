package datasets

import (
	"context"
	"fmt"
	"os"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/go-huggingface/internal/files"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
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
	return d.DownloadCtx(context.Background(), files...)
}

// DownloadCtx downloads the given parquet files using the context ctx to manage interruption.
// It returns the list of absolute paths where the files were downloaded.
func (d *Dataset) DownloadCtx(ctx context.Context, parquetFiles ...ParquetFile) (downloadedPaths []string, err error) {
	if len(parquetFiles) == 0 {
		return nil, nil
	}

	downloadManager := d.GetDownloadManager()

	repoCacheDir, err := d.CacheDir()
	if err != nil {
		return nil, err
	}

	downloadedPaths = make([]string, len(parquetFiles))

	// Compute total bytes to download and list of files needing download.
	type pendingParquet struct {
		pf       ParquetFile
		destPath string
	}
	var toDownload []pendingParquet
	var totalBytesToDownload uint64
	var downloadList []string
	logFileList := d.Verbosity >= 2 || klog.V(1).Enabled()
	for idxFile, pf := range parquetFiles {
		destPath := path.Join(repoCacheDir, "parquet", pf.Config, pf.Split, pf.Filename)
		downloadedPaths[idxFile] = destPath
		if !files.Exists(destPath) {
			toDownload = append(toDownload, pendingParquet{pf: pf, destPath: destPath})
			if pf.Size > 0 {
				totalBytesToDownload += uint64(pf.Size)
			}
			if logFileList {
				downloadList = append(downloadList, fmt.Sprintf("  - %s (%s)", pf.Filename, humanize.Bytes(uint64(pf.Size))))
			}
		}
	}

	if len(toDownload) == 0 {
		return downloadedPaths, nil
	}

	if logFileList && len(downloadList) > 0 {
		klog.Infof("Downloading %d files (total %s):", len(downloadList), humanize.Bytes(totalBytesToDownload))
		for _, item := range downloadList {
			fmt.Println(item)
		}
	}

	ctx, cancelFn := context.WithCancel(ctx)
	defer cancelFn()

	var downloadingMu sync.Mutex
	var firstError error
	requireDownload := len(toDownload)
	perFileDownloaded := make([]uint64, requireDownload)
	perFileSizeRecorded := make([]bool, requireDownload)
	for i, item := range toDownload {
		if item.pf.Size > 0 {
			perFileSizeRecorded[i] = true
		}
	}
	var allFilesDownloaded uint64
	var numDownloadedFiles int
	busyLoop := `-\|/`
	busyLoopPos := 0
	var lastLineLen int
	lastPrintTime := time.Now()
	showProgress := d.ProgressBar()

	ratePrintFn := func() {
		currentFile := min(numDownloadedFiles+1, requireDownload)
		var sizeStr string
		if totalBytesToDownload > 0 {
			displayTotal := max(totalBytesToDownload, allFilesDownloaded)
			sizeStr = fmt.Sprintf("%s / %s", humanize.Bytes(allFilesDownloaded), humanize.Bytes(displayTotal))
		} else {
			sizeStr = humanize.Bytes(allFilesDownloaded)
		}
		var line string
		if firstError == nil {
			line = fmt.Sprintf("- Downloading file %d/%d %c (%s)",
				currentFile, requireDownload, busyLoop[busyLoopPos], sizeStr)
		} else {
			line = fmt.Sprintf("- Downloading file %d/%d (%s): error - %v",
				currentFile, requireDownload, sizeStr, firstError)
		}
		padded := line
		if len(padded) < lastLineLen {
			padded += strings.Repeat(" ", lastLineLen-len(padded))
		}
		lastLineLen = len(line)
		fmt.Printf("\r%s", padded)
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

	for _, item := range toDownload {
		dir, _ := path.Split(item.destPath)
		if err = os.MkdirAll(dir, 0777); err != nil {
			return nil, errors.Wrapf(err, "while creating directory to download %q", item.destPath)
		}
	}

	if showProgress {
		fmt.Printf("\nHugging Face dataset %q:\n", d.ID)
		ratePrintFn()
	}

	var wg sync.WaitGroup
	for idxTask, item := range toDownload {
		idxTask, item := idxTask, item
		wg.Go(func() {
			err := downloadManager.LockedDownload(ctx, item.pf.URL, item.destPath, false, func(downloadedBytes, totalBytes int64) {
				downloadingMu.Lock()
				defer downloadingMu.Unlock()
				if !perFileSizeRecorded[idxTask] && totalBytes > 0 {
					totalBytesToDownload += uint64(totalBytes)
					perFileSizeRecorded[idxTask] = true
				}
				lastReportedBytes := perFileDownloaded[idxTask]
				if uint64(downloadedBytes) > lastReportedBytes {
					allFilesDownloaded += uint64(downloadedBytes) - lastReportedBytes
					perFileDownloaded[idxTask] = uint64(downloadedBytes)
				}
				if showProgress && time.Since(lastPrintTime) >= 500*time.Millisecond {
					ratePrintFn()
				}
			})

			if err != nil {
				reportErrorFn(errors.WithMessagef(err, "while downloading %q", item.pf.URL))
				return
			}

			downloadingMu.Lock()
			numDownloadedFiles++
			if showProgress && numDownloadedFiles < requireDownload {
				ratePrintFn()
			}
			downloadingMu.Unlock()
		})
	}
	wg.Wait()

	if showProgress {
		if firstError != nil {
			ratePrintFn()
			fmt.Println()
		} else {
			filesWord := "files"
			if requireDownload == 1 {
				filesWord = "file"
			}
			finalBytes := allFilesDownloaded
			if finalBytes == 0 && totalBytesToDownload > 0 {
				finalBytes = totalBytesToDownload
			}
			line := fmt.Sprintf("- Downloaded %d/%d %s (%s)",
				numDownloadedFiles, requireDownload, filesWord, humanize.Bytes(finalBytes))
			padded := line
			if len(padded) < lastLineLen {
				padded += strings.Repeat(" ", lastLineLen-len(padded))
			}
			fmt.Printf("\r%s\n", padded)
		}
	}

	if firstError != nil {
		return nil, firstError
	}
	return downloadedPaths, nil
}

// DownloadAll for the given config/split. It will download all the files into the cache,
// which can then be iterated over (without any downloads).
//
// If config or split are empty, they are not used for filtering.
//
// It returns the list of the paths to the downloaded files in the cache.
func (ds *Dataset) DownloadAll(ctx context.Context, config, split string) (downloadedPaths []string, err error) {
	remoteFiles, err := ds.ListFiles(config, split)
	if err != nil {
		return nil, err
	}
	return ds.DownloadCtx(ctx, remoteFiles...)
}

// ListDownloadedFiles returns the list of absolute paths to the parquet files that match the given config and split
// and exist in the local cache.
// If config or split are empty, they are not used for filtering.
func (d *Dataset) ListDownloadedFiles(config, split string) ([]string, error) {
	repoCacheDir, err := d.CacheDir()
	if err != nil {
		return nil, err
	}
	allFiles, err := d.ListFiles(config, split)
	if err != nil {
		return nil, err
	}
	var res []string
	for _, f := range allFiles {
		destPath := path.Join(repoCacheDir, "parquet", f.Config, f.Split, f.Filename)
		if files.Exists(destPath) {
			res = append(res, destPath)
		}
	}
	return res, nil
}
