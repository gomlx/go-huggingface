package datasets

import (
	"context"
	"io"
	"iter"
	"os"

	"github.com/parquet-go/parquet-go"
	"github.com/pkg/errors"
)

// IterParquetFromFile iterates over the records of a local Parquet file.
// Yields the records of type T mapped from the underlying parquet columns.
func IterParquetFromFile[T any](filePath string) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		f, err := os.Open(filePath)
		var zero T
		if err != nil {
			yield(zero, errors.Wrapf(err, "failed to open %q", filePath))
			return
		}
		defer f.Close()

		reader := parquet.NewGenericReader[T](f)

		batch := make([]T, 100)

		for {
			n, err := reader.Read(batch)
			for i := range n {
				if !yield(batch[i], nil) {
					return
				}
			}
			if err == io.EOF {
				break
			}
			if err != nil {
				var zero T
				yield(zero, errors.Wrapf(err, "error reading parquet from %q", filePath))
				break
			}
		}
	}
}

// IterParquetFromDataset downloads all Parquet files associated with the dataset
// and iterates over all their records sequentially.
// It will yield an error and stop if there's an issue acquiring or reading the files.
func IterParquetFromDataset[T any](ds *Dataset) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		files, err := ds.GetParquetFiles()
		if err != nil {
			var zero T
			yield(zero, errors.WithMessage(err, "failed to get parquet files info for dataset"))
			return
		}

		downloadedPaths, err := ds.DownloadCtx(context.Background(), files...)
		if err != nil {
			var zero T
			yield(zero, errors.WithMessage(err, "failed to download parquet files for dataset"))
			return
		}

		for _, filePath := range downloadedPaths {
			for record, err := range IterParquetFromFile[T](filePath) {
				if !yield(record, err) {
					return
				}
				if err != nil {
					// Stop iterating if we encounter an error reading a file.
					return
				}
			}
		}
	}
}
