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
		defer reader.Close()

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

// CreateParquetReader creates a parquet reader for the given dataset, config and split.
// This is more flexible than an iterator because it allows for random reads.
//
// It groups together all the files for the given config and split and creates a single reader for them.
func CreateParquetReader[T any](ds *Dataset, config, split string) (*parquet.GenericReader[T], error) {
	filesSelection, err := ds.ListFiles(config, split)
	if err != nil {
		return nil, errors.WithMessage(err, "failed to get parquet files info for dataset")
	}
	downloadedPaths, err := ds.DownloadCtx(context.Background(), filesSelection...)
	if err != nil {
		return nil, errors.WithMessage(err, "failed to download parquet files for dataset")
	}

	// Open each file and append them as "rowGroup"
	var rowGroups []parquet.RowGroup
	for _, filePath := range downloadedPaths {
		f, err := os.Open(filePath)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to open %q", filePath)
		}
		stat, err := f.Stat()
		if err != nil {
			return nil, errors.Wrapf(err, "failed to stat %q", filePath)
		}
		pf, err := parquet.OpenFile(f, stat.Size())
		if err != nil {
			return nil, errors.Wrapf(err, "failed to open parquet from %q", filePath)
		}
		rowGroups = append(rowGroups, pf.RowGroups()...)
	}
	multiGroup := parquet.MultiRowGroup(rowGroups...)
	reader := parquet.NewGenericRowGroupReader[T](multiGroup)
	return reader, nil
}

// IterParquetFromDataset downloads Parquet files associated with the dataset's config and split
// on demand (one by one) and iterates over all their records sequentially.
//
// If you want to pre-download all the parquet files for a dataset/config/split, use DownloadAll.
//
// It will yield an error and stop if there's an issue acquiring or reading the files.
func IterParquetFromDataset[T any](ds *Dataset, config, split string) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		filesSelection, err := ds.ListFiles(config, split)
		if err != nil {
			var zero T
			yield(zero, errors.WithMessage(err, "failed to get parquet files info for dataset"))
			return
		}

		for _, pf := range filesSelection {
			downloadedPaths, err := ds.DownloadCtx(context.Background(), pf)
			if err != nil {
				var zero T
				yield(zero, errors.WithMessage(err, "failed to download parquet file for dataset"))
				return
			}
			if len(downloadedPaths) == 0 {
				var zero T
				yield(zero, errors.New("no files downloaded"))
				return
			}
			filePath := downloadedPaths[0]
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

// IterParquetFromDatasetAt provides the same functionality as IterParquetFromDataset
// but starts at the given record.
//
// It uses CreateParquetReader to seek to the required position.
// It will yield an error and stop if there's an issue acquiring or reading the files.
func IterParquetFromDatasetAt[T any](ds *Dataset, config, split string, at int64) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		reader, err := CreateParquetReader[T](ds, config, split)
		if err != nil {
			var zero T
			yield(zero, errors.WithMessage(err, "failed to create parquet reader for dataset"))
			return
		}
		defer reader.Close()

		numRows := reader.NumRows()
		startRow := at

		if startRow > numRows {
			startRow = numRows
		}
		if startRow < 0 {
			startRow = 0
		}

		if startRow > 0 {
			if err := reader.SeekToRow(startRow); err != nil {
				var zero T
				yield(zero, errors.WithMessagef(err, "failed to seek to row %d", startRow))
				return
			}
		}

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
				yield(zero, errors.WithMessage(err, "error reading parquet dataset"))
				break
			}
		}
	}
}

// ParquetFixListSchema returns the Parquet schema derived from T.
//
// Deprecated: parquet-go now natively handles non-standard Parquet list schemas
// (such as PyArrow's "item", Athena's "bag"/"array_element", and legacy 2-level lists)
// during schema conversion in NewGenericReader and NewGenericRowGroupReader.
// Pass parquet.SchemaOf(new(T)) directly, or omit the schema argument altogether.
func ParquetFixListSchema[T any](_ string) (*parquet.Schema, error) {
	return parquet.SchemaOf(new(T)), nil
}
