package datasets

import (
	"context"
	"fmt"
	"io"
	"iter"
	"os"
	"reflect"

	"github.com/parquet-go/parquet-go"
	"github.com/pkg/errors"
)

// IterParquetFromFile iterates over the records of a local Parquet file.
// Yields the records of type T mapped from the underlying parquet columns.
func IterParquetFromFile[T any](filePath string) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		// Fix the schema first
		fixedSchema, err := FixListSchema[T](filePath)
		if err != nil {
			var zero T
			yield(zero, err)
			return
		}
		fmt.Printf("Fixed schema: %s\n\n", fixedSchema)

		f, err := os.Open(filePath)
		var zero T
		if err != nil {
			yield(zero, errors.Wrapf(err, "failed to open %q", filePath))
			return
		}
		defer f.Close()
		reader := parquet.NewGenericReader[T](f, fixedSchema)

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

// IterParquetFromDataset downloads all Parquet files associated with the dataset's config and split
// and iterates over all their records sequentially.
// It will yield an error and stop if there's an issue acquiring or reading the files.
func IterParquetFromDataset[T any](ds *Dataset, config, split string) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		filesSelection, err := ds.ListFiles(config, split)
		if err != nil {
			var zero T
			yield(zero, errors.WithMessage(err, "failed to get parquet files info for dataset"))
			return
		}
		downloadedPaths, err := ds.DownloadCtx(context.Background(), filesSelection...)
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

// FixListSchema recursively transforms a struct-based schema to match a file's naming.
func FixListSchema[T any](filePath string) (*parquet.Schema, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return nil, err
	}

	pf, err := parquet.OpenFile(f, stat.Size())
	if err != nil {
		return nil, err
	}

	structSchema := parquet.SchemaOf(new(T))
	fixedRoot := transformToMatchFile(structSchema, pf.Schema())

	return parquet.NewSchema(structSchema.Name(), fixedRoot), nil
}

func transformToMatchFile(sNode, fNode parquet.Node) parquet.Node {
	if sNode.Leaf() {
		return sNode
	}

	// Detect if sNode is a LIST logical type (3-level structure)
	if lt := sNode.Type().LogicalType(); lt != nil && lt.List != nil {
		sFields := sNode.Fields()
		fFields := fNode.Fields()

		if len(sFields) > 0 && len(fFields) > 0 {
			// sRepeated is usually named "list"
			// sElement is usually named "element"
			sRepeated := sFields[0].(parquet.Node)
			sElement := sRepeated.Fields()[0].(parquet.Node)

			fRepeated := fFields[0].(parquet.Node)
			fElement := fRepeated.Fields()[0].(parquet.Node)

			// Recursively transform the element (e.g., if it's a list of structs)
			transformedElement := transformToMatchFile(sElement, fElement)

			// Use parquet.List to wrap the new element node with the file's names.
			// This returns a Node that knows it maps to a Go slice/array.
			listNode := parquet.List(parquet.Group{
				fElement.Name(): transformedElement,
			})

			// If the file used a name other than "list" for the repeated group:
			if fRepeated.Name() != "list" {
				listNode = parquet.Group{
					fRepeated.Name(): parquet.Repeated(parquet.Group{
						fElement.Name(): transformedElement,
					}),
				}
			}

			return &typedNode{Node: listNode, gotype: sNode.GoType()}
		}
	}

	// For standard groups, map children by name and preserve GoType
	fields := sNode.Fields()
	newGroup := parquet.Group{}
	fFieldMap := make(map[string]parquet.Node)
	for _, f := range fNode.Fields() {
		fFieldMap[f.Name()] = f.(parquet.Node)
	}

	for _, f := range fields {
		sChild := f.(parquet.Node)
		if fChild, ok := fFieldMap[f.Name()]; ok {
			newGroup[f.Name()] = transformToMatchFile(sChild, fChild)
		} else {
			newGroup[f.Name()] = sChild
		}
	}

	return &typedNode{Node: newGroup, gotype: sNode.GoType()}
}

type typedNode struct {
	parquet.Node
	gotype reflect.Type
}

func (t *typedNode) GoType() reflect.Type { return t.gotype }
