package datasets

import (
	"context"
	"io"
	"iter"
	"os"
	"reflect"

	"github.com/parquet-go/parquet-go"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// IterParquetFromFile iterates over the records of a local Parquet file.
// Yields the records of type T mapped from the underlying parquet columns.
func IterParquetFromFile[T any](filePath string) iter.Seq2[T, error] {
	return func(yield func(T, error) bool) {
		// Fix the schema first
		fixedSchema, err := ParquetFixListSchema[T](filePath)
		if err != nil {
			var zero T
			yield(zero, err)
			return
		}
		if klog.V(1).Enabled() {
			klog.Infof("Fixed schema: %s\n\n", fixedSchema)
		}

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

// CreateParquetReader creates a parquet reader for the given dataset, config and split.
// This is more flexible than an iterator because it allows for random reads.
//
// It groups together all the files for the given config and split and creates a single reader for them.
// It will fix the schema to match the given type T.
func CreateParquetReader[T any](ds *Dataset, config, split string) (*parquet.GenericReader[T], error) {
	filesSelection, err := ds.ListFiles(config, split)
	if err != nil {
		return nil, errors.WithMessage(err, "failed to get parquet files info for dataset")
	}
	downloadedPaths, err := ds.DownloadCtx(context.Background(), filesSelection...)
	if err != nil {
		return nil, errors.WithMessage(err, "failed to download parquet files for dataset")
	}

	// Fix the schema first
	fixedSchema, err := ParquetFixListSchema[T](downloadedPaths[0])
	if err != nil {
		return nil, err
	}
	if klog.V(1).Enabled() {
		klog.Infof("Fixed schema: %s\n\n", fixedSchema)
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
	reader := parquet.NewGenericRowGroupReader[T](multiGroup, fixedSchema)
	return reader, nil
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

// ParquetFixListSchema recursively transforms a struct-based schema to match a file's naming.
func ParquetFixListSchema[T any](filePath string) (*parquet.Schema, error) {
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
			sRepeated := sFields[0]
			sElement := sRepeated.Fields()[0]

			fRepeated := fFields[0]
			fElement := fRepeated.Fields()[0]

			transformedElement := transformToMatchFile(sElement, fElement)

			// Create standard 3-level list structure but with standard names "list" and "element".
			listNode := parquet.List(transformedElement)

			// Wrap it to rename the internal fields to match what the file schema has.
			listNode = &customNamingNode{
				Node:         listNode,
				repeatedName: fRepeated.Name(),
				elementName:  fElement.Name(),
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

type customNamingNode struct {
	parquet.Node
	repeatedName string
	elementName  string
}

func (n *customNamingNode) Fields() []parquet.Field {
	fields := n.Node.Fields()
	if len(fields) == 0 {
		return fields
	}
	ret := make([]parquet.Field, len(fields))
	copy(ret, fields)
	ret[0] = &customNamingField{
		Field:             fields[0],
		overrideName:      n.repeatedName,
		overrideChildName: n.elementName,
	}
	return ret
}

type customNamingField struct {
	parquet.Field
	overrideName      string
	overrideChildName string
}

func (f *customNamingField) Name() string {
	if f.overrideName != "" {
		return f.overrideName
	}
	return f.Field.Name()
}

func (f *customNamingField) Fields() []parquet.Field {
	fields := f.Field.Fields()
	if f.overrideChildName != "" && len(fields) > 0 {
		ret := make([]parquet.Field, len(fields))
		copy(ret, fields)
		ret[0] = &customNamingField{
			Field:        fields[0],
			overrideName: f.overrideChildName,
		}
		return ret
	}
	return fields
}
