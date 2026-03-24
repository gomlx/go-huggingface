package datasets

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/parquet-go/parquet-go"
	"github.com/pkg/errors"
)

// Acronyms is a map of common abbreviations that should be fully capitalized when generating Go structs.
// For example, "id" becomes "ID" instead of "Id".
var Acronyms = map[string]struct{}{
	"id":    {},
	"html":  {},
	"url":   {},
	"http":  {},
	"api":   {},
	"ascii": {},
	"css":   {},
	"dns":   {},
	"https": {},
	"json":  {},
	"ip":    {},
	"uri":   {},
	"tcp":   {},
}

// GenerateGoStructFromParquet reads the schema of a local Parquet file and generates Go source code
// defining the structures to hold its records.
func GenerateGoStructFromParquet(parquetFilePath, rootStructName string) (string, error) {
	f, err := os.Open(parquetFilePath)
	if err != nil {
		return "", errors.Wrapf(err, "failed to open %q", parquetFilePath)
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return "", errors.Wrapf(err, "failed to stat %q", parquetFilePath)
	}

	pf, err := parquet.OpenFile(f, stat.Size())
	if err != nil {
		return "", errors.Wrapf(err, "failed to open parquet file %q", parquetFilePath)
	}

	var defs []StructDef
	schema := pf.Schema()
	var mainFields []FieldDef
	for _, field := range schema.Fields() {
		mainFields = append(mainFields, resolveParquetField(field, &defs))
	}

	defs = append([]StructDef{{Name: rootStructName, Fields: mainFields}}, defs...)

	var buf bytes.Buffer
	for _, d := range defs {
		buf.WriteString(fmt.Sprintf("type %s struct {\n", d.Name))
		for _, f := range d.Fields {
			parquetTag := f.JSONName
			if f.IsList {
				parquetTag += ",list"
			}
			tags := []string{fmt.Sprintf(`json:"%s"`, f.JSONName), fmt.Sprintf(`parquet:"%s"`, parquetTag)}
			buf.WriteString(fmt.Sprintf("\t%s %s `%s`\n", f.GoName, f.GoType, strings.Join(tags, " ")))
		}
		buf.WriteString("}\n\n")
	}
	return buf.String(), nil
}

type FieldDef struct {
	GoName   string
	GoType   string
	JSONName string
	IsList   bool
}

type StructDef struct {
	Name   string
	Fields []FieldDef
}

func resolveParquetField(field parquet.Field, defs *[]StructDef) FieldDef {
	goName := toCamelCase(field.Name())
	jsonName := field.Name()
	var goType string
	var isList bool

	if field.Leaf() {
		goType = mapParquetTypeToGoType(field.Type().Kind())
	} else if field.Type().LogicalType() != nil && field.Type().LogicalType().List != nil {
		isList = true
		subfields := field.Fields()
		if len(subfields) > 0 {
			elementFields := subfields[0].Fields()
			if len(elementFields) > 0 {
				elementDef := resolveParquetField(elementFields[0], defs)
				goType = "[]" + elementDef.GoType
			} else {
				goType = "[]any"
			}
		} else {
			goType = "[]any"
		}
	} else {
		subStructName := goName + "Item"
		var fields []FieldDef
		for _, sub := range field.Fields() {
			fields = append(fields, resolveParquetField(sub, defs))
		}
		*defs = append(*defs, StructDef{Name: subStructName, Fields: fields})
		goType = subStructName
	}

	return FieldDef{
		GoName:   goName,
		GoType:   goType,
		JSONName: jsonName,
		IsList:   isList,
	}
}

func mapParquetTypeToGoType(kind parquet.Kind) string {
	switch kind {
	case parquet.Boolean:
		return "bool"
	case parquet.Int32:
		return "int32"
	case parquet.Int64:
		return "int64"
	case parquet.Float:
		return "float32"
	case parquet.Double:
		return "float64"
	case parquet.ByteArray, parquet.FixedLenByteArray:
		return "string"
	default:
		return "any"
	}
}

// GenerateGoStruct lists files for the given config and split, downloads one parquet file
// to inspect its schema, and returns Go source code defining the structures to hold its records.
func (d *Dataset) GenerateGoStruct(config, split string) (string, error) {
	filesSelection, err := d.ListFiles(config, split)
	if err != nil {
		return "", errors.WithMessage(err, "failed to get parquet files info for dataset")
	}
	if len(filesSelection) == 0 {
		return "", errors.New("no files found for the given config and split")
	}

	// Just download the first file to get the schema
	downloadedPaths, err := d.DownloadCtx(context.Background(), filesSelection[0])
	if err != nil {
		return "", errors.WithMessage(err, "failed to download parquet file for dataset")
	}
	if len(downloadedPaths) == 0 {
		return "", errors.New("no files downloaded")
	}

	parts := strings.Split(d.ID, "/")
	namePart := parts[len(parts)-1]
	rootStructName := toCamelCase(namePart) + "Record"

	return GenerateGoStructFromParquet(downloadedPaths[0], rootStructName)
}

func toCamelCase(s string) string {
	parts := strings.Split(s, "_")
	for i := range parts {
		if len(parts[i]) > 0 {
			lower := strings.ToLower(parts[i])
			if _, ok := Acronyms[lower]; ok {
				parts[i] = strings.ToUpper(parts[i])
			} else {
				parts[i] = strings.ToUpper(parts[i][:1]) + parts[i][1:]
			}
		}
	}
	return strings.Join(parts, "")
}
