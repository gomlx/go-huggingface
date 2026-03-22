package datasets

import (
	"bytes"
	"fmt"
	"sort"
	"strings"
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

// GenerateGoStruct returns Go source code defining the structures to hold records
// for this dataset configuration.
// It parses the ConfigInfo.Features into type-safe Go structs.
// It uses "json" and "parquet" mappings based on the configuration builder name.
func (c *ConfigInfo) GenerateGoStruct(rootStructName string) string {
	var defs []StructDef

	var mainFields []FieldDef
	var keys []string
	for k := range c.Features {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	for _, k := range keys {
		f := c.Features[k]
		mainFields = append(mainFields, FieldDef{
			GoName:   toCamelCase(k),
			GoType:   c.resolveType(k, f, &defs),
			JSONName: k,
			Parquet:  c.BuilderName == "parquet",
		})
	}

	// Insert main struct at the beginning
	defs = append([]StructDef{{Name: rootStructName, Fields: mainFields}}, defs...)

	var buf bytes.Buffer
	for _, d := range defs {
		buf.WriteString(fmt.Sprintf("type %s struct {\n", d.Name))
		for _, f := range d.Fields {
			tags := []string{fmt.Sprintf(`json:"%s"`, f.JSONName)}
			if f.Parquet {
				tags = append(tags, fmt.Sprintf(`parquet:"%s"`, f.JSONName))
			}
			buf.WriteString(fmt.Sprintf("\t%s %s `%s`\n", f.GoName, f.GoType, strings.Join(tags, " ")))
		}
		buf.WriteString("}\n\n")
	}
	return buf.String()
}

type FieldDef struct {
	GoName   string
	GoType   string
	JSONName string
	Parquet  bool
}

type StructDef struct {
	Name   string
	Fields []FieldDef
}

func (c *ConfigInfo) resolveType(featName string, feat Feature, defs *[]StructDef) string {
	if feat.Type == "Value" {
		return mapDTypeToGoType(feat.DType)
	} else if feat.Type == "ClassLabel" {
		return "int"
	} else if feat.Type == "Sequence" {
		if single, ok := feat.SubFeature[""]; ok && len(feat.SubFeature) == 1 {
			return "[]" + c.resolveType(featName, single, defs)
		} else if len(feat.SubFeature) > 0 {
			subStructName := toCamelCase(featName) + "Item"
			var fields []FieldDef
			var keys []string
			for k := range feat.SubFeature {
				keys = append(keys, k)
			}
			sort.Strings(keys)

			for _, k := range keys {
				f := feat.SubFeature[k]
				fields = append(fields, FieldDef{
					GoName:   toCamelCase(k),
					GoType:   c.resolveType(k, f, defs),
					JSONName: k,
					Parquet:  c.BuilderName == "parquet",
				})
			}
			*defs = append(*defs, StructDef{Name: subStructName, Fields: fields})
			return "[]" + subStructName
		}
	} else if feat.Type == "Translation" {
		return "map[string]string"
	}
	return "any"
}

func mapDTypeToGoType(dtype string) string {
	switch dtype {
	case "string", "large_string":
		return "string"
	case "int8":
		return "int8"
	case "int16":
		return "int16"
	case "int32":
		return "int32"
	case "int64":
		return "int64"
	case "float32":
		return "float32"
	case "float64":
		return "float64"
	case "bool":
		return "bool"
	default:
		return "any"
	}
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
