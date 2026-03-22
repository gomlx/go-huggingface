package datasets

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGenerateGoStruct(t *testing.T) {
	config := ConfigInfo{
		BuilderName: "parquet",
		Features: map[string]Feature{
			"query": {
				Type:  "Value",
				DType: "string",
			},
			"query_id": {
				Type:  "Value",
				DType: "int32",
			},
			"passages": {
				Type: "Sequence",
				SubFeature: map[string]Feature{
					"is_selected": {
						Type:  "Value",
						DType: "int32",
					},
					"passage_text": {
						Type:  "Value",
						DType: "string",
					},
					"url": {
						Type:  "Value",
						DType: "string",
					},
				},
			},
			"answers": {
				Type: "Sequence",
				SubFeature: map[string]Feature{
					"": {
						Type:  "Value",
						DType: "string",
					},
				},
			},
		},
	}

	generated := config.GenerateGoStruct("MsMarcoRecord")

	expected1 := "type MsMarcoRecord struct {\n" +
		"\tAnswers []string `json:\"answers\" parquet:\"answers\"`\n" +
		"\tPassages []PassagesItem `json:\"passages\" parquet:\"passages\"`\n" +
		"\tQuery string `json:\"query\" parquet:\"query\"`\n" +
		"\tQueryId int32 `json:\"query_id\" parquet:\"query_id\"`\n" +
		"}"

	expected2 := "type PassagesItem struct {\n" +
		"\tIsSelected int32 `json:\"is_selected\" parquet:\"is_selected\"`\n" +
		"\tPassageText string `json:\"passage_text\" parquet:\"passage_text\"`\n" +
		"\tUrl string `json:\"url\" parquet:\"url\"`\n" +
		"}"

	// Ensure the builder included both structs precisely as formatted
	assert.True(t, strings.Contains(generated, expected1), "Missing or mismatched MsMarcoRecord struct definition\nGenerated output:\n%s", generated)
	assert.True(t, strings.Contains(generated, expected2), "Missing or mismatched PassagesItem struct definition")
}

func TestGenerateGoStructNoParquet(t *testing.T) {
	config := ConfigInfo{
		BuilderName: "csv",
		Features: map[string]Feature{
			"query_id": {
				Type:  "Value",
				DType: "int32",
			},
		},
	}

	generated := config.GenerateGoStruct("SimpleRecord")
	
	expected := "type SimpleRecord struct {\n" +
		"\tQueryId int32 `json:\"query_id\"`\n" +
		"}"

	assert.True(t, strings.Contains(generated, expected), "Expected json-only annotations on non-parquet builders")
}
