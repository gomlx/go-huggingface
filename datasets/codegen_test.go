package datasets

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGenerateGoStruct(t *testing.T) {
	generated, err := GenerateGoStructFromParquet("ms_marco_v2.1_validation_10.parquet", "MsMarcoRecord")
	require.NoError(t, err)

	expected1 := "type MsMarcoRecord struct {\n" +
		"\tAnswers []string `json:\"answers\" parquet:\"answers,list\"`\n" +
		"\tPassages PassagesItem `json:\"passages\" parquet:\"passages\"`\n" +
		"\tQuery string `json:\"query\" parquet:\"query\"`\n" +
		"\tQueryID int32 `json:\"query_id\" parquet:\"query_id\"`\n" +
		"\tQueryType string `json:\"query_type\" parquet:\"query_type\"`\n" +
		"\tWellFormedAnswers []string `json:\"wellFormedAnswers\" parquet:\"wellFormedAnswers,list\"`\n" +
		"}"

	expected2 := "type PassagesItem struct {\n" +
		"\tIsSelected []int32 `json:\"is_selected\" parquet:\"is_selected,list\"`\n" +
		"\tPassageText []string `json:\"passage_text\" parquet:\"passage_text,list\"`\n" +
		"\tURL []string `json:\"url\" parquet:\"url,list\"`\n" +
		"}"

	// Ensure the builder included both structs precisely as formatted
	assert.True(t, strings.Contains(generated, expected1), "Missing or mismatched MsMarcoRecord struct definition\nGenerated output:\n%s", generated)
	assert.True(t, strings.Contains(generated, expected2), "Missing or mismatched PassagesItem struct definition\nGenerated output:\n%s", generated)
}
