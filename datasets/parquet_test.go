package datasets

import (
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type TestRecord struct {
	Query string `parquet:"query"`
}

func TestIterParquetFromFile(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.Skip("Skipping network tests in CI")
	}

	ds := New("microsoft/ms_marco")
	files, err := ds.ListFiles("v1.1", "test")
	require.NoError(t, err)
	require.NotEmpty(t, files)

	downloaded, err := ds.Download(files[0])
	require.NoError(t, err)
	require.NotEmpty(t, downloaded)

	count := 0
	for rec, err := range IterParquetFromFile[TestRecord](downloaded[0]) {
		require.NoError(t, err)
		assert.NotEmpty(t, rec.Query)
		count++
		if count >= 10 { // Just test the first 10 records to keep it fast
			break
		}
	}
	assert.Greater(t, count, 0)
}

func TestParquetFixListSchema(t *testing.T) {
	// Based on HuggingFace dataset "microsoft/ms_marco", config "v2.1", split "validation"
	type PassagesGroup struct {
		IsSelected  []int32  `parquet:"is_selected,list"`
		PassageText []string `parquet:"passage_text,list"`
		URL         []string `parquet:"url,list"`
	}

	type MsMarcoRecord struct {
		Answers           []string      `parquet:"answers,list"`
		Passages          PassagesGroup `parquet:"passages"`
		Query             string        `parquet:"query"`
		QueryID           int32         `parquet:"query_id"`
		QueryType         string        `parquet:"query_type"`
		WellFormedAnswers []string      `parquet:"wellFormedAnswers,list"`
	}

	count := 0
	limit := 12
	for record, err := range IterParquetFromFile[MsMarcoRecord]("ms_marco_v2.1_validation_10.parquet") {
		require.NoError(t, err)
		fmt.Printf("Record #%02d: %+v\n", count, record)
		assert.NotEmpty(t, record.Query)
		assert.NotEmpty(t, record.Answers)
		assert.NotEmpty(t, record.Answers[0])
		assert.NotEmpty(t, record.Passages)
		assert.NotEmpty(t, record.Passages.IsSelected)
		assert.NotEmpty(t, record.Passages.PassageText)
		assert.NotEmpty(t, record.Passages.PassageText[0])
		assert.NotEmpty(t, record.Passages.URL)
		assert.NotEmpty(t, record.Passages.URL[0])
		assert.NotEmpty(t, record.QueryID)
		assert.NotEmpty(t, record.QueryType)
		count++
		if count >= limit {
			break
		}
	}
	assert.Equal(t, count, 10) // There are only 10 records in the file, it should yield exactly 10.
}
