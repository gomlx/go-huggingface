package datasets

import (
	"fmt"
	"io"
	"os"
	"testing"

	"github.com/parquet-go/parquet-go"
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
	type PassagesItem struct {
		IsSelected  []int32  `json:"is_selected" parquet:"is_selected,list"`
		PassageText []string `json:"passage_text" parquet:"passage_text,list"`
		URL         []string `json:"url" parquet:"url,list"`
	}

	type MsMarcoRecord struct {
		Answers           []string     `json:"answers" parquet:"answers,list"`
		Passages          PassagesItem `json:"passages" parquet:"passages"`
		Query             string       `json:"query" parquet:"query"`
		QueryID           int32        `json:"query_id" parquet:"query_id"`
		QueryType         string       `json:"query_type" parquet:"query_type"`
		WellFormedAnswers []string     `json:"wellFormedAnswers" parquet:"wellFormedAnswers,list"`
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
	assert.Equal(t, 10, count) // There are only 10 records in the file, it should yield exactly 10.

	// Verify deprecated ParquetFixListSchema remains compatible with parquet.NewGenericReader.
	fixedSchema, err := ParquetFixListSchema[MsMarcoRecord]("ms_marco_v2.1_validation_10.parquet")
	require.NoError(t, err)
	f, err := os.Open("ms_marco_v2.1_validation_10.parquet")
	require.NoError(t, err)
	defer f.Close()
	reader := parquet.NewGenericReader[MsMarcoRecord](f, fixedSchema)
	defer reader.Close()
	rows := make([]MsMarcoRecord, 12)
	n, err := reader.Read(rows)
	if err != io.EOF {
		require.NoError(t, err)
	}
	assert.Equal(t, 10, n)
	assert.NotEmpty(t, rows[0].Query)
	assert.NotEmpty(t, rows[0].Answers)
	assert.NotEmpty(t, rows[0].Passages.PassageText)

	// Verify MultiRowGroup + NewGenericRowGroupReader + SeekToRow (as used by CreateParquetReader).
	stat, err := f.Stat()
	require.NoError(t, err)
	pf, err := parquet.OpenFile(f, stat.Size())
	require.NoError(t, err)
	multiGroup := parquet.MultiRowGroup(append(pf.RowGroups(), pf.RowGroups()...)...)
	rgReader := parquet.NewGenericRowGroupReader[MsMarcoRecord](multiGroup)
	defer rgReader.Close()
	assert.Equal(t, int64(20), rgReader.NumRows())
	require.NoError(t, rgReader.SeekToRow(12))
	seekRows := make([]MsMarcoRecord, 10)
	n, err = rgReader.Read(seekRows)
	if err != io.EOF {
		require.NoError(t, err)
	}
	assert.Equal(t, 8, n)
	assert.Equal(t, rows[2], seekRows[0])
}
