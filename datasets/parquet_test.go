package datasets

import (
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
