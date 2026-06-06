package datasets

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDatasetInfo(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.Skip("Skipping network tests in CI")
	}

	ds := New("microsoft/ms_marco")

	// Test creating the dataset properly inherited the Repo ID and Type
	assert.Equal(t, "microsoft/ms_marco", ds.ID)

	info, err := ds.Info()
	require.NoError(t, err)
	err = ds.DownloadParquetFilesInfo(context.Background(), false)
	require.NoError(t, err)

	require.NotNil(t, info, "Dataset info should not be nil")
	require.NotNil(t, info.DatasetInfo, "DatasetInfo block should not be nil")

	// ms_marco has a "v1.1" configuration
	configInfo, ok := info.DatasetInfo["v1.1"]
	require.True(t, ok, "Expected 'v1.1' config in dataset info")

	require.NotEmpty(t, configInfo.Features, "Features should not be empty")

	fmt.Printf("%s\n", ds)
	gen, _ := ds.GenerateGoStruct("v1.1", "test")
	fmt.Printf("\nGenerated Struct for Config v1.1:\n%s", gen)
}

func TestListFiles(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.Skip("Skipping network tests in CI")
	}

	ds := New("microsoft/ms_marco")
	files, err := ds.ListFiles("v1.1", "test")
	require.NoError(t, err)
	require.NotEmpty(t, files)

	for _, f := range files {
		assert.Equal(t, "v1.1", f.Config)
		assert.Equal(t, "test", f.Split)
		assert.NotEmpty(t, f.Filename)
		assert.NotEmpty(t, f.URL)
	}
}

func TestDatasetStringFineweb(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.Skip("Skipping network tests in CI")
	}

	ds := New("HuggingFaceFW/fineweb")
	desc := ds.String()
	require.NotEmpty(t, desc)
	assert.Contains(t, desc, "Dataset ID: HuggingFaceFW/fineweb")
	assert.Contains(t, desc, "sample-10BT")
}

func TestDatasetListFilesFineweb(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.Skip("Skipping network tests in CI")
	}

	ds := New("HuggingFaceFW/fineweb")
	files, err := ds.ListFiles("sample-10BT", "train")
	require.NoError(t, err)
	assert.NotEmpty(t, files, "Should find parquet files for sample-10BT config")
}

func TestListDownloadedFiles(t *testing.T) {
	if os.Getenv("CI") != "" {
		t.Skip("Skipping network tests in CI")
	}

	ds := New("microsoft/ms_marco")

	// Ensure the files are downloaded first.
	_, err := ds.DownloadAll(context.Background(), "v1.1", "test")
	require.NoError(t, err)

	downloaded, err := ds.ListDownloadedFiles("v1.1", "test")
	require.NoError(t, err)
	assert.NotEmpty(t, downloaded, "Should find downloaded files for config v1.1 and split test")
}

