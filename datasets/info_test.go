package datasets

import (
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

	info := ds.Info()

	require.NotNil(t, info, "Dataset info should not be nil")
	require.NotNil(t, info.DatasetInfo, "DatasetInfo block should not be nil")

	// ms_marco has a "v1.1" configuration
	configInfo, ok := info.DatasetInfo["v1.1"]
	require.True(t, ok, "Expected 'v1.1' config in dataset info")

	require.NotEmpty(t, configInfo.Features, "Features should not be empty")

	fmt.Printf("Info for %q:\n%s\n", ds.ID, info)

	fmt.Printf("\nGenerated Struct for Config v1.1:\n%s", configInfo.GenerateGoStruct("MsMarcoRecord"))
}
