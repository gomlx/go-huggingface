package hub

import (
	"encoding/json"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRepoInfoUnmarshal(t *testing.T) {
	// Read repo.json from workspace root
	data, err := os.ReadFile("repo_test.json")
	require.NoError(t, err)

	var info RepoInfo
	err = json.Unmarshal(data, &info)
	require.NoError(t, err)

	// Assert basic fields
	assert.Equal(t, "680da718233834890aa01f51", info.InternalID)
	assert.Equal(t, "Qwen/Qwen3-0.6B", info.ID)
	assert.Equal(t, "Qwen/Qwen3-0.6B", info.ModelID)
	assert.False(t, info.Private)
	assert.Equal(t, "text-generation", info.PipelineTag)
	assert.Equal(t, "transformers", info.LibraryName)
	assert.Contains(t, info.Tags, "safetensors")
	assert.Equal(t, int64(21888310), info.Downloads)
	assert.Equal(t, int64(1295), info.Likes)
	assert.Equal(t, "Qwen", info.Author)
	assert.Equal(t, "c1899de289a04d12100db370d81485cdf75e47ca", info.CommitHash)
	assert.False(t, info.Gated.(bool))
	assert.False(t, info.Disabled)

	// Assert widget data
	require.Len(t, info.WidgetData, 4)
	assert.Equal(t, "Hi, what can you help me with?", info.WidgetData[0]["text"])

	// Assert config
	require.NotNil(t, info.Config)
	assert.Equal(t, []string{"Qwen3ForCausalLM"}, info.Config.Architectures)
	assert.Equal(t, "qwen3", info.Config.ModelType)
	require.NotNil(t, info.Config.TokenizerConfig)
	assert.Equal(t, "<|im_end|>", info.Config.TokenizerConfig["eos_token"])

	// Assert cardData
	require.NotNil(t, info.CardData)
	assert.Equal(t, "transformers", info.CardData.LibraryName)
	assert.Equal(t, "apache-2.0", info.CardData.License)
	assert.Equal(t, "https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/LICENSE", info.CardData.LicenseLink)
	assert.Equal(t, "text-generation", info.CardData.PipelineTag)
	assert.Equal(t, []any{"Qwen/Qwen3-0.6B-Base"}, info.CardData.BaseModel)

	// Assert transformersInfo
	require.NotNil(t, info.Transformers)
	assert.Equal(t, "AutoModelForCausalLM", info.Transformers.AutoModel)
	assert.Equal(t, "text-generation", info.Transformers.PipelineTag)
	assert.Equal(t, "AutoTokenizer", info.Transformers.Processor)

	// Assert spaces
	assert.Contains(t, info.Spaces, "k2-fsa/OmniVoice")

	// Assert dates
	expectedLastModified, err := time.Parse(time.RFC3339, "2025-07-26T03:46:27.000Z")
	require.NoError(t, err)
	assert.True(t, info.LastModified.Equal(expectedLastModified))

	expectedCreatedAt, err := time.Parse(time.RFC3339, "2025-04-27T03:40:08.000Z")
	require.NoError(t, err)
	assert.True(t, info.CreatedAt.Equal(expectedCreatedAt))

	// Assert safetensors
	assert.Equal(t, int64(751632384), info.SafeTensors.Total)
	assert.Equal(t, int64(751632384), info.SafeTensors.Parameters["BF16"])

	assert.Equal(t, int64(4522815806), info.UsedStorage)

	// Assert siblings
	require.NotEmpty(t, info.Siblings)
	var foundSafetensors bool
	for _, f := range info.Siblings {
		if f.Name == "model.safetensors" {
			foundSafetensors = true
			assert.Equal(t, int64(1503300328), f.Size)
			assert.Equal(t, "a0458bbfda764d86cf930900d5f0f933933da9a3", f.BlobID)
			require.NotNil(t, f.LFS)
			assert.Equal(t, "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b", f.LFS.SHA256)
			assert.Equal(t, int64(1503300328), f.LFS.Size)
			assert.Equal(t, int64(135), f.LFS.PointerSize)
		}
	}
	assert.True(t, foundSafetensors)
}

func TestRepoInfoLegacyModelID(t *testing.T) {
	// Test legacy unmarshaling where model_id is used instead of modelId
	jsonData := `{"id":"foo","model_id":"legacy_model"}`
	var info RepoInfo
	err := json.Unmarshal([]byte(jsonData), &info)
	require.NoError(t, err)
	assert.Equal(t, "legacy_model", info.ModelID)
}

// Hugging Face dataset cards often set license as a YAML list, which the Hub API
// returns as a JSON array (e.g. bigcode/the-stack-v2 → ["other"], openai/gsm8k → ["mit"]).
// CardData.License stays a string for callers; arrays are joined with ", ".
func TestRepoInfoUnmarshal_LicenseAsArray(t *testing.T) {
	jsonData := `{
		"id": "bigcode/the-stack-v2",
		"sha": "e565caa3a78c2423bd374333a472b049eb090e47",
		"siblings": [{"rfilename": "README.md"}],
		"cardData": {
			"license": ["other"],
			"library_name": "datasets",
			"pipeline_tag": "text-generation",
			"pretty_name": "The Stack v2"
		}
	}`
	var info RepoInfo
	err := json.Unmarshal([]byte(jsonData), &info)
	require.NoError(t, err, "cardData.license as JSON array must unmarshal (Hub datasets often use list licenses)")
	require.NotNil(t, info.CardData)
	assert.Equal(t, "other", info.CardData.License)
	assert.Equal(t, "datasets", info.CardData.LibraryName)
	assert.Equal(t, "text-generation", info.CardData.PipelineTag)
	assert.Equal(t, "e565caa3a78c2423bd374333a472b049eb090e47", info.CommitHash)
}

func TestRepoInfoUnmarshal_LicenseAsString(t *testing.T) {
	jsonData := `{"id":"demo","cardData":{"license":"apache-2.0"}}`
	var info RepoInfo
	require.NoError(t, json.Unmarshal([]byte(jsonData), &info))
	require.NotNil(t, info.CardData)
	assert.Equal(t, "apache-2.0", info.CardData.License)
}

func TestRepoInfoUnmarshal_LicenseAsMultiValueArray(t *testing.T) {
	jsonData := `{"id":"demo","cardData":{"license":["cc-by-sa-3.0","gfdl"]}}`
	var info RepoInfo
	require.NoError(t, json.Unmarshal([]byte(jsonData), &info))
	require.NotNil(t, info.CardData)
	assert.Equal(t, "cc-by-sa-3.0, gfdl", info.CardData.License)
}

func TestRepoInfoUnmarshal_LicenseMissingOrNullOrEmpty(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		json string
	}{
		{name: "missing", json: `{"id":"demo","cardData":{"library_name":"datasets"}}`},
		{name: "null", json: `{"id":"demo","cardData":{"license":null}}`},
		{name: "empty array", json: `{"id":"demo","cardData":{"license":[]}}`},
		{name: "empty string", json: `{"id":"demo","cardData":{"license":""}}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var info RepoInfo
			require.NoError(t, json.Unmarshal([]byte(tc.json), &info))
			require.NotNil(t, info.CardData)
			assert.Equal(t, "", info.CardData.License)
		})
	}
}

func TestRepoInfoUnmarshal_LicenseInvalidType(t *testing.T) {
	t.Parallel()
	cases := []string{
		`{"id":"demo","cardData":{"license":123}}`,
		`{"id":"demo","cardData":{"license":true}}`,
		`{"id":"demo","cardData":{"license":{"name":"mit"}}}`,
	}
	for _, jsonData := range cases {
		var info RepoInfo
		err := json.Unmarshal([]byte(jsonData), &info)
		require.Error(t, err, "expected error for %s", jsonData)
		assert.Contains(t, err.Error(), "cardData.license")
	}
}

func TestCardDataUnmarshal_Null(t *testing.T) {
	t.Parallel()
	var c CardData
	require.NoError(t, json.Unmarshal([]byte("null"), &c))
	assert.Equal(t, CardData{}, c)
}

func TestRepoInfoUnmarshal_CardDataNull(t *testing.T) {
	t.Parallel()
	var info RepoInfo
	require.NoError(t, json.Unmarshal([]byte(`{"id":"demo","cardData":null}`), &info))
	assert.Equal(t, "demo", info.ID)
	assert.Nil(t, info.CardData)
}

func TestDownloadInfo_DatasetLicenseArray(t *testing.T) {
	// Live Hub call: openai/gsm8k has cardData.license = ["mit"] and is public/ungated.
	if testing.Short() {
		t.Skip("skipping live Hub call in short mode")
	}
	t.Parallel()
	dir := t.TempDir()
	repo := New("openai/gsm8k").
		WithType(RepoTypeDataset).
		WithCacheDir(dir).
		WithProgressBar(false)
	repo.Verbosity = 0

	err := repo.DownloadInfo(false)
	require.NoError(t, err, "DownloadInfo must succeed for datasets whose card license is a JSON array")
	info := repo.Info()
	require.NotNil(t, info)
	require.NotEmpty(t, info.CommitHash)
	require.NotNil(t, info.CardData)
	assert.Equal(t, "mit", info.CardData.License)
}
