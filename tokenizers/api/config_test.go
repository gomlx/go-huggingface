package api

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestTokensDecoderUnmarshalJSON(t *testing.T) {
	// Plain string tokens.
	cfg, err := ParseConfigContent([]byte(`{
		"bos_token": "<s>",
		"eos_token": "</s>",
		"unk_token": "<unk>"
	}`))
	require.NoError(t, err)
	assert.Equal(t, "<s>", cfg.BosToken.String())
	assert.Equal(t, "</s>", cfg.EosToken.String())
	assert.Equal(t, "<unk>", cfg.UnkToken.String())

	// Object tokens with "content" and extra AddedToken fields.
	cfgObj, err := ParseConfigContent([]byte(`{
		"bos_token": {"__type": "AddedToken", "content": "<s>", "lstrip": true, "normalized": true, "special": true},
		"eos_token": {"content": "</s>", "rstrip": true},
		"pad_token": "<pad>"
	}`))
	require.NoError(t, err)
	assert.Equal(t, "<s>", cfgObj.BosToken.String())
	assert.Equal(t, "AddedToken", cfgObj.BosToken.Type)
	assert.True(t, cfgObj.BosToken.Lstrip)
	assert.True(t, cfgObj.BosToken.Normalized)
	assert.True(t, cfgObj.BosToken.Special)
	assert.Equal(t, "</s>", cfgObj.EosToken.String())
	assert.True(t, cfgObj.EosToken.Rstrip)
	assert.Equal(t, "<pad>", cfgObj.PadToken.String())

	// Invalid JSON token type (e.g., array).
	_, err = ParseConfigContent([]byte(`{"bos_token": [1, 2, 3]}`))
	assert.Error(t, err)
}
