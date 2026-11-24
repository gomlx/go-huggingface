// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package ollamaclient

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestNewBaseURL(t *testing.T) {
	t.Run("DefaultAndEnv", func(t *testing.T) {
		t.Setenv("OLLAMA_HOST", "")
		require.Equal(t, DefaultBaseURL, New("").BaseURL())

		t.Setenv("OLLAMA_HOST", "127.0.0.1:11435/")
		require.Equal(t, "http://127.0.0.1:11435", New("").BaseURL())
	})

	t.Run("ExplicitOverridesEnv", func(t *testing.T) {
		t.Setenv("OLLAMA_HOST", "http://ignored:11434")
		require.Equal(t, "https://ollama.example.com", New("https://ollama.example.com/").BaseURL())
		require.Equal(t, "http://10.0.0.2:11434", New("10.0.0.2:11434").BaseURL())
	})
}

func TestVersionAndListModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/version":
			_, _ = w.Write([]byte(`{"version":"0.6.5"}`))
		case "/api/tags":
			_, _ = w.Write([]byte(`{"models":[{"name":"gemma4:e2b","model":"gemma4:e2b","size":4200000000}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	client := New(srv.URL)
	ctx := context.Background()

	ver, err := client.Version(ctx)
	require.NoError(t, err)
	require.Equal(t, "0.6.5", ver)

	models, err := client.ListModels(ctx)
	require.NoError(t, err)
	require.Len(t, models, 1)
	require.Equal(t, "gemma4:e2b", models[0].Name)
	require.Equal(t, int64(4200000000), models[0].Size)
}

func TestGenerate(t *testing.T) {
	var receivedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, http.MethodPost, r.Method)
		require.Equal(t, "/api/generate", r.URL.Path)
		require.NoError(t, json.NewDecoder(r.Body).Decode(&receivedBody))

		resp := map[string]any{
			"model":                "gemma4:e2b",
			"response":             "31 August 2000",
			"done":                 true,
			"done_reason":          "stop",
			"total_duration":       int64(250 * time.Millisecond),
			"load_duration":        int64(10 * time.Millisecond),
			"prompt_eval_count":    100,
			"prompt_eval_duration": int64(50 * time.Millisecond),
			"eval_count":           20,
			"eval_duration":        int64(200 * time.Millisecond),
		}
		require.NoError(t, json.NewEncoder(w).Encode(resp))
	}))
	defer srv.Close()

	client := New(srv.URL)
	temp := 0.0
	numPredict := 128
	think := false
	res, err := client.Generate(context.Background(), &GenerateRequest{
		Model:  "gemma4:e2b",
		Prompt: "When did Virgin Australia start operating?",
		Raw:    true,
		Think:  &think,
		Options: &Options{
			Temperature: &temp,
			NumPredict:  &numPredict,
		},
	})
	require.NoError(t, err)

	// Verify explicit zero temperature and stream=false were serialized.
	require.Equal(t, false, receivedBody["stream"])
	require.Equal(t, true, receivedBody["raw"])
	require.Equal(t, false, receivedBody["think"])
	opts, ok := receivedBody["options"].(map[string]any)
	require.True(t, ok)
	require.Contains(t, opts, "temperature")
	require.Equal(t, float64(0), opts["temperature"])
	require.Equal(t, float64(128), opts["num_predict"])

	// Verify decoded response and computed throughput.
	require.Equal(t, "31 August 2000", res.Response)
	require.True(t, res.Done)
	require.Equal(t, "stop", res.DoneReason)
	require.Equal(t, 100, res.PromptEvalCount)
	require.Equal(t, 50*time.Millisecond, res.PromptEvalDuration)
	require.InDelta(t, 2000.0, res.PrefillTokensPerSec(), 1e-6)
	require.Equal(t, 20, res.EvalCount)
	require.Equal(t, 200*time.Millisecond, res.EvalDuration)
	require.InDelta(t, 100.0, res.DecodeTokensPerSec(), 1e-6)
	require.Greater(t, res.WallDuration, time.Duration(0))
}

func TestGenerateStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/api/generate", r.URL.Path)
		flusher, _ := w.(http.Flusher)
		chunks := []string{
			`{"model":"gemma4:e2b","response":"Hello","done":false}`,
			`{"model":"gemma4:e2b","response":" world","done":false}`,
			`{"model":"gemma4:e2b","response":"!","done":true,"done_reason":"stop","prompt_eval_count":10,"prompt_eval_duration":20000000,"eval_count":3,"eval_duration":30000000}`,
		}
		for _, c := range chunks {
			_, _ = fmt.Fprintln(w, c)
			if flusher != nil {
				flusher.Flush()
			}
		}
	}))
	defer srv.Close()

	client := New(srv.URL)

	t.Run("FullStream", func(t *testing.T) {
		var seen []string
		res, err := client.GenerateStream(context.Background(), &GenerateRequest{
			Model:  "gemma4:e2b",
			Prompt: "Say hello",
		}, func(chunk GenerateResponse) bool {
			seen = append(seen, chunk.Response)
			return true
		})
		require.NoError(t, err)
		require.Equal(t, []string{"Hello", " world", "!"}, seen)
		require.Equal(t, "Hello world!", res.Response)
		require.True(t, res.Done)
		require.Equal(t, 10, res.PromptEvalCount)
		require.Equal(t, 3, res.EvalCount)
		require.InDelta(t, 100.0, res.DecodeTokensPerSec(), 1e-6)
		require.Greater(t, res.TTFT, time.Duration(0))
	})

	t.Run("EarlyStop", func(t *testing.T) {
		var parts []string
		res, err := client.GenerateStream(context.Background(), &GenerateRequest{
			Model:  "gemma4:e2b",
			Prompt: "Say hello",
		}, func(chunk GenerateResponse) bool {
			parts = append(parts, chunk.Response)
			return false
		})
		require.NoError(t, err)
		require.Equal(t, "Hello", strings.Join(parts, ""))
		require.Equal(t, "Hello", res.Response)
		require.False(t, res.Done)
	})
}

func TestHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"model 'missing' not found"}`))
	}))
	defer srv.Close()

	_, err := New(srv.URL).Generate(context.Background(), &GenerateRequest{Model: "missing", Prompt: "hi"})
	require.Error(t, err)
	require.Contains(t, err.Error(), "HTTP 404")
	require.Contains(t, err.Error(), "model 'missing' not found")
}
