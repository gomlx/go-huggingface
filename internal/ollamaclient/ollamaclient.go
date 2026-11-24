// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package ollamaclient provides a lightweight, zero-dependency HTTP client for the
// Ollama REST API (https://github.com/ollama/ollama/blob/main/docs/api.md).
//
// It is intended for:
//  1. Verifying exact text generation parity against Ollama for a given model/prompt.
//  2. Benchmarking and profiling prefill/decode throughput and latency (both Ollama's
//     server-side nanosecond metrics and client-side wall-clock / TTFT measurements).
package ollamaclient

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// DefaultBaseURL is the default Ollama server address when neither baseURL nor OLLAMA_HOST is set.
const DefaultBaseURL = "http://localhost:11434"

// Client talks to an Ollama HTTP server.
type Client struct {
	baseURL    string
	httpClient *http.Client
}

// New creates a new Ollama Client.
// If baseURL is empty, it checks the OLLAMA_HOST environment variable, falling back to DefaultBaseURL.
// If the address omits an http:// or https:// scheme, http:// is prepended.
func New(baseURL string) *Client {
	if baseURL == "" {
		baseURL = os.Getenv("OLLAMA_HOST")
	}
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
		baseURL = "http://" + baseURL
	}
	baseURL = strings.TrimRight(baseURL, "/")
	return &Client{
		baseURL:    baseURL,
		httpClient: &http.Client{},
	}
}

// WithHTTPClient sets a custom *http.Client and returns c for chaining.
func (c *Client) WithHTTPClient(hc *http.Client) *Client {
	if hc != nil {
		c.httpClient = hc
	}
	return c
}

// BaseURL returns the normalized base URL of the Ollama server.
func (c *Client) BaseURL() string {
	return c.baseURL
}

// Options mirrors Ollama's runtime model options for /api/generate.
// Pointer fields are used so explicit zero values (such as Temperature = 0 for greedy decoding)
// are serialized rather than omitted.
type Options struct {
	NumKeep       *int     `json:"num_keep,omitempty"`
	Seed          *int     `json:"seed,omitempty"`
	NumPredict    *int     `json:"num_predict,omitempty"`
	TopK          *int     `json:"top_k,omitempty"`
	TopP          *float64 `json:"top_p,omitempty"`
	MinP          *float64 `json:"min_p,omitempty"`
	Temperature   *float64 `json:"temperature,omitempty"`
	RepeatPenalty *float64 `json:"repeat_penalty,omitempty"`
	NumCtx        *int     `json:"num_ctx,omitempty"`
	NumBatch      *int     `json:"num_batch,omitempty"`
	NumGPU        *int     `json:"num_gpu,omitempty"`
	NumThread     *int     `json:"num_thread,omitempty"`
	Stop          []string `json:"stop,omitempty"`
}

// GenerateRequest is the payload for POST /api/generate.
type GenerateRequest struct {
	Model     string   `json:"model"`
	Prompt    string   `json:"prompt"`
	Suffix    string   `json:"suffix,omitempty"`
	System    string   `json:"system,omitempty"`
	Template  string   `json:"template,omitempty"`
	Raw       bool     `json:"raw,omitempty"`
	Think     *bool    `json:"think,omitempty"`
	Stream    *bool    `json:"stream,omitempty"`
	KeepAlive string   `json:"keep_alive,omitempty"`
	Options   *Options `json:"options,omitempty"`
}

// GenerateResponse is the response from POST /api/generate.
// Duration fields from Ollama are nanosecond integers, which unmarshal directly into time.Duration.
type GenerateResponse struct {
	Model              string        `json:"model"`
	CreatedAt          string        `json:"created_at"`
	Response           string        `json:"response"`
	Thinking           string        `json:"thinking,omitempty"`
	Done               bool          `json:"done"`
	DoneReason         string        `json:"done_reason,omitempty"`
	TotalDuration      time.Duration `json:"total_duration"`
	LoadDuration       time.Duration `json:"load_duration"`
	PromptEvalCount    int           `json:"prompt_eval_count"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration"`
	EvalCount          int           `json:"eval_count"`
	EvalDuration       time.Duration `json:"eval_duration"`

	// Client-side timing measurements (not part of Ollama's JSON payload).
	WallDuration time.Duration `json:"-"`
	TTFT         time.Duration `json:"-"`
}

// PrefillTokensPerSec returns Ollama's reported prompt evaluation speed in tokens/sec.
func (r *GenerateResponse) PrefillTokensPerSec() float64 {
	if r == nil || r.PromptEvalCount <= 0 || r.PromptEvalDuration <= 0 {
		return 0
	}
	return float64(r.PromptEvalCount) / r.PromptEvalDuration.Seconds()
}

// DecodeTokensPerSec returns Ollama's reported token generation speed in tokens/sec.
func (r *GenerateResponse) DecodeTokensPerSec() float64 {
	if r == nil || r.EvalCount <= 0 || r.EvalDuration <= 0 {
		return 0
	}
	return float64(r.EvalCount) / r.EvalDuration.Seconds()
}

// ModelInfo describes a model returned by GET /api/tags.
type ModelInfo struct {
	Name       string `json:"name"`
	Model      string `json:"model"`
	ModifiedAt string `json:"modified_at"`
	Size       int64  `json:"size"`
	Digest     string `json:"digest"`
}

// Version checks connectivity to the Ollama server and returns its version string via GET /api/version.
func (c *Client) Version(ctx context.Context) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/version", nil)
	if err != nil {
		return "", fmt.Errorf("ollamaclient: creating version request: %w", err)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollamaclient: GET /api/version: %w", err)
	}
	defer resp.Body.Close()
	if err := checkHTTPStatus(resp); err != nil {
		return "", err
	}
	var payload struct {
		Version string `json:"version"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return "", fmt.Errorf("ollamaclient: decoding /api/version: %w", err)
	}
	return payload.Version, nil
}

// ListModels returns the list of models available on the Ollama server via GET /api/tags.
func (c *Client) ListModels(ctx context.Context) ([]ModelInfo, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/tags", nil)
	if err != nil {
		return nil, fmt.Errorf("ollamaclient: creating tags request: %w", err)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollamaclient: GET /api/tags: %w", err)
	}
	defer resp.Body.Close()
	if err := checkHTTPStatus(resp); err != nil {
		return nil, err
	}
	var payload struct {
		Models []ModelInfo `json:"models"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, fmt.Errorf("ollamaclient: decoding /api/tags: %w", err)
	}
	return payload.Models, nil
}

// Generate sends a non-streaming POST /api/generate request and returns the full response and timing metrics.
func (c *Client) Generate(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("ollamaclient: Generate request is nil")
	}
	payload := *req
	stream := false
	payload.Stream = &stream

	body, err := json.Marshal(&payload)
	if err != nil {
		return nil, fmt.Errorf("ollamaclient: marshaling generate request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollamaclient: creating generate request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollamaclient: POST /api/generate: %w", err)
	}
	defer resp.Body.Close()
	if err := checkHTTPStatus(resp); err != nil {
		return nil, err
	}

	var out GenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("ollamaclient: decoding generate response: %w", err)
	}
	out.WallDuration = time.Since(start)
	return &out, nil
}

// GenerateStream sends a streaming POST /api/generate request, invoking onChunk (if non-nil)
// for each NDJSON chunk returned by Ollama. If onChunk returns false, streaming stops early.
//
// The returned GenerateResponse contains the concatenated Response (and Thinking) text,
// Ollama's final server-side timing fields, and client-side TTFT and WallDuration.
func (c *Client) GenerateStream(ctx context.Context, req *GenerateRequest, onChunk func(chunk GenerateResponse) bool) (*GenerateResponse, error) {
	if req == nil {
		return nil, fmt.Errorf("ollamaclient: GenerateStream request is nil")
	}
	payload := *req
	stream := true
	payload.Stream = &stream

	body, err := json.Marshal(&payload)
	if err != nil {
		return nil, fmt.Errorf("ollamaclient: marshaling generate request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("ollamaclient: creating generate request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollamaclient: POST /api/generate: %w", err)
	}
	defer resp.Body.Close()
	if err := checkHTTPStatus(resp); err != nil {
		return nil, err
	}

	var (
		fullResponse strings.Builder
		fullThinking strings.Builder
		final        GenerateResponse
		gotFirst     bool
	)

	scanner := bufio.NewScanner(resp.Body)
	// Allow large JSON lines (e.g. if Ollama ever returns long chunks).
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var chunk GenerateResponse
		if err := json.Unmarshal(line, &chunk); err != nil {
			return nil, fmt.Errorf("ollamaclient: decoding stream chunk: %w", err)
		}
		if !gotFirst && (chunk.Response != "" || chunk.Thinking != "" || chunk.Done) {
			final.TTFT = time.Since(start)
			gotFirst = true
		}
		fullResponse.WriteString(chunk.Response)
		fullThinking.WriteString(chunk.Thinking)

		final.Model = chunk.Model
		final.CreatedAt = chunk.CreatedAt
		final.Done = chunk.Done
		final.DoneReason = chunk.DoneReason
		if chunk.Done {
			final.TotalDuration = chunk.TotalDuration
			final.LoadDuration = chunk.LoadDuration
			final.PromptEvalCount = chunk.PromptEvalCount
			final.PromptEvalDuration = chunk.PromptEvalDuration
			final.EvalCount = chunk.EvalCount
			final.EvalDuration = chunk.EvalDuration
		}

		if onChunk != nil && !onChunk(chunk) {
			break
		}
		if chunk.Done {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("ollamaclient: reading stream: %w", err)
	}

	final.Response = fullResponse.String()
	final.Thinking = fullThinking.String()
	final.WallDuration = time.Since(start)
	return &final, nil
}

func checkHTTPStatus(resp *http.Response) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	var apiErr struct {
		Error string `json:"error"`
	}
	if json.Unmarshal(bodyBytes, &apiErr) == nil && apiErr.Error != "" {
		return fmt.Errorf("ollamaclient: HTTP %d: %s", resp.StatusCode, apiErr.Error)
	}
	msg := strings.TrimSpace(string(bodyBytes))
	if msg == "" {
		msg = resp.Status
	}
	return fmt.Errorf("ollamaclient: HTTP %d: %s", resp.StatusCode, msg)
}
