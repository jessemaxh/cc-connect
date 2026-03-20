package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// llmClient calls an OpenAI-compatible /v1/chat/completions endpoint.
// It supports streaming (SSE) and returns chunks via a channel.
type llmClient struct {
	baseURL string
	apiKey  string
	http    *http.Client
}

func newLLMClient(baseURL, apiKey string) *llmClient {
	return &llmClient{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		http:    &http.Client{Timeout: 5 * time.Minute},
	}
}

// ---- request/response types ----

type chatMessage struct {
	Role       string     `json:"role"`
	Content    any        `json:"content"` // string or []contentPart
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	Name       string     `json:"name,omitempty"`
}

type contentPart struct {
	Type     string    `json:"type"`              // "text" | "image_url"
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

type imageURL struct {
	URL string `json:"url"`
}

type toolCall struct {
	Index    int          `json:"index,omitempty"` // stream delta index; used for accumulation
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"` // "function"
	Function toolCallFunc `json:"function"`
}

type toolCallFunc struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"` // JSON string
}

type toolDef struct {
	Type     string      `json:"type"` // "function"
	Function toolFuncDef `json:"function"`
}

type toolFuncDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type completionRequest struct {
	Model        string        `json:"model"`
	Messages     []chatMessage `json:"messages"`
	Tools        []toolDef     `json:"tools,omitempty"`
	Stream       bool          `json:"stream"`
	MaxTokens    int           `json:"max_tokens,omitempty"`
	Temperature  *float64      `json:"temperature,omitempty"`
	SystemPrompt string        `json:"-"` // injected as first message
}

// ---- streaming delta types ----

type streamChunk struct {
	Choices []streamChoice `json:"choices"`
	Usage   *usageData     `json:"usage,omitempty"`
}

type streamChoice struct {
	Delta        streamDelta `json:"delta"`
	FinishReason string      `json:"finish_reason"`
}

type streamDelta struct {
	Role      string     `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []toolCall `json:"tool_calls,omitempty"`
}

type usageData struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ---- streaming result ----

// streamResult is one event returned to the session from a streaming call.
type streamResult struct {
	text      string
	toolCalls []toolCall
	usage     *usageData
	err       error
	done      bool
}

// complete sends a streaming chat completion and returns results via the channel.
// The channel is closed when the stream ends or an error occurs.
func (c *llmClient) complete(ctx context.Context, req completionRequest) <-chan streamResult {
	ch := make(chan streamResult, 32)
	go func() {
		defer close(ch)
		if err := c.streamRequest(ctx, req, ch); err != nil {
			// Use select so we don't block forever if the caller abandoned
			// the channel (e.g. broke out of the range loop on error).
			select {
			case ch <- streamResult{err: err, done: true}:
			case <-ctx.Done():
			}
		}
	}()
	return ch
}

func (c *llmClient) streamRequest(ctx context.Context, req completionRequest, ch chan<- streamResult) error {
	// Build messages: prepend system prompt if set.
	messages := req.Messages
	if req.SystemPrompt != "" {
		sys := chatMessage{Role: "system", Content: req.SystemPrompt}
		messages = append([]chatMessage{sys}, messages...)
	}

	payload := map[string]any{
		"model":    req.Model,
		"messages": messages,
		"stream":   true,
	}
	if len(req.Tools) > 0 {
		payload["tools"] = req.Tools
	}
	if req.MaxTokens > 0 {
		payload["max_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		payload["temperature"] = *req.Temperature
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		c.baseURL+"/v1/chat/completions",
		bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("api error %d: %s", resp.StatusCode, strings.TrimSpace(string(errBody)))
	}

	return c.parseSSEStream(resp.Body, ch)
}

func (c *llmClient) parseSSEStream(body io.Reader, ch chan<- streamResult) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 2*1024*1024)

	// tcAcc accumulates tool call deltas keyed by their stream index.
	// OpenAI sends tool calls as index-keyed partial chunks; we must accumulate
	// by the index field (not position in slice) to handle interleaved deltas.
	tcAcc := make(map[int]*toolCall)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			// Flush any remaining accumulated tool calls.
			if len(tcAcc) > 0 {
				ch <- streamResult{toolCalls: flattenToolCalls(tcAcc)}
			}
			ch <- streamResult{done: true}
			return nil
		}

		var chunk streamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			slog.Debug("api llm: skip non-chunk SSE data", "data", data[:min(len(data), 80)])
			continue
		}

		for _, choice := range chunk.Choices {
			delta := choice.Delta

			// Text content.
			if delta.Content != "" {
				ch <- streamResult{text: delta.Content}
			}

			// Accumulate tool call deltas by index (OpenAI streaming protocol).
			// Each chunk element has an index field; the first chunk for a given
			// index has an ID, subsequent chunks append to arguments.
			for _, tc := range delta.ToolCalls {
				if existing, ok := tcAcc[tc.Index]; ok {
					// Append delta to existing entry.
					existing.Function.Arguments += tc.Function.Arguments
					// Fill in any fields that arrive late.
					if tc.ID != "" && existing.ID == "" {
						existing.ID = tc.ID
					}
					if tc.Type != "" && existing.Type == "" {
						existing.Type = tc.Type
					}
					if tc.Function.Name != "" && existing.Function.Name == "" {
						existing.Function.Name = tc.Function.Name
					}
				} else {
					// New tool call at this index.
					entry := tc
					tcAcc[tc.Index] = &entry
				}
			}

			switch choice.FinishReason {
			case "tool_calls":
				if len(tcAcc) > 0 {
					ch <- streamResult{toolCalls: flattenToolCalls(tcAcc)}
					tcAcc = make(map[int]*toolCall)
				}
			case "stop":
				// Text-only finish; any tool calls should have been flushed already.
			}
		}

		if chunk.Usage != nil {
			ch <- streamResult{usage: chunk.Usage}
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read SSE stream: %w", err)
	}
	return nil
}

func flattenToolCalls(acc map[int]*toolCall) []toolCall {
	// Use append (not index assignment) because stream indices may not be
	// contiguous (e.g. 0 and 5), which would cause an out-of-bounds panic
	// if we assumed len(acc) == max(index)+1.
	calls := make([]toolCall, 0, len(acc))
	for _, v := range acc {
		calls = append(calls, *v)
	}
	return calls
}
