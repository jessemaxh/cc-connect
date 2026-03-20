package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"
	"time"
)

const (
	// sseMaxBodyBytes caps the total SSE response body to prevent memory exhaustion.
	sseMaxBodyBytes = 10 * 1024 * 1024 // 10 MiB
	// sseMaxLineBytes caps a single SSE data line.
	sseMaxLineBytes = 1 * 1024 * 1024 // 1 MiB
)

// sseTransport implements MCP over HTTP + Server-Sent Events.
// POST {url}/message → send JSON-RPC request, receive response.
// The response may be Content-Type: application/json or text/event-stream.
//
// SECURITY: cfg.URL must be operator-supplied (not user-controllable) and is
// validated to use http:// or https:// only to prevent SSRF.
type sseTransport struct {
	cfg    ServerConfig
	client *http.Client
	nextID atomic.Int64
}

func newSSETransport(cfg ServerConfig) (*sseTransport, error) {
	if cfg.URL == "" {
		return nil, fmt.Errorf("sse transport: URL is required")
	}
	// Validate URL scheme to prevent SSRF against internal network addresses.
	u, err := url.Parse(cfg.URL)
	if err != nil {
		return nil, fmt.Errorf("sse transport: invalid URL %q: %w", cfg.URL, err)
	}
	if u.Scheme != "https" && u.Scheme != "http" {
		return nil, fmt.Errorf("sse transport: URL scheme must be http or https, got %q", u.Scheme)
	}
	return &sseTransport{
		cfg:    cfg,
		client: &http.Client{Timeout: 30 * time.Second},
	}, nil
}

func (t *sseTransport) post(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := t.nextID.Add(1)
	req := jsonrpcRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST",
		strings.TrimRight(t.cfg.URL, "/")+"/message",
		bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json, text/event-stream")

	resp, err := t.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http post: %w", err)
	}
	defer resp.Body.Close()

	ct := resp.Header.Get("Content-Type")
	switch {
	case strings.Contains(ct, "application/json"):
		var rpcResp jsonrpcResponse
		if err := json.NewDecoder(io.LimitReader(resp.Body, sseMaxBodyBytes)).Decode(&rpcResp); err != nil {
			return nil, fmt.Errorf("decode json response: %w", err)
		}
		if rpcResp.Error != nil {
			return nil, rpcResp.Error
		}
		return rpcResp.Result, nil

	case strings.Contains(ct, "text/event-stream"):
		return t.parseSSEResponse(resp.Body)

	default:
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("http %d", resp.StatusCode)
		}
		// Try JSON anyway.
		var rpcResp jsonrpcResponse
		if err := json.NewDecoder(io.LimitReader(resp.Body, sseMaxBodyBytes)).Decode(&rpcResp); err != nil {
			return nil, fmt.Errorf("decode response (unknown ct=%s): %w", ct, err)
		}
		if rpcResp.Error != nil {
			return nil, rpcResp.Error
		}
		return rpcResp.Result, nil
	}
}

// parseSSEResponse reads an SSE stream and returns the first "message" event data.
func (t *sseTransport) parseSSEResponse(body io.Reader) (json.RawMessage, error) {
	// Limit total bytes read to prevent memory exhaustion from a misbehaving server.
	limited := io.LimitReader(body, sseMaxBodyBytes)
	scanner := bufio.NewScanner(limited)
	scanner.Buffer(make([]byte, 0, 64*1024), sseMaxLineBytes)

	var dataLine string
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data:") {
			dataLine = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		} else if line == "" && dataLine != "" {
			// End of event — parse it.
			var rpcResp jsonrpcResponse
			if err := json.Unmarshal([]byte(dataLine), &rpcResp); err != nil {
				return nil, fmt.Errorf("decode SSE data: %w", err)
			}
			if rpcResp.Error != nil {
				return nil, rpcResp.Error
			}
			return rpcResp.Result, nil
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read SSE: %w", err)
	}
	return nil, fmt.Errorf("SSE stream ended without a message event")
}

func (t *sseTransport) Initialize(ctx context.Context) (*InitializeResult, error) {
	params := InitializeParams{
		ProtocolVersion: "2024-11-05",
		ClientInfo:      ClientInfo{Name: "cc-connect", Version: "1.0"},
		Capabilities:    map[string]any{},
	}
	raw, err := t.post(ctx, "initialize", params)
	if err != nil {
		return nil, err
	}
	var result InitializeResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("decode initialize result: %w", err)
	}
	// Send notifications/initialized as required by MCP spec.
	_, _ = t.post(ctx, "notifications/initialized", nil)
	return &result, nil
}

func (t *sseTransport) ListTools(ctx context.Context) ([]Tool, error) {
	raw, err := t.post(ctx, "tools/list", nil)
	if err != nil {
		return nil, err
	}
	var result ToolsListResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("decode tools/list: %w", err)
	}
	return result.Tools, nil
}

func (t *sseTransport) CallTool(ctx context.Context, name string, args map[string]any) (*CallToolResult, error) {
	params := CallToolParams{Name: name, Arguments: args}
	raw, err := t.post(ctx, "tools/call", params)
	if err != nil {
		return nil, err
	}
	var result CallToolResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("decode tools/call: %w", err)
	}
	return &result, nil
}

func (t *sseTransport) Close() error { return nil }
