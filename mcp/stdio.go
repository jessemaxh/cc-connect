package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
)

// stdioTransport implements MCP over JSON-RPC on stdin/stdout of a subprocess.
type stdioTransport struct {
	cfg    ServerConfig
	nextID atomic.Int64

	// startMu protects the startup fields and Close().
	startMu  sync.Mutex
	started  bool
	startErr error
	cmd      *exec.Cmd
	enc      *json.Encoder
	dec      *bufio.Scanner

	// callMu serialises request/response pairs (one outstanding call at a time).
	callMu sync.Mutex
}

func newStdioTransport(cfg ServerConfig) (*stdioTransport, error) {
	if cfg.Command == "" {
		return nil, fmt.Errorf("stdio transport: command is required")
	}
	return &stdioTransport{cfg: cfg}, nil
}

// start launches the subprocess on the first call. Subsequent calls are no-ops
// and return the result of the first attempt. Uses startMu (not sync.Once) so
// that a failed start can be diagnosed and the error is always readable.
func (t *stdioTransport) start() error {
	t.startMu.Lock()
	defer t.startMu.Unlock()

	if t.started {
		return t.startErr
	}
	t.started = true

	args := t.cfg.Args
	cmd := exec.Command(t.cfg.Command, args...) //nolint:gosec — command comes from operator config

	// Build env: inherit parent env and add overrides.
	env := os.Environ()
	for k, v := range t.cfg.Env {
		env = append(env, k+"="+v)
	}
	cmd.Env = env

	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.startErr = fmt.Errorf("stdin pipe: %w", err)
		return t.startErr
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.startErr = fmt.Errorf("stdout pipe: %w", err)
		return t.startErr
	}
	// Discard stderr so it doesn't inherit the parent's stderr and doesn't block.
	cmd.Stderr = io.Discard

	if err := cmd.Start(); err != nil {
		t.startErr = fmt.Errorf("start: %w", err)
		return t.startErr
	}

	t.cmd = cmd
	t.enc = json.NewEncoder(stdin)
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	t.dec = scanner
	return nil
}

func (t *stdioTransport) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	if err := t.start(); err != nil {
		return nil, err
	}

	id := t.nextID.Add(1)
	req := jsonrpcRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}

	t.callMu.Lock()
	defer t.callMu.Unlock()

	if err := t.enc.Encode(req); err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}

	// Read lines until we find the response matching our ID.
	for {
		// Check for cancellation before each scan attempt.
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		if !t.dec.Scan() {
			if err := t.dec.Err(); err != nil {
				return nil, fmt.Errorf("read response: %w", err)
			}
			return nil, fmt.Errorf("server closed connection")
		}
		line := t.dec.Bytes()
		if len(line) == 0 {
			continue
		}

		var resp jsonrpcResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			slog.Debug("mcp stdio: skip non-JSON line", "line", string(line[:min(len(line), 120)]))
			continue
		}

		// Notifications from the server (no ID or null ID) — skip them.
		if resp.ID == nil {
			continue
		}

		// Check if this response matches our request ID.
		// IDs come back as float64 when unmarshalled into any.
		switch v := resp.ID.(type) {
		case float64:
			if int64(v) != id {
				continue
			}
		case int64:
			if v != id {
				continue
			}
		case json.Number:
			if n, err := v.Int64(); err != nil || n != id {
				continue
			}
		default:
			continue
		}

		if resp.Error != nil {
			return nil, resp.Error
		}
		return resp.Result, nil
	}
}

func (t *stdioTransport) Initialize(ctx context.Context) (*InitializeResult, error) {
	params := InitializeParams{
		ProtocolVersion: "2024-11-05",
		ClientInfo:      ClientInfo{Name: "cc-connect", Version: "1.0"},
		Capabilities:    map[string]any{},
	}
	raw, err := t.call(ctx, "initialize", params)
	if err != nil {
		return nil, err
	}
	var result InitializeResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("decode initialize result: %w", err)
	}
	// Send initialized notification (fire-and-forget, spec requirement).
	t.callMu.Lock()
	_ = t.enc.Encode(jsonrpcRequest{JSONRPC: "2.0", Method: "notifications/initialized"})
	t.callMu.Unlock()
	return &result, nil
}

func (t *stdioTransport) ListTools(ctx context.Context) ([]Tool, error) {
	raw, err := t.call(ctx, "tools/list", nil)
	if err != nil {
		return nil, err
	}
	var result ToolsListResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("decode tools/list: %w", err)
	}
	return result.Tools, nil
}

func (t *stdioTransport) CallTool(ctx context.Context, name string, args map[string]any) (*CallToolResult, error) {
	params := CallToolParams{Name: name, Arguments: args}
	raw, err := t.call(ctx, "tools/call", params)
	if err != nil {
		return nil, err
	}
	var result CallToolResult
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, fmt.Errorf("decode tools/call: %w", err)
	}
	return &result, nil
}

// Close kills the subprocess. It first sends SIGKILL (which unblocks any
// in-progress Scan()), then waits for the active call to release callMu
// before calling cmd.Wait() to reap the zombie.
func (t *stdioTransport) Close() error {
	t.startMu.Lock()
	cmd := t.cmd
	t.startMu.Unlock()

	if cmd == nil || cmd.Process == nil {
		return nil
	}

	// Kill first: this makes any blocked Scan() in call() return immediately.
	_ = cmd.Process.Kill()

	// Wait for the in-progress call (if any) to observe the EOF and release callMu.
	t.callMu.Lock()
	t.callMu.Unlock() //nolint:staticcheck — intentional: just waiting for in-flight call to exit

	// Now safe to reap.
	_ = cmd.Wait()
	return nil
}
