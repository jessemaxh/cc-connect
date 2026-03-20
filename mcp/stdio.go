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
	"time"
)

// lineResult is one line produced by the dedicated reader goroutine.
type lineResult struct {
	bytes []byte
	err   error // non-nil only on scanner error (not EOF)
}

// stdioTransport implements MCP over JSON-RPC on stdin/stdout of a subprocess.
//
// Design: a single dedicated readLoop goroutine owns the bufio.Scanner. It sends
// lines to linesCh. call() reads from linesCh using a select, so context
// cancellation is prompt and there is never more than one goroutine touching
// the scanner.
type stdioTransport struct {
	cfg    ServerConfig
	nextID atomic.Int64

	// startMu protects startup fields and Close().
	startMu    sync.Mutex
	started    bool
	startErr   error
	cmd        *exec.Cmd
	enc        *json.Encoder
	linesCh    chan lineResult
	readerDone chan struct{} // closed when readLoop exits

	// callMu serialises request/response pairs (one at a time).
	callMu sync.Mutex
}

func newStdioTransport(cfg ServerConfig) (*stdioTransport, error) {
	if cfg.Command == "" {
		return nil, fmt.Errorf("stdio transport: command is required")
	}
	return &stdioTransport{cfg: cfg}, nil
}

// start launches the subprocess on the first call. Thread-safe; subsequent
// calls return the result of the first attempt.
func (t *stdioTransport) start() error {
	t.startMu.Lock()
	defer t.startMu.Unlock()

	if t.started {
		return t.startErr
	}
	t.started = true

	cmd := exec.Command(t.cfg.Command, t.cfg.Args...) //nolint:gosec — operator config only

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
	cmd.Stderr = io.Discard // discard, not nil (nil inherits parent stderr)

	if err := cmd.Start(); err != nil {
		t.startErr = fmt.Errorf("start: %w", err)
		return t.startErr
	}

	t.cmd = cmd
	t.enc = json.NewEncoder(stdin)
	t.linesCh = make(chan lineResult, 64)
	t.readerDone = make(chan struct{})
	go t.readLoop(stdout)
	return nil
}

// readLoop is the SOLE goroutine that reads from the subprocess stdout.
// Owning the scanner exclusively eliminates all data races on bufio.Scanner.
// It runs for the lifetime of the transport and exits when the subprocess
// closes its stdout (after Kill or natural exit).
func (t *stdioTransport) readLoop(stdout io.ReadCloser) {
	defer close(t.readerDone)
	defer close(t.linesCh)

	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for scanner.Scan() {
		b := make([]byte, len(scanner.Bytes()))
		copy(b, scanner.Bytes())
		t.linesCh <- lineResult{bytes: b}
	}
	if err := scanner.Err(); err != nil {
		// Best-effort send; Close() may be draining the channel concurrently.
		select {
		case t.linesCh <- lineResult{err: err}:
		default:
		}
	}
}

// call sends a JSON-RPC request and reads the matching response.
// It serialises access via callMu so that only one request is outstanding at a time.
// Context cancellation is detected promptly via select on linesCh.
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

	for {
		var r lineResult
		var ok bool
		select {
		case r, ok = <-t.linesCh:
		case <-ctx.Done():
			return nil, ctx.Err()
		}

		if !ok {
			return nil, fmt.Errorf("server closed connection")
		}
		if r.err != nil {
			return nil, fmt.Errorf("read response: %w", r.err)
		}
		if len(r.bytes) == 0 {
			continue
		}

		var resp jsonrpcResponse
		if err := json.Unmarshal(r.bytes, &resp); err != nil {
			slog.Debug("mcp stdio: skip non-JSON line", "line", string(r.bytes[:min(len(r.bytes), 120)]))
			continue
		}

		// Skip server-initiated notifications (no ID).
		if resp.ID == nil {
			continue
		}

		// Match response ID to our request.
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
	// Send notifications/initialized while holding callMu so that no other
	// request (e.g. ListTools) can slip in before the notification is sent
	// (MCP spec requires initialized notification before any subsequent request).
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

// Close kills the subprocess and waits for the reader goroutine to exit before
// calling cmd.Wait(), preventing use-after-close on the stdout file descriptor.
func (t *stdioTransport) Close() error {
	t.startMu.Lock()
	cmd := t.cmd
	linesCh := t.linesCh
	readerDone := t.readerDone
	t.startMu.Unlock()

	if cmd == nil || cmd.Process == nil {
		return nil
	}

	// Kill the subprocess: causes readLoop's scanner to see EOF and exit.
	_ = cmd.Process.Kill()

	// Drain linesCh so readLoop is not blocked trying to send. The drain
	// goroutine exits automatically when linesCh is closed (after readLoop exits).
	if linesCh != nil {
		go func() {
			for range linesCh {
			}
		}()
	}

	// Wait for readLoop to exit before calling cmd.Wait() to avoid a
	// use-after-close race on the stdout file descriptor.
	if readerDone != nil {
		select {
		case <-readerDone:
		case <-time.After(5 * time.Second):
			slog.Warn("mcp stdio: reader goroutine did not exit after Kill")
		}
	}

	_ = cmd.Wait()
	return nil
}
