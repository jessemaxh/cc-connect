package mcp

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"
)

// Transport is the interface that both stdio and HTTP/SSE transports implement.
type Transport interface {
	// Initialize performs the MCP handshake. Must be called before any other method.
	Initialize(ctx context.Context) (*InitializeResult, error)
	// ListTools returns all tools advertised by the server.
	ListTools(ctx context.Context) ([]Tool, error)
	// CallTool calls a named tool with the provided arguments.
	CallTool(ctx context.Context, name string, args map[string]any) (*CallToolResult, error)
	// Close tears down the transport.
	Close() error
}

// serverEntry holds a transport and its cached tool list.
type serverEntry struct {
	cfg   ServerConfig
	tr    Transport
	tools []Tool
}

// Manager manages a pool of MCP server connections and routes tool calls.
type Manager struct {
	mu      sync.Mutex
	servers []*serverEntry
	// qualifiedIndex: "<server>__<tool>" → entry (always populated)
	// bareIndex:      "<tool>"           → entry (populated only when no collision)
	qualifiedIndex map[string]*serverEntry
	bareIndex      map[string]*serverEntry
}

// NewManager creates an empty manager. Call AddServer to register servers.
func NewManager() *Manager {
	return &Manager{
		qualifiedIndex: make(map[string]*serverEntry),
		bareIndex:      make(map[string]*serverEntry),
	}
}

// AddServer connects to an MCP server, performs the initialize handshake,
// fetches the tool list, and registers all tools for routing.
func (m *Manager) AddServer(ctx context.Context, cfg ServerConfig) error {
	var tr Transport
	var err error

	if cfg.URL != "" {
		tr, err = newSSETransport(cfg)
	} else {
		tr, err = newStdioTransport(cfg)
	}
	if err != nil {
		return fmt.Errorf("mcp: server %q: create transport: %w", cfg.Name, err)
	}

	info, err := tr.Initialize(ctx)
	if err != nil {
		_ = tr.Close()
		return fmt.Errorf("mcp: server %q: initialize: %w", cfg.Name, err)
	}
	slog.Info("mcp: server connected", "name", cfg.Name, "server", info.ServerInfo.Name, "version", info.ServerInfo.Version)

	tools, err := tr.ListTools(ctx)
	if err != nil {
		_ = tr.Close()
		return fmt.Errorf("mcp: server %q: list tools: %w", cfg.Name, err)
	}
	slog.Info("mcp: tools registered", "server", cfg.Name, "count", len(tools))

	entry := &serverEntry{cfg: cfg, tr: tr, tools: tools}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.servers = append(m.servers, entry)
	for _, t := range tools {
		qualified := cfg.Name + "__" + t.Name
		m.qualifiedIndex[qualified] = entry
		// Only register bare name if there is no collision; on collision remove it
		// so callers are forced to use the qualified form.
		if _, exists := m.bareIndex[t.Name]; exists {
			delete(m.bareIndex, t.Name)
		} else {
			m.bareIndex[t.Name] = entry
		}
	}
	return nil
}

// AllTools returns the merged list of all tools across all registered servers.
// Tool names are prefixed with "<server>__" when there is a collision across servers.
func (m *Manager) AllTools() []Tool {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Count bare name occurrences to detect collisions.
	counts := make(map[string]int)
	for _, e := range m.servers {
		for _, t := range e.tools {
			counts[t.Name]++
		}
	}

	var out []Tool
	for _, e := range m.servers {
		for _, t := range e.tools {
			if counts[t.Name] > 1 {
				// Disambiguate: expose as "<server>__<tool>"
				t2 := t
				t2.Name = e.cfg.Name + "__" + t.Name
				out = append(out, t2)
			} else {
				out = append(out, t)
			}
		}
	}
	return out
}

// CallTool dispatches a tool call to the appropriate server.
// The name may be either the bare tool name or the "<server>__<tool>" qualified form.
func (m *Manager) CallTool(ctx context.Context, name string, args map[string]any) (*CallToolResult, error) {
	m.mu.Lock()
	// Try qualified name first (exact match in qualifiedIndex).
	entry := m.qualifiedIndex[name]
	if entry == nil {
		// Try bare name.
		entry = m.bareIndex[name]
	}
	m.mu.Unlock()

	if entry == nil {
		return nil, fmt.Errorf("mcp: unknown tool %q", name)
	}

	// Strip server prefix before forwarding to the transport.
	// Use SplitN so that tool names containing "__" are preserved correctly.
	bareName := name
	if strings.Contains(name, "__") {
		if parts := strings.SplitN(name, "__", 2); len(parts) == 2 {
			bareName = parts[1]
		}
	}

	return entry.tr.CallTool(ctx, bareName, args)
}

// Close shuts down all server transports.
func (m *Manager) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, e := range m.servers {
		if err := e.tr.Close(); err != nil {
			slog.Warn("mcp: close server", "name", e.cfg.Name, "error", err)
		}
	}
	m.servers = nil
	m.qualifiedIndex = make(map[string]*serverEntry)
	m.bareIndex = make(map[string]*serverEntry)
}
