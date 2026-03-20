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
	Initialize(ctx context.Context) (*InitializeResult, error)
	ListTools(ctx context.Context) ([]Tool, error)
	CallTool(ctx context.Context, name string, args map[string]any) (*CallToolResult, error)
	Close() error
}

type serverEntry struct {
	cfg   ServerConfig
	tr    Transport
	tools []Tool
}

// Manager manages a pool of MCP server connections and routes tool calls.
type Manager struct {
	mu sync.Mutex

	servers []*serverEntry

	// qualifiedIndex maps "<server>__<tool>" → entry (always populated).
	qualifiedIndex map[string]*serverEntry

	// bareIndex maps "<tool>" → entry (populated only when there is exactly
	// one server that provides a given tool name).
	bareIndex map[string]*serverEntry

	// collisionNames tracks bare tool names that appear on ≥2 servers.
	// Once a name is in collisionNames it is NEVER re-added to bareIndex,
	// even if a third server registers the same name.
	collisionNames map[string]struct{}
}

func NewManager() *Manager {
	return &Manager{
		qualifiedIndex: make(map[string]*serverEntry),
		bareIndex:      make(map[string]*serverEntry),
		collisionNames: make(map[string]struct{}),
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

		// Bare name: register only when there is no collision. Track collisions in
		// collisionNames so that a third server registering the same name doesn't
		// silently re-activate the bare mapping.
		if _, collided := m.collisionNames[t.Name]; collided {
			// Already a known collision — never use bare name for this tool.
		} else if _, exists := m.bareIndex[t.Name]; exists {
			// First collision detected: remove bare mapping and mark as collided.
			delete(m.bareIndex, t.Name)
			m.collisionNames[t.Name] = struct{}{}
		} else {
			m.bareIndex[t.Name] = entry
		}
	}
	return nil
}

// AllTools returns the merged list of tools from all servers.
// Colliding tool names are exposed as "<server>__<tool>"; unique names as-is.
func (m *Manager) AllTools() []Tool {
	m.mu.Lock()
	defer m.mu.Unlock()

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
// Accepts both bare names and "<server>__<tool>" qualified names.
func (m *Manager) CallTool(ctx context.Context, name string, args map[string]any) (*CallToolResult, error) {
	m.mu.Lock()
	// Try qualified name first.
	entry := m.qualifiedIndex[name]
	if entry == nil {
		// Try bare name.
		entry = m.bareIndex[name]
	}
	m.mu.Unlock()

	if entry == nil {
		return nil, fmt.Errorf("mcp: unknown tool %q", name)
	}

	// Strip the server prefix to get the bare tool name sent to the transport.
	// Use the entry's own server name as the prefix to avoid ambiguity when
	// the server name itself contains "__".
	bareName := name
	prefix := entry.cfg.Name + "__"
	if strings.HasPrefix(name, prefix) {
		bareName = name[len(prefix):]
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
	m.collisionNames = make(map[string]struct{})
}
