// Package mcp implements the Model Context Protocol (MCP) JSON-RPC 2.0 client.
// It supports stdio and HTTP/SSE transports and provides a unified interface
// for listing and calling tools across multiple MCP servers.
package mcp

import "encoding/json"

// ---- JSON-RPC 2.0 ----

type jsonrpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      any    `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type jsonrpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      any             `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *jsonrpcError   `json:"error,omitempty"`
}

type jsonrpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (e *jsonrpcError) Error() string { return e.Message }

// ---- MCP protocol types ----

// InitializeParams is sent as the first call to a new MCP server.
type InitializeParams struct {
	ProtocolVersion string         `json:"protocolVersion"`
	ClientInfo      ClientInfo     `json:"clientInfo"`
	Capabilities    map[string]any `json:"capabilities"`
}

// ClientInfo identifies the MCP client to the server.
type ClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// InitializeResult is the server's response to initialize.
type InitializeResult struct {
	ProtocolVersion string         `json:"protocolVersion"`
	ServerInfo      ServerInfo     `json:"serverInfo"`
	Capabilities    map[string]any `json:"capabilities"`
}

// ServerInfo identifies the MCP server.
type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// Tool describes one tool exposed by an MCP server.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

// ToolsListResult is returned by tools/list.
type ToolsListResult struct {
	Tools []Tool `json:"tools"`
}

// CallToolParams is sent to tools/call.
type CallToolParams struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments,omitempty"`
}

// ToolContent is one item in a tool result.
type ToolContent struct {
	Type string `json:"type"` // "text" | "image" | "resource"
	Text string `json:"text,omitempty"`
	Data string `json:"data,omitempty"`
}

// CallToolResult is returned by tools/call.
type CallToolResult struct {
	Content []ToolContent `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

// ServerConfig holds the configuration for one MCP server.
type ServerConfig struct {
	Name    string            // human-readable label
	Command string            // path/command to launch (stdio)
	Args    []string          // arguments for the command (stdio)
	Env     map[string]string // extra environment variables (stdio)
	URL     string            // base URL for HTTP/SSE transport; empty → stdio
}
