// Package api implements the "api" agent type for cc-connect.
//
// It calls any OpenAI-compatible LLM API directly over HTTP (no CLI binary
// required), supports streaming, and integrates with MCP servers for tool use.
//
// Configuration (in config.toml):
//
//	[agent]
//	type = "api"
//
//	[agent.options]
//	base_url = "https://api.anthropic.com"  # or OpenAI, Ollama, etc.
//	api_key  = "sk-..."
//	model    = "claude-opus-4-6"
//	system   = "You are a helpful assistant."  # optional
//
//	# Zero or more MCP server definitions
//	[[agent.options.mcp_servers]]
//	name    = "filesystem"
//	command = "npx"
//	args    = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
//
//	[[agent.options.mcp_servers]]
//	name = "weather"
//	url  = "https://mcp.example.com"
//
// SECURITY NOTE: mcp_servers[].command is executed as a subprocess and
// mcp_servers[].url is used for outbound HTTP requests. Both must come from
// trusted operator configuration — never from user input.
package api

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/chenhg5/cc-connect/core"
	"github.com/chenhg5/cc-connect/mcp"
)

func init() {
	core.RegisterAgent("api", New)
}

// Agent implements core.Agent for direct API-based LLM calls.
type Agent struct {
	llm    *llmClient
	mcpMgr *mcp.Manager
	skills *core.SkillEngine

	mu        sync.Mutex
	model     string
	system    string
	providers []core.ProviderConfig
	activeIdx int
	started   atomic.Bool
}

// New creates a new api Agent from options map.
func New(opts map[string]any) (core.Agent, error) {
	baseURL, _ := opts["base_url"].(string)
	if baseURL == "" {
		baseURL = "https://api.openai.com"
	}
	apiKey, _ := opts["api_key"].(string)
	model, _ := opts["model"].(string)
	if model == "" {
		model = "gpt-4o"
	}
	system, _ := opts["system"].(string)

	a := &Agent{
		llm:       newLLMClient(baseURL, apiKey),
		mcpMgr:    mcp.NewManager(),
		model:     model,
		system:    system,
		activeIdx: -1,
	}

	// Parse MCP server definitions.
	if servers, ok := opts["mcp_servers"].([]any); ok {
		if err := a.loadMCPServers(servers); err != nil {
			slog.Warn("api agent: some MCP servers failed to connect", "error", err)
		}
	}

	// Initialize SkillEngine if server URL and project ID are configured.
	serverURL, _ := opts["server_url"].(string)
	projectID, _ := opts["project_id"].(string)
	webhookSecret, _ := opts["webhook_secret"].(string)
	if serverURL != "" && projectID != "" {
		a.skills = core.NewSkillEngine(a.mcpMgr, serverURL, projectID, webhookSecret)
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		if err := a.skills.Load(ctx); err != nil {
			slog.Warn("api agent: skill engine load failed", "error", err)
		}
		cancel()
	}

	a.started.Store(true)
	return a, nil
}

func (a *Agent) loadMCPServers(servers []any) error {
	var lastErr error
	for _, raw := range servers {
		m, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		cfg := mcp.ServerConfig{
			Name:    strOpt(m, "name"),
			Command: strOpt(m, "command"),
			URL:     strOpt(m, "url"),
		}
		if rawArgs, ok := m["args"].([]any); ok {
			for _, a := range rawArgs {
				if s, ok := a.(string); ok {
					cfg.Args = append(cfg.Args, s)
				}
			}
		}
		if rawEnv, ok := m["env"].(map[string]any); ok {
			cfg.Env = make(map[string]string, len(rawEnv))
			for k, v := range rawEnv {
				if s, ok := v.(string); ok {
					cfg.Env[k] = s
				}
			}
		}
		if cfg.Name == "" {
			cfg.Name = cfg.Command
		}
		// Each server gets its own 30-second budget so a slow server doesn't
		// starve subsequent ones.
		svrCtx, svrCancel := context.WithTimeout(context.Background(), 30*time.Second)
		err := a.mcpMgr.AddServer(svrCtx, cfg)
		svrCancel()
		if err != nil {
			slog.Error("api agent: connect MCP server", "name", cfg.Name, "error", err)
			lastErr = err
		}
	}
	return lastErr
}

func strOpt(m map[string]any, key string) string {
	v, _ := m[key].(string)
	return v
}

// ---- core.Agent ----

func (a *Agent) Name() string { return "api" }

func (a *Agent) StartSession(ctx context.Context, sessionID string) (core.AgentSession, error) {
	a.mu.Lock()
	model := a.model
	system := a.system
	llm := a.llm
	// If a provider is active, use its settings.
	if a.activeIdx >= 0 && a.activeIdx < len(a.providers) {
		p := &a.providers[a.activeIdx]
		if p.Model != "" {
			model = p.Model
		}
		if p.BaseURL != "" || p.APIKey != "" {
			llm = newLLMClient(p.BaseURL, p.APIKey)
		}
	}
	a.mu.Unlock()

	return newAPISession(ctx, llm, a.mcpMgr, a.skills, model, system, sessionID), nil
}

func (a *Agent) ListSessions(_ context.Context) ([]core.AgentSessionInfo, error) {
	return nil, nil
}

func (a *Agent) Stop() error {
	a.started.Store(false)
	a.mcpMgr.Close()
	return nil
}

// HealthCheck implements core.HealthChecker.
func (a *Agent) HealthCheck() error {
	if !a.started.Load() {
		return fmt.Errorf("api agent: not started")
	}
	if a.llm == nil {
		return fmt.Errorf("api agent: llm client is nil")
	}
	a.mu.Lock()
	model := a.model
	a.mu.Unlock()
	if model == "" {
		return fmt.Errorf("api agent: model is empty")
	}
	return nil
}

// ---- core.ModelSwitcher ----

func (a *Agent) SetModel(model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.model = model
	slog.Info("api agent: model changed", "model", model)
}

func (a *Agent) GetModel() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.activeIdx >= 0 && a.activeIdx < len(a.providers) {
		if m := a.providers[a.activeIdx].Model; m != "" {
			return m
		}
	}
	return a.model
}

func (a *Agent) AvailableModels(_ context.Context) []core.ModelOption {
	return []core.ModelOption{
		{Name: "claude-opus-4-6", Desc: "Claude Opus 4.6 (most capable)"},
		{Name: "claude-sonnet-4-6", Desc: "Claude Sonnet 4.6 (balanced)"},
		{Name: "claude-haiku-4-5-20251001", Desc: "Claude Haiku 4.5 (fast)"},
		{Name: "gpt-4o", Desc: "GPT-4o"},
		{Name: "gpt-4o-mini", Desc: "GPT-4o Mini (fast)"},
		{Name: "gemini-2.0-flash", Desc: "Gemini 2.0 Flash"},
		{Name: "qwen-plus", Desc: "Qwen Plus"},
	}
}

// ---- core.ProviderSwitcher ----

func (a *Agent) SetProviders(providers []core.ProviderConfig) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.providers = providers
	a.activeIdx = -1
	if len(providers) > 0 {
		a.activeIdx = 0
		a.applyActiveProvider()
	}
}

func (a *Agent) SetActiveProvider(name string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	for i, p := range a.providers {
		if p.Name == name {
			a.activeIdx = i
			a.applyActiveProvider()
			slog.Info("api agent: provider switched", "provider", name)
			return true
		}
	}
	return false
}

func (a *Agent) GetActiveProvider() *core.ProviderConfig {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.activeIdx >= 0 && a.activeIdx < len(a.providers) {
		p := a.providers[a.activeIdx]
		return &p
	}
	return nil
}

func (a *Agent) ListProviders() []core.ProviderConfig {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]core.ProviderConfig, len(a.providers))
	copy(out, a.providers)
	return out
}

// applyActiveProvider updates the llm client when the active provider changes.
// Must be called with a.mu held.
func (a *Agent) applyActiveProvider() {
	if a.activeIdx < 0 || a.activeIdx >= len(a.providers) {
		return
	}
	p := &a.providers[a.activeIdx]
	if p.BaseURL != "" || p.APIKey != "" {
		a.llm = newLLMClient(p.BaseURL, p.APIKey)
	}
	if p.Model != "" {
		a.model = p.Model
	}
}

// ---- core.SystemPromptSupporter ----

func (a *Agent) HasSystemPromptSupport() bool { return true }

// Compile-time interface checks.
var _ core.ModelSwitcher = (*Agent)(nil)
var _ core.ProviderSwitcher = (*Agent)(nil)
var _ core.SystemPromptSupporter = (*Agent)(nil)
var _ core.HealthChecker = (*Agent)(nil)
