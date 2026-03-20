package core

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

const maxPlatformMessageLen = 4000

const (
	defaultThinkingMaxLen = 300
	defaultToolMaxLen     = 500
)

// Slow-operation thresholds. Operations exceeding these durations produce a
// slog.Warn so operators can quickly pinpoint bottlenecks.
const (
	slowPlatformSend    = 2 * time.Second  // platform Reply / Send
	slowAgentStart      = 5 * time.Second  // agent.StartSession
	slowAgentClose      = 3 * time.Second  // agentSession.Close
	slowAgentSend       = 2 * time.Second  // agentSession.Send
	slowAgentFirstEvent = 15 * time.Second // time from send to first agent event
)

// VersionInfo is set by main at startup so that /version works.
var VersionInfo string

// CurrentVersion is the semver tag (e.g. "v1.2.0-beta.1"), set by main.
var CurrentVersion string

// RestartRequest carries info needed to send a post-restart notification.
type RestartRequest struct {
	SessionKey string `json:"session_key"`
	Platform   string `json:"platform"`
}

// SaveRestartNotify persists restart info so the new process can send
// a "restart successful" message after startup.
func SaveRestartNotify(dataDir string, req RestartRequest) error {
	dir := filepath.Join(dataDir, "run")
	os.MkdirAll(dir, 0o755)
	data, _ := json.Marshal(req)
	return os.WriteFile(filepath.Join(dir, "restart_notify"), data, 0o644)
}

// ConsumeRestartNotify reads and deletes the restart notification file.
// Returns nil if no notification is pending.
func ConsumeRestartNotify(dataDir string) *RestartRequest {
	p := filepath.Join(dataDir, "run", "restart_notify")
	data, err := os.ReadFile(p)
	if err != nil {
		return nil
	}
	os.Remove(p)
	var req RestartRequest
	if json.Unmarshal(data, &req) != nil {
		return nil
	}
	return &req
}

// SendRestartNotification sends a "restart successful" message to the
// platform/session that initiated the restart.
func (e *Engine) SendRestartNotification(platformName, sessionKey string) {
	for _, p := range e.platforms {
		if p.Name() != platformName {
			continue
		}
		rc, ok := p.(ReplyContextReconstructor)
		if !ok {
			slog.Debug("restart notify: platform does not support ReconstructReplyCtx", "platform", platformName)
			return
		}
		rctx, err := rc.ReconstructReplyCtx(sessionKey)
		if err != nil {
			slog.Debug("restart notify: reconstruct failed", "error", err)
			return
		}
		text := e.i18n.T(MsgRestartSuccess)
		if CurrentVersion != "" {
			text += fmt.Sprintf(" (%s)", CurrentVersion)
		}
		if err := p.Send(e.ctx, rctx, text); err != nil {
			slog.Debug("restart notify: send failed", "error", err)
		}
		return
	}
}

// RestartCh is signaled when /restart is invoked. main listens on it
// to perform a graceful shutdown followed by syscall.Exec.
var RestartCh = make(chan RestartRequest, 1)

// DisplayCfg controls truncation of intermediate messages.
// A value of -1 means "use default", 0 means "no truncation".
type DisplayCfg struct {
	ThinkingMaxLen int // max runes for thinking preview; 0 = no truncation
	ToolMaxLen     int // max runes for tool use preview; 0 = no truncation
}

// RateLimitCfg controls per-session message rate limiting.
type RateLimitCfg struct {
	MaxMessages int           // max messages per window; 0 = disabled
	Window      time.Duration // sliding window size
}

// Engine routes messages between platforms and the agent for a single project.
type Engine struct {
	name         string
	agent        Agent
	platforms    []Platform
	sessions     *SessionManager
	ctx          context.Context
	cancel       context.CancelFunc
	i18n         *I18n
	speech       SpeechCfg
	tts          *TTSCfg
	display      DisplayCfg
	defaultQuiet bool
	injectSender bool
	startedAt    time.Time

	providerSaveFunc       func(providerName string) error
	providerAddSaveFunc    func(p ProviderConfig) error
	providerRemoveSaveFunc func(name string) error

	ttsSaveFunc func(mode string) error

	commandSaveAddFunc func(name, description, prompt, exec, workDir string) error
	commandSaveDelFunc func(name string) error

	displaySaveFunc  func(thinkingMaxLen, toolMaxLen *int) error
	configReloadFunc func() (*ConfigReloadResult, error)

	cronScheduler      *CronScheduler
	heartbeatScheduler *HeartbeatScheduler

	commands *CommandRegistry
	skills   *SkillRegistry
	aliases  map[string]string // trigger → command (e.g. "帮助" → "/help")
	aliasMu  sync.RWMutex

	aliasSaveAddFunc func(name, command string) error
	aliasSaveDelFunc func(name string) error

	bannedWords []string
	bannedMu    sync.RWMutex

	disabledCmds map[string]bool
	adminFrom    string // comma-separated user IDs for privileged commands; "*" = all allowed users; "" = deny

	rateLimiter       *RateLimiter
	streamPreview     StreamPreviewCfg
	relayManager      *RelayManager
	eventIdleTimeout  time.Duration
	authWebhookURL    string // URL to call for message authentication
	authWebhookSecret string // shared secret sent as X-Webhook-Secret header

	// Multi-workspace mode
	multiWorkspace    bool
	baseDir           string
	workspaceBindings *WorkspaceBindingManager
	workspacePool     *workspacePool
	initFlows         map[string]*workspaceInitFlow // channelID → init state
	initFlowsMu       sync.Mutex

	// Interactive agent session management
	interactiveMu     sync.Mutex
	interactiveStates map[string]*interactiveState // key = sessionKey

	quietMu sync.RWMutex
	quiet   bool // when true, suppress thinking and tool progress messages globally

	activeTurns atomic.Int32
}

// workspaceInitFlow tracks a channel that is being onboarded to a workspace.
type workspaceInitFlow struct {
	state       string // "awaiting_url", "awaiting_confirm"
	repoURL     string
	cloneTo     string
	channelName string
}

func NewEngine(name string, ag Agent, platforms []Platform, sessionStorePath string, lang Language) *Engine {
	ctx, cancel := context.WithCancel(context.Background())
	e := &Engine{
		name:              name,
		agent:             ag,
		platforms:         platforms,
		sessions:          NewSessionManager(sessionStorePath),
		ctx:               ctx,
		cancel:            cancel,
		i18n:              NewI18n(lang),
		display:           DisplayCfg{ThinkingMaxLen: defaultThinkingMaxLen, ToolMaxLen: defaultToolMaxLen},
		commands:          NewCommandRegistry(),
		skills:            NewSkillRegistry(),
		aliases:           make(map[string]string),
		interactiveStates: make(map[string]*interactiveState),
		startedAt:         time.Now(),
		streamPreview:     DefaultStreamPreviewCfg(),
		eventIdleTimeout:  defaultEventIdleTimeout,
	}

	if ag != nil {
		e.sessions.InvalidateForAgent(ag.Name())
	}

	if cp, ok := ag.(CommandProvider); ok {
		e.commands.SetAgentDirs(cp.CommandDirs())
	}
	if sp, ok := ag.(SkillProvider); ok {
		e.skills.SetDirs(sp.SkillDirs())
	}

	return e
}

// SetMultiWorkspace enables multi-workspace mode for the engine.
func (e *Engine) SetMultiWorkspace(baseDir, bindingStorePath string) {
	e.multiWorkspace = true
	e.baseDir = baseDir
	e.workspaceBindings = NewWorkspaceBindingManager(bindingStorePath)
	e.workspacePool = newWorkspacePool(15 * time.Minute)
	e.initFlows = make(map[string]*workspaceInitFlow)
	go e.runIdleReaper()
}

func (e *Engine) runIdleReaper() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	for {
		select {
		case <-e.ctx.Done():
			return
		case <-ticker.C:
			if e.workspacePool == nil {
				continue
			}
			reaped := e.workspacePool.ReapIdle()
			for _, ws := range reaped {
				e.interactiveMu.Lock()
				for key, state := range e.interactiveStates {
					if state.workspaceDir == ws {
						if state.agentSession != nil {
							state.agentSession.Close()
						}
						delete(e.interactiveStates, key)
					}
				}
				e.interactiveMu.Unlock()
				slog.Info("workspace idle-reaped", "workspace", ws)
			}
		}
	}
}

// SetSpeechConfig configures the speech-to-text subsystem.
func (e *Engine) SetSpeechConfig(cfg SpeechCfg) {
	e.speech = cfg
}

// SetTTSConfig configures the text-to-speech subsystem.
func (e *Engine) SetTTSConfig(cfg *TTSCfg) {
	e.tts = cfg
}

// SetTTSSaveFunc registers a callback that persists TTS mode changes.
func (e *Engine) SetTTSSaveFunc(fn func(mode string) error) {
	e.ttsSaveFunc = fn
}

// SetDisplayConfig overrides the default truncation settings.
func (e *Engine) SetDisplayConfig(cfg DisplayCfg) {
	e.display = cfg
}

// SetDefaultQuiet sets whether new sessions start in quiet mode.
func (e *Engine) SetDefaultQuiet(q bool) {
	e.defaultQuiet = q
}

// SetInjectSender controls whether sender identity (platform and user ID) is
// prepended to each message before forwarding it to the agent. When enabled,
// the agent receives a preamble line like:
//
//	[cc-connect sender_id=ou_abc123 platform=feishu]
//
// This allows the agent to identify who sent the message and adjust behavior
// accordingly (e.g. personal task views, role-based access control).
func (e *Engine) SetInjectSender(v bool) {
	e.injectSender = v
}

func (e *Engine) SetLanguageSaveFunc(fn func(Language) error) {
	e.i18n.SetSaveFunc(fn)
}

func (e *Engine) SetProviderSaveFunc(fn func(providerName string) error) {
	e.providerSaveFunc = fn
}

func (e *Engine) SetProviderAddSaveFunc(fn func(ProviderConfig) error) {
	e.providerAddSaveFunc = fn
}

func (e *Engine) SetProviderRemoveSaveFunc(fn func(string) error) {
	e.providerRemoveSaveFunc = fn
}

// AddPlatform appends a platform to the engine after construction.
// The platform is started and wired during the next Engine.Start call,
// or if the engine is already running, it is started immediately.
func (e *Engine) AddPlatform(p Platform) {
	e.platforms = append(e.platforms, p)
}

func (e *Engine) SetCronScheduler(cs *CronScheduler) {
	e.cronScheduler = cs
}

func (e *Engine) SetHeartbeatScheduler(hs *HeartbeatScheduler) {
	e.heartbeatScheduler = hs
}

func (e *Engine) SetCommandSaveAddFunc(fn func(name, description, prompt, exec, workDir string) error) {
	e.commandSaveAddFunc = fn
}

func (e *Engine) SetCommandSaveDelFunc(fn func(name string) error) {
	e.commandSaveDelFunc = fn
}

func (e *Engine) SetDisplaySaveFunc(fn func(thinkingMaxLen, toolMaxLen *int) error) {
	e.displaySaveFunc = fn
}

// ConfigReloadResult describes what was updated by a config reload.
type ConfigReloadResult struct {
	DisplayUpdated   bool
	ProvidersUpdated int
	CommandsUpdated  int
}

func (e *Engine) SetConfigReloadFunc(fn func() (*ConfigReloadResult, error)) {
	e.configReloadFunc = fn
}

// GetAgent returns the engine's agent (for type assertions like ProviderSwitcher).
func (e *Engine) GetAgent() Agent {
	return e.agent
}

// AddCommand registers a custom slash command.
func (e *Engine) AddCommand(name, description, prompt, exec, workDir, source string) {
	e.commands.Add(name, description, prompt, exec, workDir, source)
}

// ClearCommands removes all commands from the given source.
func (e *Engine) ClearCommands(source string) {
	e.commands.ClearSource(source)
}

// AddAlias registers a command alias.
func (e *Engine) AddAlias(name, command string) {
	e.aliasMu.Lock()
	defer e.aliasMu.Unlock()
	e.aliases[name] = command
}

func (e *Engine) SetAliasSaveAddFunc(fn func(name, command string) error) {
	e.aliasSaveAddFunc = fn
}

func (e *Engine) SetAliasSaveDelFunc(fn func(name string) error) {
	e.aliasSaveDelFunc = fn
}

// ClearAliases removes all aliases (for config reload).
func (e *Engine) ClearAliases() {
	e.aliasMu.Lock()
	defer e.aliasMu.Unlock()
	e.aliases = make(map[string]string)
}

// SetDisabledCommands sets the list of command IDs that are disabled for this project.
func (e *Engine) SetDisabledCommands(cmds []string) {
	m := make(map[string]bool, len(cmds))
	for _, c := range cmds {
		c = strings.ToLower(strings.TrimPrefix(c, "/"))
		// Resolve alias names to canonical IDs
		id := matchPrefix(c, builtinCommands)
		if id != "" {
			m[id] = true
		} else {
			m[c] = true
		}
	}
	e.disabledCmds = m
}

// SetAdminFrom sets the admin allowlist for privileged commands.
// "*" means all users who pass allow_from are admins.
// Empty string means privileged commands are denied for everyone.
func (e *Engine) SetAdminFrom(adminFrom string) {
	e.adminFrom = strings.TrimSpace(adminFrom)
	if e.adminFrom == "" && !e.disabledCmds["shell"] {
		slog.Warn("admin_from is not set — privileged commands (/shell, /restart, /upgrade) are blocked. "+
			"Set admin_from in config to enable them, or use disabled_commands to hide them.",
			"project", e.name)
	}
}

// privilegedCommands are commands that require admin_from authorization.
var privilegedCommands = map[string]bool{
	"shell":   true,
	"restart": true,
	"upgrade": true,
}

// isAdmin checks whether the given user ID is authorized for privileged commands.
// Unlike AllowList, empty adminFrom means deny-all (fail-closed).
func (e *Engine) isAdmin(userID string) bool {
	af := strings.TrimSpace(e.adminFrom)
	if af == "" {
		return false
	}
	if af == "*" {
		return true
	}
	for _, id := range strings.Split(af, ",") {
		if strings.EqualFold(strings.TrimSpace(id), userID) {
			return true
		}
	}
	return false
}

// SetBannedWords replaces the banned words list.
func (e *Engine) SetBannedWords(words []string) {
	e.bannedMu.Lock()
	defer e.bannedMu.Unlock()
	lower := make([]string, len(words))
	for i, w := range words {
		lower[i] = strings.ToLower(w)
	}
	e.bannedWords = lower
}

// SetRateLimitCfg configures per-session message rate limiting.
// It stops the previous rate limiter's background goroutine before replacing it.
func (e *Engine) SetRateLimitCfg(cfg RateLimitCfg) {
	if e.rateLimiter != nil {
		e.rateLimiter.Stop()
	}
	e.rateLimiter = NewRateLimiter(cfg.MaxMessages, cfg.Window)
}

// SetStreamPreviewCfg configures the streaming preview behavior.
func (e *Engine) SetStreamPreviewCfg(cfg StreamPreviewCfg) {
	e.streamPreview = cfg
}

// SetEventIdleTimeout sets the maximum time to wait between consecutive agent events.
// 0 disables the timeout entirely.
func (e *Engine) SetEventIdleTimeout(d time.Duration) {
	e.eventIdleTimeout = d
}

func (e *Engine) SetAuthWebhook(url, secret string) {
	e.authWebhookURL = url
	e.authWebhookSecret = secret
}

func (e *Engine) SetRelayManager(rm *RelayManager) {
	e.relayManager = rm
}

func (e *Engine) RelayManager() *RelayManager {
	return e.relayManager
}

// RemoveCommand removes a custom command by name. Returns false if not found.
func (e *Engine) RemoveCommand(name string) bool {
	return e.commands.Remove(name)
}

func (e *Engine) ProjectName() string {
	return e.name
}

// ActiveSessionKeys returns the session keys of all active interactive sessions.
func (e *Engine) ActiveSessionKeys() []string {
	e.interactiveMu.Lock()
	defer e.interactiveMu.Unlock()
	var keys []string
	for key, state := range e.interactiveStates {
		if state.platform != nil {
			keys = append(keys, key)
		}
	}
	return keys
}

func (e *Engine) Start() error {
	var startErrs []error
	for _, p := range e.platforms {
		if err := p.Start(e.handleMessage); err != nil {
			slog.Warn("platform start failed", "project", e.name, "platform", p.Name(), "error", err)
			startErrs = append(startErrs, fmt.Errorf("[%s] start platform %s: %w", e.name, p.Name(), err))
			continue
		}
		slog.Info("platform started", "project", e.name, "platform", p.Name())

		// Register commands on platforms that support it (e.g. Telegram setMyCommands)
		if registrar, ok := p.(CommandRegistrar); ok {
			commands := e.GetAllCommands()
			if err := registrar.RegisterCommands(commands); err != nil {
				slog.Error("platform command registration failed", "project", e.name, "platform", p.Name(), "error", err)
			} else {
				slog.Debug("platform commands registered", "project", e.name, "platform", p.Name(), "count", len(commands))
			}
		}

		if nav, ok := p.(CardNavigable); ok {
			nav.SetCardNavigationHandler(e.handleCardNav)
		}
	}

	// Log summary
	startedCount := len(e.platforms) - len(startErrs)
	if len(startErrs) > 0 {
		slog.Warn("engine started with some failures", "project", e.name, "agent", e.agent.Name(), "started", startedCount, "failed", len(startErrs))
	} else {
		slog.Info("engine started", "project", e.name, "agent", e.agent.Name(), "platforms", len(e.platforms))
	}

	// Only return error if ALL platforms failed
	if len(startErrs) == len(e.platforms) && len(e.platforms) > 0 {
		return startErrs[0] // Return first error
	}
	return nil
}

// HealthCheck represents the health status of a single component.
type HealthCheck struct {
	Component string `json:"component"` // "platform:<name>", "agent", "skills"
	Status    string `json:"status"`    // "ok", "warning", "error"
	Message   string `json:"message"`
}

// HealthChecks returns deep health information for this engine's components.
func (e *Engine) HealthChecks() []HealthCheck {
	var checks []HealthCheck

	// Check platforms
	for _, p := range e.platforms {
		check := HealthCheck{
			Component: "platform:" + p.Name(),
			Status:    "ok",
			Message:   p.Name() + " connected",
		}
		if hc, ok := p.(HealthChecker); ok {
			if err := hc.HealthCheck(); err != nil {
				check.Status = "error"
				check.Message = p.Name() + ": " + err.Error()
			}
		}
		checks = append(checks, check)
	}

	// Check agent
	agentCheck := HealthCheck{
		Component: "agent",
		Status:    "ok",
		Message:   e.agent.Name() + " ready",
	}
	if hc, ok := e.agent.(HealthChecker); ok {
		if err := hc.HealthCheck(); err != nil {
			agentCheck.Status = "warning"
			agentCheck.Message = e.agent.Name() + ": " + err.Error()
		}
	}
	checks = append(checks, agentCheck)

	// Skills count
	if e.skills != nil {
		count := len(e.skills.ListAll())
		checks = append(checks, HealthCheck{
			Component: "skills",
			Status:    "ok",
			Message:   fmt.Sprintf("%d skills loaded", count),
		})
	}

	return checks
}

func (e *Engine) Stop() error {
	// Stop platforms first to prevent new incoming messages
	var errs []error
	for _, p := range e.platforms {
		if err := p.Stop(); err != nil {
			errs = append(errs, fmt.Errorf("stop platform %s: %w", p.Name(), err))
		}
	}

	// Now cancel context and clean up sessions
	e.cancel()

	e.interactiveMu.Lock()
	states := make(map[string]*interactiveState, len(e.interactiveStates))
	for k, v := range e.interactiveStates {
		states[k] = v
		delete(e.interactiveStates, k)
	}
	e.interactiveMu.Unlock()

	for key, state := range states {
		if state.agentSession != nil {
			slog.Debug("engine.Stop: closing agent session", "session", key)
			state.agentSession.Close()
		}
	}

	if e.rateLimiter != nil {
		e.rateLimiter.Stop()
	}

	if err := e.agent.Stop(); err != nil {
		errs = append(errs, fmt.Errorf("stop agent %s: %w", e.agent.Name(), err))
	}
	if len(errs) > 0 {
		return fmt.Errorf("engine stop errors: %v", errs)
	}
	return nil
}

