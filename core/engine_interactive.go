package core

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// interactiveState tracks a running interactive agent session and its permission state.
type interactiveState struct {
	agentSession AgentSession
	platform     Platform
	replyCtx     any
	workspaceDir string
	mu           sync.Mutex
	pending      *pendingPermission
	approveAll   bool // when true, auto-approve all permission requests for this session
	quiet        bool // when true, suppress thinking and tool progress for this session
	fromVoice    bool // true if current turn originated from voice transcription
	deleteMode   *deleteModeState
}

type deleteModeState struct {
	page        int
	selectedIDs map[string]struct{}
	phase       string
	hint        string
	result      string
}

// ──────────────────────────────────────────────────────────────
// Interactive agent processing
// ──────────────────────────────────────────────────────────────

func (e *Engine) processInteractiveMessage(p Platform, msg *Message, session *Session) {
	e.processInteractiveMessageWith(p, msg, session, e.agent, msg.SessionKey, "")
}

// processInteractiveMessageWith is the core interactive processing loop.
// It accepts an explicit agent, interactiveKey (for the interactiveStates map),
// and workspaceDir so that multi-workspace mode can route to per-workspace agents.
func (e *Engine) processInteractiveMessageWith(p Platform, msg *Message, session *Session, agent Agent, interactiveKey string, workspaceDir string) {
	defer session.Unlock()
	e.activeTurns.Add(1)
	defer e.activeTurns.Add(-1)

	if e.ctx.Err() != nil {
		return
	}

	turnStart := time.Now()
	networkStart, netErr := capturePodNetworkSnapshot()
	if netErr != nil {
		slog.Debug("network usage: start snapshot unavailable", "error", netErr)
	}

	e.i18n.DetectAndSet(msg.Content)
	session.AddHistory("user", msg.Content)

	// Use the agent override when available (multi-workspace mode)
	var agentOverride Agent
	if agent != e.agent {
		agentOverride = agent
	}
	state := e.getOrCreateInteractiveStateWith(interactiveKey, p, msg.ReplyCtx, session, agentOverride)

	// Set workspaceDir on the state for idle reaper identification
	if workspaceDir != "" {
		state.mu.Lock()
		state.workspaceDir = workspaceDir
		state.mu.Unlock()
	}

	// Update reply context for this turn
	state.mu.Lock()
	state.platform = p
	state.replyCtx = msg.ReplyCtx
	state.mu.Unlock()

	if state.agentSession == nil {
		e.reply(p, msg.ReplyCtx, fmt.Sprintf(e.i18n.T(MsgError), "failed to start agent session"))
		return
	}

	// Start typing indicator if platform supports it
	var stopTyping func()
	if ti, ok := p.(TypingIndicator); ok {
		stopTyping = ti.StartTyping(e.ctx, msg.ReplyCtx)
	}
	defer func() {
		if stopTyping != nil {
			stopTyping()
		}
	}()

	// Drain any stale events left in the channel from a previous turn.
	// This prevents the next processInteractiveEvents from reading an old
	// EventResult that was pushed after the previous turn already returned.
	drainEvents(state.agentSession.Events())

	// Prepend sender identity when enabled, so the agent knows who sent the message.
	promptContent := msg.Content
	if e.injectSender && msg.UserID != "" {
		chatID := extractChannelID(msg.SessionKey)
		promptContent = fmt.Sprintf("[cc-connect sender_id=%s platform=%s chat_id=%s]\n%s", msg.UserID, msg.Platform, chatID, msg.Content)
	}

	sendStart := time.Now()
	state.mu.Lock()
	state.fromVoice = msg.FromVoice
	state.mu.Unlock()
	if err := state.agentSession.Send(promptContent, msg.Images, msg.Files); err != nil {
		slog.Error("failed to send prompt", "error", err)

		if !state.agentSession.Alive() {
			e.cleanupInteractiveState(interactiveKey, state)
			e.send(p, msg.ReplyCtx, e.i18n.T(MsgSessionRestarting))

			state = e.getOrCreateInteractiveStateWith(interactiveKey, p, msg.ReplyCtx, session, agentOverride)
			if workspaceDir != "" {
				state.mu.Lock()
				state.workspaceDir = workspaceDir
				state.mu.Unlock()
			}
			if state.agentSession == nil {
				e.reply(p, msg.ReplyCtx, fmt.Sprintf(e.i18n.T(MsgError), "failed to restart agent session"))
				return
			}
			sendStart = time.Now()
			if err := state.agentSession.Send(promptContent, msg.Images, msg.Files); err != nil {
				e.reply(p, msg.ReplyCtx, fmt.Sprintf(e.i18n.T(MsgError), err))
				return
			}
		} else {
			e.reply(p, msg.ReplyCtx, fmt.Sprintf(e.i18n.T(MsgError), err))
			return
		}
	}
	if elapsed := time.Since(sendStart); elapsed >= slowAgentSend {
		slog.Warn("slow agent send", "elapsed", elapsed, "session", msg.SessionKey, "content_len", len(msg.Content))
	}

	e.processInteractiveEvents(state, session, interactiveKey, msg, msg.Content, agent, msg.MessageID, turnStart, networkStart)
}

// getOrCreateWorkspaceAgent returns (or creates) a per-workspace agent and session manager.
func (e *Engine) getOrCreateWorkspaceAgent(workspace string) (Agent, *SessionManager, error) {
	ws := e.workspacePool.GetOrCreate(workspace)
	ws.mu.Lock()
	defer ws.mu.Unlock()

	if ws.agent != nil {
		return ws.agent, ws.sessions, nil
	}

	// Create a new agent instance with this workspace's work_dir
	opts := make(map[string]any)
	opts["work_dir"] = workspace

	// Copy model from original agent if possible
	if ma, ok := e.agent.(interface{ GetModel() string }); ok {
		if m := ma.GetModel(); m != "" {
			opts["model"] = m
		}
	}
	// Copy permission mode
	if ma, ok := e.agent.(interface{ GetMode() string }); ok {
		if m := ma.GetMode(); m != "" {
			opts["mode"] = m
		}
	}

	agent, err := CreateAgent(e.agent.Name(), opts)
	if err != nil {
		return nil, nil, fmt.Errorf("create workspace agent for %s: %w", workspace, err)
	}

	// Wire providers if original agent has them
	if ps, ok := e.agent.(ProviderSwitcher); ok {
		if ps2, ok2 := agent.(ProviderSwitcher); ok2 {
			ps2.SetProviders(ps.ListProviders())
		}
	}

	// Create per-workspace session manager
	h := sha256.Sum256([]byte(workspace))
	sessionFile := filepath.Join(filepath.Dir(e.sessions.StorePath()),
		fmt.Sprintf("%s_ws_%s.json", e.name, hex.EncodeToString(h[:4])))
	sessions := NewSessionManager(sessionFile)

	ws.agent = agent
	ws.sessions = sessions
	return agent, sessions, nil
}

func (e *Engine) getOrCreateInteractiveState(sessionKey string, p Platform, replyCtx any, session *Session) *interactiveState {
	return e.getOrCreateInteractiveStateWith(sessionKey, p, replyCtx, session, nil)
}

// getOrCreateInteractiveStateWith is like getOrCreateInteractiveState but accepts
// an optional agent override for multi-workspace mode. When agentOverride is non-nil
// it is used instead of e.agent to start the session.
func (e *Engine) getOrCreateInteractiveStateWith(sessionKey string, p Platform, replyCtx any, session *Session, agentOverride Agent) *interactiveState {
	e.interactiveMu.Lock()
	defer e.interactiveMu.Unlock()

	state, ok := e.interactiveStates[sessionKey]
	if ok && state.agentSession != nil && state.agentSession.Alive() {
		// Verify the running agent session matches the current active session.
		// After /new or /switch the active session changes, but the old agent
		// process may still be alive. Reusing it would send messages to the
		// wrong conversation context.
		session.mu.Lock()
		wantID := session.AgentSessionID
		session.mu.Unlock()
		currentID := state.agentSession.CurrentSessionID()
		if wantID == "" || currentID == "" || wantID == currentID {
			return state
		}
		// Active session has changed — tear down the stale agent so we can
		// start a new one that matches the current session below.
		slog.Info("interactive session mismatch, recycling",
			"session_key", sessionKey,
			"want_agent_session", wantID,
			"have_agent_session", currentID,
		)
		go state.agentSession.Close()
		delete(e.interactiveStates, sessionKey)
		ok = false // prevent reading stale settings below
	}

	// Preserve quiet setting from existing state (e.g. set via /quiet before session started)
	quietMode := e.defaultQuiet
	if ok && state != nil {
		state.mu.Lock()
		quietMode = state.quiet
		state.mu.Unlock()
	}

	// Select the agent to use for this session
	agent := e.agent
	if agentOverride != nil {
		agent = agentOverride
	}

	// Inject per-session env vars so the agent subprocess can call `cc-connect cron add` etc.
	if inj, ok := agent.(SessionEnvInjector); ok {
		envVars := []string{
			"CC_PROJECT=" + e.name,
			"CC_SESSION_KEY=" + sessionKey,
		}
		if exePath, err := os.Executable(); err == nil {
			binDir := filepath.Dir(exePath)
			if curPath := os.Getenv("PATH"); curPath != "" {
				envVars = append(envVars, "PATH="+binDir+string(filepath.ListSeparator)+curPath)
			} else {
				envVars = append(envVars, "PATH="+binDir)
			}
		}
		inj.SetSessionEnv(envVars)
	}

	// Check if context is already canceled (e.g. during shutdown/restart)
	if e.ctx.Err() != nil {
		slog.Debug("skipping session start: context canceled", "session_key", sessionKey)
		state = &interactiveState{platform: p, replyCtx: replyCtx, quiet: quietMode}
		e.interactiveStates[sessionKey] = state
		return state
	}

	startAt := time.Now()
	agentSession, err := agent.StartSession(e.ctx, session.AgentSessionID)
	startElapsed := time.Since(startAt)
	if err != nil {
		slog.Error("failed to start interactive session", "error", err, "elapsed", startElapsed)
		state = &interactiveState{platform: p, replyCtx: replyCtx, quiet: quietMode}
		e.interactiveStates[sessionKey] = state
		return state
	}
	if startElapsed >= slowAgentStart {
		slog.Warn("slow agent session start", "elapsed", startElapsed, "agent", agent.Name(), "session_id", session.AgentSessionID)
	}

	// Immediately capture the agent-side session ID so that if the agent
	// process crashes before emitting its first session_id event we still
	// have the binding. The relay path already does this (see HandleRelay);
	// the interactive path was missing it, leaving a window where the local
	// session could lose its agent binding.
	if newID := agentSession.CurrentSessionID(); newID != "" {
		session.CompareAndSetAgentSessionID(newID, agent.Name())
	}

	state = &interactiveState{
		agentSession: agentSession,
		platform:     p,
		replyCtx:     replyCtx,
		quiet:        quietMode,
	}
	e.interactiveStates[sessionKey] = state

	slog.Info("interactive session started", "session_key", sessionKey, "agent_session", session.AgentSessionID, "elapsed", startElapsed)
	return state
}

// cleanupInteractiveState removes the interactive state for the given session key
// and closes its agent session. When an expected state is provided, cleanup is
// skipped if the map entry has been replaced by a different state — this prevents
// a stale goroutine (still running after /new created a fresh Session object and
// a new turn started on it) from accidentally destroying the replacement state.
func (e *Engine) cleanupInteractiveState(sessionKey string, expected ...*interactiveState) {
	e.interactiveMu.Lock()
	state, ok := e.interactiveStates[sessionKey]
	if len(expected) > 0 && expected[0] != nil && state != expected[0] {
		// Another turn has already replaced the state — skip cleanup.
		e.interactiveMu.Unlock()
		return
	}
	delete(e.interactiveStates, sessionKey)
	e.interactiveMu.Unlock()

	if ok && state != nil && state.agentSession != nil {
		slog.Debug("cleanupInteractiveState: closing agent session", "session", sessionKey)
		closeStart := time.Now()

		done := make(chan struct{})
		go func() {
			state.agentSession.Close()
			close(done)
		}()

		select {
		case <-done:
			if elapsed := time.Since(closeStart); elapsed >= slowAgentClose {
				slog.Warn("slow agent session close", "elapsed", elapsed, "session", sessionKey)
			}
		case <-time.After(10 * time.Second):
			slog.Error("agent session close timed out (10s), abandoning", "session", sessionKey)
		}
	}
}

const defaultEventIdleTimeout = 2 * time.Hour

func (e *Engine) processInteractiveEvents(state *interactiveState, session *Session, sessionKey string, msg *Message, inputText string, agent Agent, msgID string, turnStart time.Time, networkStart *podNetworkSnapshot) {
	var textParts []string
	toolCount := 0
	waitStart := time.Now()
	firstEventLogged := false
	var latestUsage *UsageMetrics

	state.mu.Lock()
	sp := newStreamPreview(e.streamPreview, state.platform, state.replyCtx, e.ctx)
	state.mu.Unlock()

	// Idle timeout: 0 = disabled
	var idleTimer *time.Timer
	var idleCh <-chan time.Time
	if e.eventIdleTimeout > 0 {
		idleTimer = time.NewTimer(e.eventIdleTimeout)
		defer idleTimer.Stop()
		idleCh = idleTimer.C
	}

	events := state.agentSession.Events()
	for {
		var event Event
		var ok bool

		select {
		case event, ok = <-events:
			if !ok {
				goto channelClosed
			}
		case <-idleCh:
			slog.Error("agent session idle timeout: no events for too long, killing session",
				"session_key", sessionKey, "timeout", e.eventIdleTimeout, "elapsed", time.Since(turnStart))
			sp.finish("")
			state.mu.Lock()
			p := state.platform
			replyCtx := state.replyCtx
			state.mu.Unlock()
			e.send(p, replyCtx, fmt.Sprintf(e.i18n.T(MsgError), "agent session timed out (no response)"))
			e.cleanupInteractiveState(sessionKey, state)
			return
		case <-e.ctx.Done():
			return
		}

		// Reset idle timer after receiving an event
		if idleTimer != nil {
			if !idleTimer.Stop() {
				select {
				case <-idleTimer.C:
				default:
				}
			}
			idleTimer.Reset(e.eventIdleTimeout)
		}

		if !firstEventLogged {
			firstEventLogged = true
			if elapsed := time.Since(waitStart); elapsed >= slowAgentFirstEvent {
				slog.Warn("slow agent first event", "elapsed", elapsed, "session", sessionKey, "event_type", event.Type)
			}
		}

		state.mu.Lock()
		p := state.platform
		replyCtx := state.replyCtx
		sessionQuiet := state.quiet
		state.mu.Unlock()

		e.quietMu.RLock()
		globalQuiet := e.quiet
		e.quietMu.RUnlock()

		quiet := globalQuiet || sessionQuiet

		switch event.Type {
		case EventThinking:
			if !quiet && event.Content != "" {
				sp.freeze()
				preview := truncateIf(event.Content, e.display.ThinkingMaxLen)
				e.send(p, replyCtx, fmt.Sprintf(e.i18n.T(MsgThinking), preview))
			}

		case EventToolUse:
			toolCount++
			if !quiet {
				sp.freeze()
				inputPreview := truncateIf(event.ToolInput, e.display.ToolMaxLen)
				// Use code block if content is long (>5 lines or >200 chars), otherwise inline code
				lineCount := strings.Count(inputPreview, "\n") + 1
				var formattedInput string
				if lineCount > 5 || utf8.RuneCountInString(inputPreview) > 200 {
					formattedInput = fmt.Sprintf("```\n%s\n```", inputPreview)
				} else {
					formattedInput = fmt.Sprintf("`%s`", inputPreview)
				}
				e.send(p, replyCtx, fmt.Sprintf(e.i18n.T(MsgTool), toolCount, event.ToolName, formattedInput))
			}

		case EventText:
			if event.Usage != nil {
				latestUsage = event.Usage
			}
			if event.Content != "" {
				textParts = append(textParts, event.Content)
				if sp.canPreview() {
					sp.appendText(event.Content)
				}
			}
			if event.SessionID != "" {
				if session.CompareAndSetAgentSessionID(event.SessionID, agent.Name()) {
					session.mu.Lock()
					pendingName := session.Name
					session.mu.Unlock()
					if pendingName != "" && pendingName != "session" && pendingName != "default" {
						e.sessions.SetSessionName(event.SessionID, pendingName)
					}
					e.sessions.Save()
				}
			}

		case EventPermissionRequest:
			isAskQuestion := event.ToolName == "AskUserQuestion" && len(event.Questions) > 0

			state.mu.Lock()
			autoApprove := state.approveAll
			state.mu.Unlock()

			if autoApprove && !isAskQuestion {
				slog.Debug("auto-approving (approve-all)", "request_id", event.RequestID, "tool", event.ToolName)
				_ = state.agentSession.RespondPermission(event.RequestID, PermissionResult{
					Behavior:     "allow",
					UpdatedInput: event.ToolInputRaw,
				})
				continue
			}

			// Stop streaming preview before sending prompt
			sp.freeze()

			slog.Info("permission request",
				"request_id", event.RequestID,
				"tool", event.ToolName,
			)

			if isAskQuestion {
				e.sendAskQuestionPrompt(p, replyCtx, event.Questions, 0)
			} else {
				permLimit := e.display.ToolMaxLen
				if permLimit > 0 {
					permLimit = permLimit * 8 / 5
				}
				toolInput := truncateIf(event.ToolInput, permLimit)
				prompt := fmt.Sprintf(e.i18n.T(MsgPermissionPrompt), event.ToolName, toolInput)
				e.sendPermissionPrompt(p, replyCtx, prompt, event.ToolName, toolInput)
			}

			pending := &pendingPermission{
				RequestID:    event.RequestID,
				ToolName:     event.ToolName,
				ToolInput:    event.ToolInputRaw,
				InputPreview: event.ToolInput,
				Questions:    event.Questions,
				Resolved:     make(chan struct{}),
			}
			state.mu.Lock()
			state.pending = pending
			state.mu.Unlock()

			// Stop idle timer while waiting for user permission response;
			// the user may take a long time to decide, and we don't want
			// the idle timeout to kill the session during that wait.
			if idleTimer != nil {
				idleTimer.Stop()
			}

			<-pending.Resolved
			slog.Info("permission resolved", "request_id", event.RequestID)

			// Restart idle timer after permission is resolved
			if idleTimer != nil {
				idleTimer.Reset(e.eventIdleTimeout)
			}

		case EventResult:
			if event.Usage != nil {
				latestUsage = event.Usage
			}
			if event.SessionID != "" {
				session.SetAgentSessionID(event.SessionID, agent.Name())
			}

			fullResponse := event.Content
			if fullResponse == "" && len(textParts) > 0 {
				fullResponse = strings.Join(textParts, "")
			}
			if fullResponse == "" {
				fullResponse = e.i18n.T(MsgEmptyResponse)
			}

			session.AddHistory("assistant", fullResponse)
			e.sessions.Save()

			turnDuration := time.Since(turnStart)
			slog.Info("turn complete",
				"session", session.ID,
				"agent_session", session.AgentSessionID,
				"msg_id", msgID,
				"tools", toolCount,
				"response_len", len(fullResponse),
				"turn_duration", turnDuration,
			)

			replyStart := time.Now()

			// If streaming preview was active, try to finalize in-place
			if sp.finish(fullResponse) {
				slog.Debug("EventResult: finalized via stream preview", "response_len", len(fullResponse))
			} else {
				slog.Debug("EventResult: sending via p.Send (preview inactive or failed)", "response_len", len(fullResponse), "chunks", len(splitMessage(fullResponse, maxPlatformMessageLen)))
				for _, chunk := range splitMessage(fullResponse, maxPlatformMessageLen) {
					if err := p.Send(e.ctx, replyCtx, chunk); err != nil {
						slog.Error("failed to send reply", "error", err, "msg_id", msgID)
						e.logUsageAsync(msg, agent, inputText, fullResponse, latestUsage, int(time.Since(turnStart).Milliseconds()), "error", fmt.Sprintf("failed to send reply: %v", err), measureTurnNetwork(networkStart))
						return
					}
				}
			}

			if elapsed := time.Since(replyStart); elapsed >= slowPlatformSend {
				slog.Warn("slow final reply send", "platform", p.Name(), "elapsed", elapsed, "response_len", len(fullResponse))
			}

			// TTS: async voice reply if enabled
			if e.tts != nil && e.tts.Enabled && e.tts.TTS != nil {
				state.mu.Lock()
				fromVoice := state.fromVoice
				state.mu.Unlock()
				mode := e.tts.GetTTSMode()
				if mode == "always" || (mode == "voice_only" && fromVoice) {
					go e.sendTTSReply(p, replyCtx, fullResponse)
				}
			}

			e.logUsageAsync(msg, agent, inputText, fullResponse, latestUsage, int(turnDuration.Milliseconds()), "success", "", measureTurnNetwork(networkStart))

			return

		case EventError:
			sp.finish("") // clean up preview on error
			if event.Usage != nil {
				latestUsage = event.Usage
			}
			if event.Error != nil {
				slog.Error("agent error", "error", event.Error)
				e.logUsageAsync(msg, agent, inputText, strings.Join(textParts, ""), latestUsage, int(time.Since(turnStart).Milliseconds()), "error", event.Error.Error(), measureTurnNetwork(networkStart))
				e.send(p, replyCtx, fmt.Sprintf(e.i18n.T(MsgError), event.Error))
			}
			return
		}
	}

channelClosed:
	// Channel closed - process exited unexpectedly
	slog.Warn("agent process exited", "session_key", sessionKey)
	e.cleanupInteractiveState(sessionKey, state)

	if len(textParts) > 0 {
		state.mu.Lock()
		p := state.platform
		replyCtx := state.replyCtx
		state.mu.Unlock()

		fullResponse := strings.Join(textParts, "")
		session.AddHistory("assistant", fullResponse)

		if sp.finish(fullResponse) {
			slog.Debug("stream preview: finalized in-place (process exited)")
		} else {
			for _, chunk := range splitMessage(fullResponse, maxPlatformMessageLen) {
				e.send(p, replyCtx, chunk)
			}
		}

		e.logUsageAsync(msg, agent, inputText, fullResponse, latestUsage, int(time.Since(turnStart).Milliseconds()), "error", "agent process exited unexpectedly", measureTurnNetwork(networkStart))
		return
	}

	e.logUsageAsync(msg, agent, inputText, "", latestUsage, int(time.Since(turnStart).Milliseconds()), "error", "agent process exited unexpectedly", measureTurnNetwork(networkStart))
}
