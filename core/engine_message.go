package core

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// matchBannedWord returns the first banned word found in content, or "".
func (e *Engine) matchBannedWord(content string) string {
	e.bannedMu.RLock()
	defer e.bannedMu.RUnlock()
	if len(e.bannedWords) == 0 {
		return ""
	}
	lower := strings.ToLower(content)
	for _, w := range e.bannedWords {
		if strings.Contains(lower, w) {
			return w
		}
	}
	return ""
}

// resolveAlias checks if the content (or its first word) matches an alias and replaces it.
func (e *Engine) resolveAlias(content string) string {
	e.aliasMu.RLock()
	defer e.aliasMu.RUnlock()

	if len(e.aliases) == 0 {
		return content
	}

	// Exact match on full content
	if cmd, ok := e.aliases[content]; ok {
		return cmd
	}

	// Match first word, append remaining args
	parts := strings.SplitN(content, " ", 2)
	if cmd, ok := e.aliases[parts[0]]; ok {
		if len(parts) > 1 {
			return cmd + " " + parts[1]
		}
		return cmd
	}
	return content
}

// checkAuthWebhook calls the external auth webhook to verify the message.
// Returns true if the message should be processed, false if denied.
// Fail-closed: any error (network, non-200, decode) denies the message and
// notifies the user, so a broken webhook never silently bypasses auth/quota.
func (e *Engine) checkAuthWebhook(p Platform, msg *Message) bool {
	type webhookReq struct {
		Platform   string `json:"platform"`
		UserID     string `json:"user_id"`
		UserName   string `json:"user_name"`
		SessionKey string `json:"session_key"`
	}
	type webhookResp struct {
		Allowed bool   `json:"allowed"`
		Message string `json:"message,omitempty"` // denial message to send to user
	}

	body, _ := json.Marshal(webhookReq{
		Platform:   msg.Platform,
		UserID:     msg.UserID,
		UserName:   msg.UserName,
		SessionKey: msg.SessionKey,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, "POST", e.authWebhookURL, bytes.NewReader(body))
	if err != nil {
		slog.Error("auth webhook: request creation failed, denying message", "error", err)
		e.reply(p, msg.ReplyCtx, "⚠️ Service temporarily unavailable. Please try again later.")
		return false
	}
	req.Header.Set("Content-Type", "application/json")
	if e.authWebhookSecret != "" {
		req.Header.Set("X-Webhook-Secret", e.authWebhookSecret)
	}

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		slog.Error("auth webhook: request failed, denying message", "error", err)
		e.reply(p, msg.ReplyCtx, "⚠️ Service temporarily unavailable. Please try again later.")
		return false
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		slog.Error("auth webhook: non-200 response, denying message", "status", resp.StatusCode)
		e.reply(p, msg.ReplyCtx, "⚠️ Service temporarily unavailable. Please try again later.")
		return false
	}

	var result webhookResp
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		slog.Error("auth webhook: decode failed, denying message", "error", err)
		e.reply(p, msg.ReplyCtx, "⚠️ Service temporarily unavailable. Please try again later.")
		return false
	}

	if !result.Allowed {
		if result.Message != "" {
			e.reply(p, msg.ReplyCtx, result.Message)
		}
		slog.Info("auth webhook: message denied", "user", msg.UserID, "reason", result.Message)
		return false
	}

	return true
}

func (e *Engine) handleMessage(p Platform, msg *Message) {
	slog.Info("message received",
		"platform", msg.Platform, "msg_id", msg.MessageID,
		"session", msg.SessionKey, "user", msg.UserName,
		"content_len", len(msg.Content),
		"has_images", len(msg.Images) > 0, "has_audio", msg.Audio != nil, "has_files", len(msg.Files) > 0,
	)

	// Auth webhook check
	if e.authWebhookURL != "" {
		if !e.checkAuthWebhook(p, msg) {
			return // webhook denied or returned an error message
		}
	}

	// Voice message: transcribe to text first
	if msg.Audio != nil {
		e.handleVoiceMessage(p, msg)
		return
	}

	content := strings.TrimSpace(msg.Content)
	if content == "" && len(msg.Images) == 0 && len(msg.Files) == 0 {
		return
	}

	// Resolve aliases: check if the first word (or whole content) matches an alias
	content = e.resolveAlias(content)
	msg.Content = content

	// Rate limit check
	if e.rateLimiter != nil && !e.rateLimiter.Allow(msg.SessionKey) {
		slog.Info("message rate limited", "session", msg.SessionKey, "user", msg.UserName)
		e.reply(p, msg.ReplyCtx, e.i18n.T(MsgRateLimited))
		return
	}

	// Banned words check (skip for slash commands)
	if !strings.HasPrefix(content, "/") {
		if word := e.matchBannedWord(content); word != "" {
			slog.Info("message blocked by banned word", "word", word, "user", msg.UserName)
			e.reply(p, msg.ReplyCtx, e.i18n.T(MsgBannedWordBlocked))
			return
		}
	}

	// Multi-workspace resolution
	var wsAgent Agent
	var wsSessions *SessionManager
	var resolvedWorkspace string
	if e.multiWorkspace {
		channelID := extractChannelID(msg.SessionKey)
		workspace, channelName, err := e.resolveWorkspace(p, channelID)
		if err != nil {
			slog.Error("workspace resolution failed", "err", err)
			e.reply(p, msg.ReplyCtx, e.i18n.Tf(MsgWsResolutionError, err))
			return
		}
		if workspace == "" {
			// No workspace — handle init flow (unless it's a /workspace command)
			if !strings.HasPrefix(content, "workspace") && !strings.HasPrefix(content, "ws ") {
				if e.handleWorkspaceInitFlow(p, msg, channelID, channelName) {
					return
				}
			}
			// If init flow didn't consume, only workspace commands work
			if !strings.HasPrefix(content, "/") {
				return
			}
		} else {
			resolvedWorkspace = workspace

			// Touch for idle tracking
			if ws := e.workspacePool.Get(workspace); ws != nil {
				ws.Touch()
			}

			// Get or create the workspace's agent and session manager
			wsAgent, wsSessions, err = e.getOrCreateWorkspaceAgent(workspace)
			if err != nil {
				slog.Error("failed to create workspace agent", "workspace", workspace, "err", err)
				e.reply(p, msg.ReplyCtx, fmt.Sprintf("Failed to initialize workspace: %v", err))
				return
			}
		}
	}

	// First message welcome: inject skill awareness into first user message.
	// Entries auto-expire after 24h (cleaned by runIdleReaper) so the map doesn't grow unbounded.
	if _, loaded := e.welcomeSent.LoadOrStore(msg.SessionKey, time.Now()); !loaded {
		if !strings.HasPrefix(content, "/") {
			welcomeCtx := e.buildWelcomeContext()
			if welcomeCtx != "" {
				msg.Content = msg.Content + "\n\n" + welcomeCtx
				content = msg.Content
			}
		}
	}

	if len(msg.Images) == 0 && strings.HasPrefix(content, "/") {
		if e.handleCommand(p, msg, content) {
			return
		}
		// Unrecognized slash command — fall through to agent as normal message
	}

	// Permission responses bypass the session lock
	if e.handlePendingPermission(p, msg, content) {
		return
	}

	// Select session manager and agent based on workspace mode
	sessions := e.sessions
	agent := e.agent
	interactiveKey := msg.SessionKey
	if e.multiWorkspace && wsSessions != nil {
		sessions = wsSessions
		agent = wsAgent
		interactiveKey = resolvedWorkspace + ":" + msg.SessionKey
	}

	session := sessions.GetOrCreateActive(msg.SessionKey)
	if !session.TryLock() {
		e.reply(p, msg.ReplyCtx, e.i18n.T(MsgPreviousProcessing))
		return
	}

	slog.Info("processing message",
		"platform", msg.Platform,
		"user", msg.UserName,
		"session", session.ID,
	)

	go e.processInteractiveMessageWith(p, msg, session, agent, interactiveKey, resolvedWorkspace)
}

// ──────────────────────────────────────────────────────────────
// Voice message handling
// ──────────────────────────────────────────────────────────────

func (e *Engine) handleVoiceMessage(p Platform, msg *Message) {
	if !e.speech.Enabled || e.speech.STT == nil {
		e.reply(p, msg.ReplyCtx, e.i18n.T(MsgVoiceNotEnabled))
		return
	}

	audio := msg.Audio
	if NeedsConversion(audio.Format) && !HasFFmpeg() {
		e.reply(p, msg.ReplyCtx, e.i18n.T(MsgVoiceNoFFmpeg))
		return
	}

	slog.Info("transcribing voice message",
		"platform", msg.Platform, "user", msg.UserName,
		"format", audio.Format, "size", len(audio.Data),
	)
	e.send(p, msg.ReplyCtx, e.i18n.T(MsgVoiceTranscribing))

	text, err := TranscribeAudio(e.ctx, e.speech.STT, audio, e.speech.Language)
	if err != nil {
		slog.Error("speech transcription failed", "error", err)
		e.reply(p, msg.ReplyCtx, fmt.Sprintf(e.i18n.T(MsgVoiceTranscribeFailed), err))
		return
	}

	text = strings.TrimSpace(text)
	if text == "" {
		e.reply(p, msg.ReplyCtx, e.i18n.T(MsgVoiceEmpty))
		return
	}

	slog.Info("voice transcribed", "text_len", len(text))
	e.send(p, msg.ReplyCtx, fmt.Sprintf(e.i18n.T(MsgVoiceTranscribed), text))

	// Replace audio with transcribed text and re-dispatch
	msg.Audio = nil
	msg.Content = text
	msg.FromVoice = true
	e.handleMessage(p, msg)
}

// buildWelcomeContext returns a hidden context appended to the first user message,
// prompting the AI to naturally introduce its capabilities.
func (e *Engine) buildWelcomeContext() string {
	return `[System: This is the user's first message. After answering their question, briefly mention 1-2 of your key capabilities (e.g. weather, search, reminders) so they know what you can do. Keep it natural and short — one sentence is enough. Do NOT list all capabilities.]`
}
