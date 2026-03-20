package core

import (
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const ccConnectInstructionMarker = "<!-- cc-connect-instructions -->"

func extractChannelID(sessionKey string) string {
	// Format: "platform:channelID:userID" or "platform:channelID"
	parts := strings.SplitN(sessionKey, ":", 3)
	if len(parts) >= 2 {
		return parts[1]
	}
	return ""
}

// commandContext resolves the appropriate agent, session manager, and interactive key
// for a command. In multi-workspace mode, it routes to the bound workspace if present.
func (e *Engine) commandContext(p Platform, msg *Message) (Agent, *SessionManager, string, error) {
	if !e.multiWorkspace {
		return e.agent, e.sessions, msg.SessionKey, nil
	}
	channelID := extractChannelID(msg.SessionKey)
	if channelID == "" {
		return e.agent, e.sessions, msg.SessionKey, nil
	}
	workspace, _, err := e.resolveWorkspace(p, channelID)
	if err != nil {
		return nil, nil, "", err
	}
	if workspace == "" {
		return e.agent, e.sessions, msg.SessionKey, nil
	}
	wsAgent, wsSessions, err := e.getOrCreateWorkspaceAgent(workspace)
	if err != nil {
		return nil, nil, "", err
	}
	return wsAgent, wsSessions, workspace + ":" + msg.SessionKey, nil
}

// sessionContextForKey resolves the agent and session manager for a sessionKey.
// It uses existing workspace bindings and falls back to global context if unresolved.
func (e *Engine) sessionContextForKey(sessionKey string) (Agent, *SessionManager) {
	if !e.multiWorkspace || e.workspaceBindings == nil {
		return e.agent, e.sessions
	}
	channelID := extractChannelID(sessionKey)
	if channelID == "" {
		return e.agent, e.sessions
	}
	projectKey := "project:" + e.name
	if b := e.workspaceBindings.Lookup(projectKey, channelID); b != nil {
		if wsAgent, wsSessions, err := e.getOrCreateWorkspaceAgent(b.Workspace); err == nil {
			return wsAgent, wsSessions
		}
	}
	return e.agent, e.sessions
}

// interactiveKeyForSessionKey returns the interactive state key for a sessionKey.
// In multi-workspace mode, it prefixes with the bound workspace path when available.
func (e *Engine) interactiveKeyForSessionKey(sessionKey string) string {
	if !e.multiWorkspace || e.workspaceBindings == nil {
		return sessionKey
	}
	channelID := extractChannelID(sessionKey)
	if channelID == "" {
		return sessionKey
	}
	projectKey := "project:" + e.name
	if b := e.workspaceBindings.Lookup(projectKey, channelID); b != nil {
		return b.Workspace + ":" + sessionKey
	}
	return sessionKey
}

// resolveWorkspace resolves a channel to a workspace directory.
// Returns (workspacePath, channelName, error).
// If workspacePath is empty, the init flow should be triggered.
func (e *Engine) resolveWorkspace(p Platform, channelID string) (string, string, error) {
	projectKey := "project:" + e.name

	// Step 1: Check existing binding
	if b := e.workspaceBindings.Lookup(projectKey, channelID); b != nil {
		// Verify workspace directory still exists
		if _, err := os.Stat(b.Workspace); err != nil {
			slog.Warn("bound workspace directory missing, removing binding",
				"workspace", b.Workspace, "channel", channelID)
			e.workspaceBindings.Unbind(projectKey, channelID)
			return "", b.ChannelName, nil
		}
		return b.Workspace, b.ChannelName, nil
	}

	// Step 2: Resolve channel name for convention match
	channelName := ""
	if resolver, ok := p.(ChannelNameResolver); ok {
		name, err := resolver.ResolveChannelName(channelID)
		if err != nil {
			slog.Warn("failed to resolve channel name", "channel", channelID, "err", err)
		} else {
			channelName = name
		}
	}

	if channelName == "" {
		return "", "", nil
	}

	// Step 3: Convention match — check if base_dir/<channel-name> exists
	candidate := filepath.Join(e.baseDir, channelName)
	if info, err := os.Stat(candidate); err == nil && info.IsDir() {
		// Auto-bind
		e.workspaceBindings.Bind(projectKey, channelID, channelName, candidate)
		slog.Info("workspace auto-bound by convention",
			"channel", channelName, "workspace", candidate)
		return candidate, channelName, nil
	}

	return "", channelName, nil
}

// handleWorkspaceInitFlow manages the conversational workspace setup.
// Returns true if the message was consumed by the init flow.
func (e *Engine) handleWorkspaceInitFlow(p Platform, msg *Message, channelID, channelName string) bool {
	e.initFlowsMu.Lock()
	flow, exists := e.initFlows[channelID]
	e.initFlowsMu.Unlock()

	content := strings.TrimSpace(msg.Content)

	if !exists {
		if strings.HasPrefix(content, "/") {
			return false
		}
		e.initFlowsMu.Lock()
		e.initFlows[channelID] = &workspaceInitFlow{
			state:       "awaiting_url",
			channelName: channelName,
		}
		e.initFlowsMu.Unlock()
		e.reply(p, msg.ReplyCtx, e.i18n.T(MsgWsNotFoundHint))
		return true
	}

	switch flow.state {
	case "awaiting_url":
		if !looksLikeGitURL(content) {
			e.reply(p, msg.ReplyCtx, "That doesn't look like a git URL. Please provide a URL like `https://github.com/org/repo` or `git@github.com:org/repo.git`.")
			return true
		}
		repoName := extractRepoName(content)
		cloneTo := filepath.Join(e.baseDir, repoName)

		e.initFlowsMu.Lock()
		flow.repoURL = content
		flow.cloneTo = cloneTo
		flow.state = "awaiting_confirm"
		e.initFlowsMu.Unlock()

		e.reply(p, msg.ReplyCtx, fmt.Sprintf(
			"I'll clone `%s` to `%s` and bind it to this channel. OK? (yes/no)", content, cloneTo))
		return true

	case "awaiting_confirm":
		lower := strings.ToLower(content)
		if lower != "yes" && lower != "y" {
			e.initFlowsMu.Lock()
			delete(e.initFlows, channelID)
			e.initFlowsMu.Unlock()
			e.reply(p, msg.ReplyCtx, "Cancelled. Send a repo URL anytime to try again.")
			return true
		}

		e.reply(p, msg.ReplyCtx, fmt.Sprintf("Cloning `%s` to `%s`...", flow.repoURL, flow.cloneTo))

		if err := gitClone(flow.repoURL, flow.cloneTo); err != nil {
			e.initFlowsMu.Lock()
			delete(e.initFlows, channelID)
			e.initFlowsMu.Unlock()
			e.reply(p, msg.ReplyCtx, fmt.Sprintf("Clone failed: %v\nSend a repo URL to try again.", err))
			return true
		}

		projectKey := "project:" + e.name
		e.workspaceBindings.Bind(projectKey, channelID, flow.channelName, flow.cloneTo)

		e.initFlowsMu.Lock()
		delete(e.initFlows, channelID)
		e.initFlowsMu.Unlock()

		e.reply(p, msg.ReplyCtx, fmt.Sprintf(
			"Clone complete. Bound workspace `%s` to this channel. Ready.", flow.cloneTo))
		return true
	}

	return false
}

func looksLikeGitURL(s string) bool {
	return strings.HasPrefix(s, "https://") ||
		strings.HasPrefix(s, "http://") ||
		strings.HasPrefix(s, "git@") ||
		strings.HasPrefix(s, "ssh://")
}

func extractRepoName(url string) string {
	url = strings.TrimSuffix(url, ".git")
	var name string
	// Handle git@host:org/repo format
	if idx := strings.LastIndex(url, ":"); idx != -1 && strings.HasPrefix(url, "git@") {
		remainder := url[idx+1:]
		parts := strings.Split(remainder, "/")
		if len(parts) > 0 {
			name = parts[len(parts)-1]
		}
	}
	if name == "" {
		// Handle https://host/org/repo format
		parts := strings.Split(url, "/")
		if len(parts) > 0 {
			name = parts[len(parts)-1]
		}
	}
	if name == "" {
		return "workspace"
	}
	// Sanitize: strip path separators and reject traversal components.
	name = strings.ReplaceAll(name, "/", "")
	name = strings.ReplaceAll(name, "\\", "")
	name = strings.ReplaceAll(name, "..", "")
	if name == "" || name == "." {
		return "workspace"
	}
	return name
}

func gitClone(repoURL, dest string) error {
	cmd := exec.Command("git", "clone", "--", repoURL, dest)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%s: %w", strings.TrimSpace(string(output)), err)
	}
	return nil
}
