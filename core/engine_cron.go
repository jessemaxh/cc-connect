package core

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"strings"
	"time"
)

// ExecuteCronJob runs a cron job by injecting a synthetic message into the engine.
// It finds the platform that owns the session key, reconstructs a reply context,
// and processes the message as if the user sent it.
func (e *Engine) ExecuteCronJob(job *CronJob) error {
	sessionKey := job.SessionKey
	platformName := ""
	if idx := strings.Index(sessionKey, ":"); idx > 0 {
		platformName = sessionKey[:idx]
	}

	var targetPlatform Platform
	for _, p := range e.platforms {
		if p.Name() == platformName {
			targetPlatform = p
			break
		}
	}
	if targetPlatform == nil {
		return fmt.Errorf("platform %q not found for session %q", platformName, sessionKey)
	}

	rc, ok := targetPlatform.(ReplyContextReconstructor)
	if !ok {
		return fmt.Errorf("platform %q does not support proactive messaging (cron)", platformName)
	}

	replyCtx, err := rc.ReconstructReplyCtx(sessionKey)
	if err != nil {
		return fmt.Errorf("reconstruct reply context: %w", err)
	}

	msg := &Message{
		SessionKey: sessionKey,
		Platform:   platformName,
		UserID:     "cron",
		UserName:   "cron",
		Content:    job.Prompt,
		ReplyCtx:   replyCtx,
	}

	if job.IsShellJob() && msg.Content == "" {
		msg.Content = job.Exec
	}

	if e.authWebhookURL != "" && !e.checkAuthWebhook(targetPlatform, msg) {
		return fmt.Errorf("cron denied by auth/quota policy")
	}

	// Notify user that a cron job is executing (unless silent)
	silent := false
	if e.cronScheduler != nil {
		silent = e.cronScheduler.IsSilent(job)
	}
	if !silent {
		desc := job.Description
		if desc == "" {
			if job.IsShellJob() {
				desc = truncateStr(job.Exec, 40)
			} else {
				desc = truncateStr(job.Prompt, 40)
			}
		}
		e.send(targetPlatform, replyCtx, fmt.Sprintf("⏰ %s", desc))
	}

	if job.IsShellJob() {
		return e.executeCronShell(targetPlatform, replyCtx, msg, job)
	}

	session := e.sessions.GetOrCreateActive(sessionKey)
	if !session.TryLock() {
		return fmt.Errorf("session %q is busy", sessionKey)
	}

	e.processInteractiveMessage(targetPlatform, msg, session)
	return nil
}

// executeCronShell runs a shell command for a cron job and sends the output.
func (e *Engine) executeCronShell(p Platform, replyCtx any, msg *Message, job *CronJob) error {
	workDir := job.WorkDir
	if workDir == "" {
		if wd, ok := e.agent.(interface{ GetWorkDir() string }); ok {
			workDir = wd.GetWorkDir()
		}
	}
	if workDir == "" {
		workDir, _ = os.Getwd()
	}

	startedAt := time.Now()
	networkStart, netErr := capturePodNetworkSnapshot()
	if netErr != nil {
		slog.Debug("network usage: shell cron start snapshot unavailable", "error", netErr)
	}

	ctx, cancel := context.WithTimeout(e.ctx, cronJobTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", job.Exec)
	cmd.Dir = workDir
	output, err := cmd.CombinedOutput()

	if ctx.Err() == context.DeadlineExceeded {
		timeoutMsg := fmt.Sprintf("⏰ ⚠️ timeout: `%s`", truncateStr(job.Exec, 60))
		e.send(p, replyCtx, timeoutMsg)
		e.logUsageAsync(msg, nil, job.Exec, "", nil, int(time.Since(startedAt).Milliseconds()), "error", "shell command timed out", measureTurnNetwork(networkStart))
		return fmt.Errorf("shell command timed out")
	}

	result := strings.TrimSpace(string(output))
	if err != nil {
		if result != "" {
			e.send(p, replyCtx, fmt.Sprintf("⏰ ❌ `%s`\n\n%s\n\nerror: %v", truncateStr(job.Exec, 60), truncateStr(result, 3000), err))
		} else {
			e.send(p, replyCtx, fmt.Sprintf("⏰ ❌ `%s`\nerror: %v", truncateStr(job.Exec, 60), err))
		}
		e.logUsageAsync(msg, nil, job.Exec, result, nil, int(time.Since(startedAt).Milliseconds()), "error", err.Error(), measureTurnNetwork(networkStart))
		return fmt.Errorf("shell: %w", err)
	}

	if result == "" {
		result = "(no output)"
	}
	e.send(p, replyCtx, fmt.Sprintf("⏰ ✅ `%s`\n\n%s", truncateStr(job.Exec, 60), truncateStr(result, 3000)))
	e.logUsageAsync(msg, nil, job.Exec, result, nil, int(time.Since(startedAt).Milliseconds()), "success", "", measureTurnNetwork(networkStart))
	return nil
}

// ExecuteHeartbeat runs a heartbeat check by injecting a synthetic message
// into the main session, similar to cron but designed for periodic awareness.
func (e *Engine) ExecuteHeartbeat(sessionKey, prompt string, silent bool) error {
	platformName := ""
	if idx := strings.Index(sessionKey, ":"); idx > 0 {
		platformName = sessionKey[:idx]
	}

	var targetPlatform Platform
	for _, p := range e.platforms {
		if p.Name() == platformName {
			targetPlatform = p
			break
		}
	}
	if targetPlatform == nil {
		return fmt.Errorf("platform %q not found for session %q", platformName, sessionKey)
	}

	rc, ok := targetPlatform.(ReplyContextReconstructor)
	if !ok {
		return fmt.Errorf("platform %q does not support proactive messaging (heartbeat)", platformName)
	}

	replyCtx, err := rc.ReconstructReplyCtx(sessionKey)
	if err != nil {
		return fmt.Errorf("reconstruct reply context: %w", err)
	}

	if !silent {
		e.send(targetPlatform, replyCtx, "💓 heartbeat")
	}

	msg := &Message{
		SessionKey: sessionKey,
		Platform:   platformName,
		UserID:     "heartbeat",
		UserName:   "heartbeat",
		Content:    prompt,
		ReplyCtx:   replyCtx,
	}

	session := e.sessions.GetOrCreateActive(sessionKey)
	if !session.TryLock() {
		return fmt.Errorf("session %q is busy", sessionKey)
	}

	e.processInteractiveMessage(targetPlatform, msg, session)
	return nil
}
