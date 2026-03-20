package core

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"time"
)

func (e *Engine) logUsageAsync(msg *Message, agent Agent, inputText, outputText string, usage *UsageMetrics, latencyMs int, status, errorMessage string, networkUsage NetworkUsage) {
	if e.authWebhookURL == "" || msg == nil {
		return
	}

	logURL, err := usageLogURL(e.authWebhookURL)
	if err != nil {
		slog.Warn("usage log: invalid auth webhook URL", "error", err)
		return
	}

	model := "unknown"
	if agent == nil {
		model = "shell"
	} else {
		if switcher, ok := agent.(interface{ GetModel() string }); ok {
			if m := switcher.GetModel(); m != "" {
				model = m
			}
		}
	}

	inputTokens := 0
	outputTokens := 0
	costCents := 0
	if usage != nil {
		inputTokens = usage.InputTokens
		outputTokens = usage.OutputTokens
		costCents = usage.CostCents
	}

	payload := map[string]any{
		"platform":          msg.Platform,
		"user_id":           msg.UserID,
		"session_key":       msg.SessionKey,
		"model":             model,
		"input_text":        inputText,
		"input_tokens":      inputTokens,
		"output_text":       outputText,
		"output_tokens":     outputTokens,
		"latency_ms":        latencyMs,
		"cost_cents":        costCents,
		"status":            status,
		"error_message":     errorMessage,
		"network_bytes_in":  networkUsage.BytesIn,
		"network_bytes_out": networkUsage.BytesOut,
		"compute_seconds":   0,
	}

	go func() {
		body, _ := json.Marshal(payload)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, logURL, bytes.NewReader(body))
		if err != nil {
			slog.Warn("usage log: request creation failed", "error", err)
			return
		}
		req.Header.Set("Content-Type", "application/json")
		if e.authWebhookSecret != "" {
			req.Header.Set("X-Webhook-Secret", e.authWebhookSecret)
		}

		resp, err := (&http.Client{Timeout: 5 * time.Second}).Do(req)
		if err != nil {
			slog.Warn("usage log: request failed", "error", err)
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			slog.Warn("usage log: non-200 response", "status", resp.StatusCode)
		}
	}()
}

func usageLogURL(authWebhookURL string) (string, error) {
	parsed, err := url.Parse(authWebhookURL)
	if err != nil {
		return "", err
	}
	if strings.HasSuffix(parsed.Path, "/verify") {
		parsed.Path = strings.TrimSuffix(parsed.Path, "/verify") + "/log"
		return parsed.String(), nil
	}
	return "", fmt.Errorf("auth webhook path %q does not end with /verify", parsed.Path)
}
