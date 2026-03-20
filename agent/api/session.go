package api

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/chenhg5/cc-connect/core"
	"github.com/chenhg5/cc-connect/mcp"
)

const (
	// maxHistoryMessages caps the sliding window of conversation history.
	maxHistoryMessages = 40
	// maxToolLoopIterations prevents infinite tool-call loops.
	maxToolLoopIterations = 20
)

// apiSession manages a single multi-turn conversation via direct LLM API calls.
// It supports an agentic tool-calling loop using MCP servers.
type apiSession struct {
	llm    *llmClient
	mcp    *mcp.Manager
	model  string
	system string

	history   []chatMessage
	historyMu sync.Mutex

	sessionID string
	events    chan core.Event
	// closeEventsOnce ensures events is closed exactly once, even if Close() is
	// called concurrently or the 10-second watchdog fires while agentLoop is active.
	closeEventsOnce sync.Once

	ctx    context.Context
	cancel context.CancelFunc
	alive  atomic.Bool
	wg     sync.WaitGroup

	// loopRunning ensures only one agentLoop runs at a time.
	loopRunning atomic.Bool
}

func newAPISession(
	ctx context.Context,
	llm *llmClient,
	mcpMgr *mcp.Manager,
	model, system, sessionID string,
) *apiSession {
	sCtx, cancel := context.WithCancel(ctx)
	s := &apiSession{
		llm:       llm,
		mcp:       mcpMgr,
		model:     model,
		system:    system,
		sessionID: sessionID,
		events:    make(chan core.Event, 64),
		ctx:       sCtx,
		cancel:    cancel,
	}
	s.alive.Store(true)
	return s
}

// Send implements core.AgentSession. It appends the user message to history
// and launches the agentic loop in a goroutine. Returns an error if the session
// is already processing a previous message.
func (s *apiSession) Send(prompt string, images []core.ImageAttachment, files []core.FileAttachment) error {
	if !s.alive.Load() {
		return fmt.Errorf("session is closed")
	}
	if !s.loopRunning.CompareAndSwap(false, true) {
		return fmt.Errorf("session is busy, previous message still being processed")
	}

	// Build user message content.
	var userMsg chatMessage
	if len(images) > 0 {
		parts := []contentPart{{Type: "text", Text: prompt}}
		for _, img := range images {
			dataURL := "data:" + img.MimeType + ";base64," + base64.StdEncoding.EncodeToString(img.Data)
			parts = append(parts, contentPart{
				Type:     "image_url",
				ImageURL: &imageURL{URL: dataURL},
			})
		}
		userMsg = chatMessage{Role: "user", Content: parts}
	} else {
		text := prompt
		if len(files) > 0 {
			// Write attachments to a session-scoped temp directory.
			// Cleaned up after the turn completes in agentLoop.
			tmpDir, err := os.MkdirTemp("", "cc-connect-api-*")
			if err != nil {
				slog.Warn("api session: cannot create temp dir for attachments", "error", err)
				tmpDir = os.TempDir()
			} else {
				// Schedule cleanup at the end of this goroutine's turn.
				defer os.RemoveAll(tmpDir) //nolint:errcheck
			}
			filePaths := core.SaveFilesToDisk(tmpDir, files)
			text = core.AppendFileRefs(text, filePaths)
		}
		userMsg = chatMessage{Role: "user", Content: text}
	}

	s.historyMu.Lock()
	s.history = append(s.history, userMsg)
	s.trimHistory()
	s.historyMu.Unlock()

	s.wg.Add(1)
	go func() {
		defer s.loopRunning.Store(false)
		s.agentLoop()
	}()
	return nil
}

// agentLoop runs the LLM → tool → LLM cycle until the model returns a text-only response.
func (s *apiSession) agentLoop() {
	defer s.wg.Done()

	tools := s.buildToolDefs()

	for iter := 0; iter < maxToolLoopIterations; iter++ {
		select {
		case <-s.ctx.Done():
			return
		default:
		}

		s.historyMu.Lock()
		msgs := make([]chatMessage, len(s.history))
		copy(msgs, s.history)
		s.historyMu.Unlock()

		req := completionRequest{
			Model:        s.model,
			Messages:     msgs,
			Tools:        tools,
			Stream:       true,
			SystemPrompt: s.system,
		}

		var (
			textBuf   strings.Builder
			allTCalls []toolCall
			hasErr    bool
		)

		for chunk := range s.llm.complete(s.ctx, req) {
			if chunk.err != nil {
				slog.Error("api session: llm error", "error", chunk.err)
				s.emit(core.Event{Type: core.EventError, Error: chunk.err})
				hasErr = true
				break
			}
			if chunk.text != "" {
				textBuf.WriteString(chunk.text)
				s.emit(core.Event{Type: core.EventText, Content: chunk.text})
			}
			if len(chunk.toolCalls) > 0 {
				allTCalls = append(allTCalls, chunk.toolCalls...)
			}
		}

		if hasErr {
			s.emitDone("")
			return
		}

		// Filter to function-type only before building the assistant history message.
		// Non-function tool types (e.g. "retrieval", "code_interpreter") are OpenAI
		// built-ins that don't go through MCP; we cannot provide results for them,
		// so including them in the assistant message would create orphaned tool_calls
		// that the API would reject.
		var functionTCalls []toolCall
		for _, tc := range allTCalls {
			if tc.Type == "function" || tc.Type == "" {
				functionTCalls = append(functionTCalls, tc)
			}
		}

		// No (function) tool calls → final response.
		if len(functionTCalls) == 0 {
			finalText := textBuf.String()
			s.historyMu.Lock()
			s.history = append(s.history, chatMessage{Role: "assistant", Content: finalText})
			s.trimHistory()
			s.historyMu.Unlock()
			s.emitDone(finalText)
			return
		}

		// Append assistant message with ONLY the function tool calls.
		assistantMsg := chatMessage{
			Role:      "assistant",
			Content:   textBuf.String(),
			ToolCalls: functionTCalls,
		}
		s.historyMu.Lock()
		s.history = append(s.history, assistantMsg)
		s.historyMu.Unlock()

		// Execute each tool call via MCP, append results to history.
		for _, tc := range functionTCalls {
			s.emit(core.Event{
				Type:      core.EventToolUse,
				ToolName:  tc.Function.Name,
				ToolInput: tc.Function.Arguments,
			})

			result := s.executeTool(tc)

			s.emit(core.Event{
				Type:       core.EventToolResult,
				ToolName:   tc.Function.Name,
				ToolResult: result,
			})

			s.historyMu.Lock()
			s.history = append(s.history, chatMessage{
				Role:       "tool",
				Content:    result,
				ToolCallID: tc.ID,
				Name:       tc.Function.Name,
			})
			s.historyMu.Unlock()
		}
	}

	// Exceeded iteration limit.
	s.emit(core.Event{Type: core.EventError, Error: fmt.Errorf("tool loop exceeded %d iterations", maxToolLoopIterations)})
	s.emitDone("")
}

func (s *apiSession) executeTool(tc toolCall) string {
	var args map[string]any
	if tc.Function.Arguments != "" {
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			return fmt.Sprintf("error: could not parse arguments: %v", err)
		}
	}

	if s.mcp == nil {
		return fmt.Sprintf("error: no MCP servers configured, cannot call tool %q", tc.Function.Name)
	}

	result, err := s.mcp.CallTool(s.ctx, tc.Function.Name, args)
	if err != nil {
		slog.Warn("api session: tool call failed", "tool", tc.Function.Name, "error", err)
		return fmt.Sprintf("error: %v", err)
	}

	if result.IsError {
		parts := extractTextParts(result.Content)
		return "error: " + strings.Join(parts, "\n")
	}

	return strings.Join(extractTextParts(result.Content), "\n")
}

func extractTextParts(content []mcp.ToolContent) []string {
	var out []string
	for _, c := range content {
		if c.Text != "" {
			out = append(out, c.Text)
		}
	}
	return out
}

func (s *apiSession) buildToolDefs() []toolDef {
	if s.mcp == nil {
		return nil
	}
	tools := s.mcp.AllTools()
	defs := make([]toolDef, 0, len(tools))
	for _, t := range tools {
		schema := t.InputSchema
		if schema == nil {
			schema = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		defs = append(defs, toolDef{
			Type: "function",
			Function: toolFuncDef{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  schema,
			},
		})
	}
	return defs
}

// trimHistory enforces the sliding window. Must be called with historyMu held.
// It always trims at user-message boundaries to avoid stranding tool/tool_calls
// sequences. trimHistory is only called when appending user messages (before any
// tool loop starts) and after final text-only responses, so there are no
// mid-loop orphaned tool messages in the kept portion.
func (s *apiSession) trimHistory() {
	if len(s.history) <= maxHistoryMessages {
		return
	}
	// Scan forward from the natural trim point until we find a user message
	// as the safe start of the kept window.
	cutoff := len(s.history) - maxHistoryMessages
	for cutoff < len(s.history) && s.history[cutoff].Role != "user" {
		cutoff++
	}
	if cutoff < len(s.history) {
		s.history = s.history[cutoff:]
	}
}

// emit sends an event to the events channel.
// Uses recover() to guard against the unlikely race where the channel is
// closed (by the watchdog timeout in Close()) while this goroutine is still
// active. In normal operation the ctx.Done() branch is taken instead.
func (s *apiSession) emit(e core.Event) {
	defer func() { recover() }() //nolint:errcheck
	select {
	case s.events <- e:
	case <-s.ctx.Done():
	}
}

func (s *apiSession) emitDone(text string) {
	s.emit(core.Event{
		Type:      core.EventResult,
		Content:   text,
		SessionID: s.sessionID,
		Done:      true,
	})
}

// doCloseEvents closes the events channel exactly once via sync.Once.
func (s *apiSession) doCloseEvents() {
	s.closeEventsOnce.Do(func() { close(s.events) })
}

// ---- core.AgentSession interface ----

func (s *apiSession) RespondPermission(_ string, _ core.PermissionResult) error {
	return nil
}

func (s *apiSession) Events() <-chan core.Event {
	return s.events
}

func (s *apiSession) CurrentSessionID() string {
	return s.sessionID
}

func (s *apiSession) Alive() bool {
	return s.alive.Load()
}

// Close cancels the session context and arranges for the events channel to be
// closed. It returns immediately; the channel is closed asynchronously once all
// active goroutines have exited or a 10-second watchdog fires.
//
// The events channel is protected by sync.Once and emit() uses recover(), so
// there is no panic even if a goroutine is still running when the watchdog fires.
func (s *apiSession) Close() error {
	s.alive.Store(false)
	s.cancel() // signals all in-flight LLM calls and MCP requests to abort

	go func() {
		done := make(chan struct{})
		go func() {
			s.wg.Wait()
			close(done)
		}()
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			slog.Warn("api session: close timed out waiting for agentLoop to exit — goroutine may leak")
		}
		s.doCloseEvents()
	}()
	return nil
}
