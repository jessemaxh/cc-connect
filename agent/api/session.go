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
type apiSession struct {
	llm    *llmClient
	mcp    *mcp.Manager
	skills *core.SkillEngine
	model  string
	system string

	history   []chatMessage
	historyMu sync.Mutex

	sessionID string
	events    chan core.Event
	// closeEventsOnce ensures events is closed exactly once across normal exit,
	// timeout path, and any future code paths.
	closeEventsOnce sync.Once

	ctx    context.Context
	cancel context.CancelFunc
	alive  atomic.Bool
	wg     sync.WaitGroup

	// loopRunning ensures only one agentLoop goroutine runs at a time.
	loopRunning atomic.Bool
}

func newAPISession(
	ctx context.Context,
	llm *llmClient,
	mcpMgr *mcp.Manager,
	skills *core.SkillEngine,
	model, system, sessionID string,
) *apiSession {
	// Append skill prompt section to system prompt.
	if skills != nil {
		section := skills.BuildPromptSection()
		if section != "" {
			system += section
		}
	}
	sCtx, cancel := context.WithCancel(ctx)
	s := &apiSession{
		llm:       llm,
		mcp:       mcpMgr,
		skills:    skills,
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

// Send implements core.AgentSession. Appends the user message to history and
// launches agentLoop in a goroutine. Returns an error if a loop is already running.
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
		// Capture tmpDir here so the goroutine can clean it up after the LLM call.
		// Do NOT defer RemoveAll in Send() — the defer fires when Send() returns,
		// before agentLoop starts, so the files would be deleted too early.
		var tmpDir string
		if len(files) > 0 {
			var err error
			tmpDir, err = os.MkdirTemp("", "cc-connect-api-*")
			if err != nil {
				slog.Warn("api session: cannot create temp dir for attachments", "error", err)
				tmpDir = ""
			}
			filePaths := core.SaveFilesToDisk(func() string {
				if tmpDir != "" {
					return tmpDir
				}
				return os.TempDir()
			}(), files)
			text = core.AppendFileRefs(text, filePaths)
		}
		userMsg = chatMessage{Role: "user", Content: text}

		s.historyMu.Lock()
		s.history = append(s.history, userMsg)
		s.trimHistory()
		s.historyMu.Unlock()

		s.wg.Add(1)
		go func() {
			defer s.loopRunning.Store(false)
			// Clean up temp files AFTER agentLoop returns (after the LLM has used the paths).
			if tmpDir != "" {
				defer os.RemoveAll(tmpDir) //nolint:errcheck
			}
			s.agentLoop()
		}()
		return nil
	}

	// Image path: no temp files needed.
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

// agentLoop runs the LLM → tool → LLM cycle until the model returns text-only.
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
				s.emit(core.Event{
					Type:      core.EventError,
					Error:     chunk.err,
					Done:      true,
					SessionID: s.sessionID,
				})
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
			return
		}

		// Filter to function-type only: non-function types (e.g. "retrieval",
		// "code_interpreter") are OpenAI built-ins we can't call via MCP. Including
		// them in the assistant message without a matching tool response causes 400.
		var functionTCalls []toolCall
		for _, tc := range allTCalls {
			if tc.Type == "function" || tc.Type == "" {
				functionTCalls = append(functionTCalls, tc)
			}
		}

		// No function tool calls → final text response.
		if len(functionTCalls) == 0 {
			finalText := textBuf.String()
			s.historyMu.Lock()
			s.history = append(s.history, chatMessage{Role: "assistant", Content: finalText})
			s.trimHistory()
			s.historyMu.Unlock()
			s.emit(core.Event{
				Type:      core.EventResult,
				Content:   finalText,
				SessionID: s.sessionID,
				Done:      true,
			})
			return
		}

		// Append assistant message with ONLY function tool calls.
		s.historyMu.Lock()
		s.history = append(s.history, chatMessage{
			Role:      "assistant",
			Content:   textBuf.String(),
			ToolCalls: functionTCalls,
		})
		s.historyMu.Unlock()

		// Execute each tool call, append result to history.
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

		// Trim after each complete tool-exchange iteration so the history doesn't
		// grow without bound during long tool loops (20 iter × N tools × large results).
		s.historyMu.Lock()
		s.trimHistory()
		s.historyMu.Unlock()
	}

	// Exceeded iteration limit.
	s.emit(core.Event{
		Type:      core.EventError,
		Error:     fmt.Errorf("tool loop exceeded %d iterations", maxToolLoopIterations),
		Done:      true,
		SessionID: s.sessionID,
	})
}

func (s *apiSession) executeTool(tc toolCall) string {
	var args map[string]any
	if tc.Function.Arguments != "" {
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			return fmt.Sprintf("error: could not parse arguments: %v", err)
		}
	}

	// Check if it's a skill management tool.
	if s.skills != nil && core.IsSkillTool(tc.Function.Name) {
		result, err := s.skills.HandleToolCall(s.ctx, tc.Function.Name, args)
		if err != nil {
			return fmt.Sprintf("error: %v", err)
		}
		return result
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
		return "error: " + strings.Join(extractTextParts(result.Content), "\n")
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
	var defs []toolDef

	// MCP tools.
	if s.mcp != nil {
		tools := s.mcp.AllTools()
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
	}

	// Skill management tools.
	if s.skills != nil {
		for _, st := range s.skills.SkillToolDefs() {
			defs = append(defs, toolDef{
				Type: "function",
				Function: toolFuncDef{
					Name:        st.Name,
					Description: st.Description,
					Parameters:  st.Parameters,
				},
			})
		}
	}

	return defs
}

// trimHistory trims the history to maxHistoryMessages, always starting at a
// user-message boundary to prevent orphaned tool sequences. Called only at
// safe boundaries: after user message append, after final assistant response,
// and after each tool-loop iteration.
func (s *apiSession) trimHistory() {
	if len(s.history) <= maxHistoryMessages {
		return
	}
	cutoff := len(s.history) - maxHistoryMessages
	for cutoff < len(s.history) && s.history[cutoff].Role != "user" {
		cutoff++
	}
	if cutoff < len(s.history) {
		s.history = s.history[cutoff:]
	}
}

// emit sends an event to the events channel.
// recover() guards against the rare race where the channel was closed by the
// watchdog timeout while this goroutine's ctx.Done() branch wasn't taken.
func (s *apiSession) emit(e core.Event) {
	defer func() { recover() }() //nolint:errcheck
	select {
	case s.events <- e:
	case <-s.ctx.Done():
	}
}

// ---- core.AgentSession interface ----

func (s *apiSession) RespondPermission(_ string, _ core.PermissionResult) error { return nil }

func (s *apiSession) Events() <-chan core.Event { return s.events }

func (s *apiSession) CurrentSessionID() string { return s.sessionID }

func (s *apiSession) Alive() bool { return s.alive.Load() }

// Close cancels the session context and arranges for the events channel to be
// closed asynchronously. The channel is protected by sync.Once and emit() uses
// recover(), so there is no panic even if the watchdog fires while a goroutine
// is still active.
func (s *apiSession) Close() error {
	s.alive.Store(false)
	s.cancel()

	go func() {
		done := make(chan struct{})
		go func() { s.wg.Wait(); close(done) }()
		select {
		case <-done:
		case <-time.After(10 * time.Second):
			slog.Warn("api session: close timed out — agentLoop goroutine may leak")
		}
		s.closeEventsOnce.Do(func() { close(s.events) })
	}()
	return nil
}
