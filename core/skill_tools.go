package core

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"
)

// SkillToolDefs returns tool definitions for the AI to manage skills.
// These are formatted as OpenAI-compatible function tools.
func (se *SkillEngine) SkillToolDefs() []SkillToolDef {
	return []SkillToolDef{
		{
			Name:        "skill_create",
			Description: "Create a new custom skill from a user's natural language description. Compose from existing skills when possible.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"name":        map[string]any{"type": "string", "description": "Skill display name"},
					"instruction": map[string]any{"type": "string", "description": "Detailed execution instruction for the AI"},
					"depends_on":  map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "System skill slugs this skill depends on"},
					"params":      map[string]any{"type": "object", "description": "Parameters from user's request"},
					"cron_expr":   map[string]any{"type": "string", "description": "Cron expression for scheduled execution (optional)"},
					"icon":        map[string]any{"type": "string", "description": "Emoji icon (optional)"},
				},
				"required": []string{"name", "instruction"},
			},
		},
		{
			Name:        "skill_update",
			Description: "Update an existing user-created skill's parameters, instruction, or schedule.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"skill_name":  map[string]any{"type": "string", "description": "Name or slug of the skill to update"},
					"name":        map[string]any{"type": "string", "description": "New name (optional)"},
					"instruction": map[string]any{"type": "string", "description": "New instruction (optional)"},
					"params":      map[string]any{"type": "object", "description": "Updated parameters (optional)"},
					"cron_expr":   map[string]any{"type": "string", "description": "New cron expression (optional)"},
					"enabled":     map[string]any{"type": "boolean", "description": "Enable or disable (optional)"},
				},
				"required": []string{"skill_name"},
			},
		},
		{
			Name:        "skill_delete",
			Description: "Delete a user-created skill.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"skill_name": map[string]any{"type": "string", "description": "Name or slug of the skill to delete"},
				},
				"required": []string{"skill_name"},
			},
		},
		{
			Name:        "skill_personalize",
			Description: "Personalize a system skill — set default parameters (like default city, watchlist) or disable it.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"skill_slug":         map[string]any{"type": "string", "description": "System skill slug (e.g. weather, stocks)"},
					"custom_params":      map[string]any{"type": "object", "description": "Parameter overrides (e.g. {default_city: 'Shanghai'})"},
					"custom_instruction": map[string]any{"type": "string", "description": "Additional instruction to append"},
					"enabled":            map[string]any{"type": "boolean", "description": "Enable or disable this system skill"},
				},
				"required": []string{"skill_slug"},
			},
		},
	}
}

// SkillToolDef describes a tool definition for skill management.
type SkillToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// HandleToolCall processes a skill management tool call from the AI.
// Returns the result text to send back to the AI.
func (se *SkillEngine) HandleToolCall(ctx context.Context, toolName string, args map[string]any) (string, error) {
	switch toolName {
	case "skill_create":
		return se.handleSkillCreate(ctx, args)
	case "skill_update":
		return se.handleSkillUpdate(ctx, args)
	case "skill_delete":
		return se.handleSkillDelete(ctx, args)
	case "skill_personalize":
		return se.handleSkillPersonalize(ctx, args)
	default:
		return "", fmt.Errorf("unknown skill tool: %s", toolName)
	}
}

// IsSkillTool returns true if the tool name is a skill management tool.
func IsSkillTool(name string) bool {
	switch name {
	case "skill_create", "skill_update", "skill_delete", "skill_personalize":
		return true
	}
	return false
}

func (se *SkillEngine) handleSkillCreate(ctx context.Context, args map[string]any) (string, error) {
	name, _ := args["name"].(string)
	instruction, _ := args["instruction"].(string)
	if name == "" || instruction == "" {
		return "Error: name and instruction are required.", nil
	}

	// Validate depends_on
	var dependsOn []string
	if deps, ok := args["depends_on"].([]any); ok {
		for _, d := range deps {
			if s, ok := d.(string); ok {
				found := false
				for _, sys := range se.systemSkills {
					if sys.Slug == s && sys.Enabled {
						found = true
						break
					}
				}
				if !found {
					return fmt.Sprintf("Error: depends_on references unknown or disabled skill %q.", s), nil
				}
				dependsOn = append(dependsOn, s)
			}
		}
	}

	params := make(map[string]any)
	if p, ok := args["params"].(map[string]any); ok {
		params = p
	}
	cronExpr, _ := args["cron_expr"].(string)

	body := map[string]any{
		"project_id":  se.projectID,
		"name":        name,
		"instruction": instruction,
		"depends_on":  dependsOn,
		"params":      params,
		"cron_expr":   cronExpr,
	}

	result, err := se.postInternal(ctx, "/api/internal/skills?project_id="+se.projectID, body)
	if err != nil {
		slog.Warn("skill_create: server error", "error", err)
		return "Error: failed to save skill. Please try again.", nil
	}

	// Add to local cache
	se.mu.Lock()
	se.userSkills = append(se.userSkills, ManagedSkill{
		ID:            getStringFromMap(result, "id"),
		Slug:          managedSkillSlugify(name),
		Name:          name,
		Source:        "user",
		SkillType:     SkillTypeComposite,
		Instruction:   instruction,
		DependsOn:     dependsOn,
		Params:        params,
		CronExpr:      cronExpr,
		Enabled:       true,
		UserInvocable: true,
	})
	se.mu.Unlock()

	msg := fmt.Sprintf("Skill %q created successfully.", name)
	if cronExpr != "" {
		msg += fmt.Sprintf(" Scheduled: %s", cronExpr)
	}
	return msg, nil
}

func (se *SkillEngine) handleSkillUpdate(ctx context.Context, args map[string]any) (string, error) {
	skillName, _ := args["skill_name"].(string)
	skill := se.ResolveManaged(skillName)
	if skill == nil {
		return fmt.Sprintf("Error: skill %q not found.", skillName), nil
	}
	if skill.Source != "user" {
		return "Error: only user-created skills can be updated. Use skill_personalize for system skills.", nil
	}

	body := map[string]any{"project_id": se.projectID}
	if v, ok := args["name"].(string); ok && v != "" {
		body["name"] = v
	}
	if v, ok := args["instruction"].(string); ok && v != "" {
		body["instruction"] = v
	}
	if v, ok := args["params"].(map[string]any); ok {
		body["params"] = v
	}
	if v, ok := args["cron_expr"].(string); ok {
		body["cron_expr"] = v
	}
	if v, ok := args["enabled"].(bool); ok {
		body["enabled"] = v
	}

	_, err := se.putInternal(ctx, "/api/internal/skills/"+skill.ID+"?project_id="+se.projectID, body)
	if err != nil {
		return "Error: failed to update skill.", nil
	}

	return fmt.Sprintf("Skill %q updated successfully.", skillName), nil
}

func (se *SkillEngine) handleSkillDelete(ctx context.Context, args map[string]any) (string, error) {
	skillName, _ := args["skill_name"].(string)
	skill := se.ResolveManaged(skillName)
	if skill == nil {
		return fmt.Sprintf("Error: skill %q not found.", skillName), nil
	}
	if skill.Source != "user" {
		return "Error: system skills cannot be deleted. Use skill_personalize to disable.", nil
	}

	err := se.deleteInternal(ctx, "/api/internal/skills/"+skill.ID+"?project_id="+se.projectID)
	if err != nil {
		return "Error: failed to delete skill.", nil
	}

	// Remove from local cache
	se.mu.Lock()
	for i, s := range se.userSkills {
		if s.ID == skill.ID {
			se.userSkills = append(se.userSkills[:i], se.userSkills[i+1:]...)
			break
		}
	}
	se.mu.Unlock()

	return fmt.Sprintf("Skill %q deleted.", skillName), nil
}

func (se *SkillEngine) handleSkillPersonalize(ctx context.Context, args map[string]any) (string, error) {
	slug, _ := args["skill_slug"].(string)

	// Verify it's a system skill
	var found bool
	for _, s := range se.systemSkills {
		if s.Slug == slug {
			found = true
			break
		}
	}
	if !found {
		return fmt.Sprintf("Error: system skill %q not found.", slug), nil
	}

	body := map[string]any{
		"project_id": se.projectID,
		"skill_slug": slug,
	}
	if v, ok := args["custom_params"].(map[string]any); ok {
		body["custom_params"] = v
	}
	if v, ok := args["custom_instruction"].(string); ok {
		body["custom_instruction"] = v
	}
	if v, ok := args["enabled"].(bool); ok {
		body["enabled"] = v
	}

	_, err := se.putInternal(ctx, "/api/internal/skills/personalize?project_id="+se.projectID, body)
	if err != nil {
		return "Error: failed to save personalization.", nil
	}

	// Update local cache
	se.mu.Lock()
	for i := range se.systemSkills {
		if se.systemSkills[i].Slug == slug {
			if cp, ok := args["custom_params"].(map[string]any); ok {
				se.systemSkills[i].Params = mergeSkillParams(se.systemSkills[i].DefaultParams, cp)
			}
			if ci, ok := args["custom_instruction"].(string); ok && ci != "" {
				// Re-derive from original
				for _, orig := range SystemSkills {
					if orig.Slug == slug {
						se.systemSkills[i].Instruction = orig.Instruction + "\n\nAdditional user instructions:\n" + ci
						break
					}
				}
			}
			if en, ok := args["enabled"].(bool); ok {
				se.systemSkills[i].Enabled = en
			}
			break
		}
	}
	se.mu.Unlock()

	return fmt.Sprintf("System skill %q personalized.", slug), nil
}

// HTTP helpers for internal API calls

func (se *SkillEngine) postInternal(ctx context.Context, path string, body any) (map[string]any, error) {
	return se.doInternal(ctx, "POST", path, body)
}

func (se *SkillEngine) putInternal(ctx context.Context, path string, body any) (map[string]any, error) {
	return se.doInternal(ctx, "PUT", path, body)
}

func (se *SkillEngine) deleteInternal(ctx context.Context, path string) error {
	reqCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, "DELETE", se.serverURL+path, nil)
	if err != nil {
		return err
	}
	req.Header.Set("X-Webhook-Secret", se.secret)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return fmt.Errorf("server returned %d", resp.StatusCode)
	}
	return nil
}

func (se *SkillEngine) doInternal(ctx context.Context, method, path string, body any) (map[string]any, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	reqCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, method, se.serverURL+path, bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Webhook-Secret", se.secret)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(respBody))
	}

	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result) // best-effort
	return result, nil
}

func getStringFromMap(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}
