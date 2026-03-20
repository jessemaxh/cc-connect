package core

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/chenhg5/cc-connect/mcp"
)

// SkillEngine manages skill loading, personalization, and prompt injection.
type SkillEngine struct {
	mu               sync.RWMutex
	systemSkills     []ManagedSkill               // copy of SystemSkills with personalizations applied
	userSkills       []ManagedSkill               // from DB
	personalizations map[string]*SkillPersonalization // slug → override
	mcpManager       *mcp.Manager

	serverURL string // core-api base URL
	projectID string
	secret    string // webhook secret
}

// SkillPersonalization holds user overrides for a system skill.
type SkillPersonalization struct {
	Enabled           bool
	CustomParams      map[string]any
	CustomInstruction string
}

// NewSkillEngine creates a new SkillEngine.
func NewSkillEngine(mcpMgr *mcp.Manager, serverURL, projectID, secret string) *SkillEngine {
	return &SkillEngine{
		personalizations: make(map[string]*SkillPersonalization),
		mcpManager:       mcpMgr,
		serverURL:        strings.TrimRight(serverURL, "/"),
		projectID:        projectID,
		secret:           secret,
	}
}

// Load fetches user skills + personalizations from core-api and initializes MCP servers.
func (se *SkillEngine) Load(ctx context.Context) error {
	// 1. Build system skills (deep copy + apply defaults)
	system := make([]ManagedSkill, len(SystemSkills))
	for i, s := range SystemSkills {
		system[i] = s
		// Deep copy params
		system[i].Params = mergeSkillParams(s.DefaultParams, nil)
	}

	// 2. Fetch from server (if configured)
	if se.serverURL != "" && se.projectID != "" {
		resp, err := se.fetchProjectSkills(ctx)
		if err != nil {
			slog.Warn("skill engine: failed to load from server, using defaults", "error", err)
		} else {
			// Apply personalizations
			for _, p := range resp.Personalizations {
				se.personalizations[p.SkillSlug] = &SkillPersonalization{
					Enabled:           p.Enabled,
					CustomParams:      p.CustomParams,
					CustomInstruction: p.CustomInstruction,
				}
			}
			for i := range system {
				if p, ok := se.personalizations[system[i].Slug]; ok {
					system[i].Enabled = p.Enabled
					system[i].Params = mergeSkillParams(system[i].DefaultParams, p.CustomParams)
					if p.CustomInstruction != "" {
						system[i].Instruction += "\n\nAdditional user instructions:\n" + p.CustomInstruction
					}
				}
			}

			// Load user skills
			var userSkills []ManagedSkill
			for _, us := range resp.UserSkills {
				userSkills = append(userSkills, ManagedSkill{
					ID:            us.ID,
					Slug:          managedSkillSlugify(us.Name),
					Name:          us.Name,
					Source:        "user",
					SkillType:     SkillTypeComposite,
					Instruction:   us.Instruction,
					DependsOn:     us.DependsOn,
					Params:        us.Params,
					CronExpr:      us.CronExpr,
					CronSession:   us.CronSession,
					Enabled:       us.Enabled,
					UserInvocable: true,
				})
			}
			se.userSkills = userSkills
		}
	}

	se.systemSkills = system

	// 3. Start MCP servers for enabled MCP skills
	for _, s := range se.EnabledSkills() {
		if s.SkillType == SkillTypeMCP || s.SkillType == SkillTypeComposite {
			for _, srv := range s.MCPServers {
				if err := se.mcpManager.AddServer(ctx, srv); err != nil {
					slog.Warn("skill engine: failed to connect MCP server", "skill", s.Slug, "server", srv.Name, "error", err)
				}
			}
		}
	}

	return nil
}

// EnabledSkills returns all enabled skills (system + user).
func (se *SkillEngine) EnabledSkills() []ManagedSkill {
	se.mu.RLock()
	defer se.mu.RUnlock()

	var result []ManagedSkill
	for _, s := range se.systemSkills {
		if s.Enabled && !s.DisableModelInvocation {
			result = append(result, s)
		}
	}
	for _, s := range se.userSkills {
		if s.Enabled {
			result = append(result, s)
		}
	}
	return result
}

// AllSkills returns all skills including disabled ones.
func (se *SkillEngine) AllSkills() []ManagedSkill {
	se.mu.RLock()
	defer se.mu.RUnlock()
	var result []ManagedSkill
	result = append(result, se.systemSkills...)
	result = append(result, se.userSkills...)
	return result
}

// BuildPromptSection generates the XML skill section for the AI system prompt.
func (se *SkillEngine) BuildPromptSection() string {
	skills := se.EnabledSkills()
	if len(skills) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("\n<available-skills>\n")

	for _, s := range skills {
		fmt.Fprintf(&sb, "<skill name=%q source=%q>\n", s.Slug, s.Source)
		sb.WriteString(s.Instruction)
		if len(s.Params) > 0 {
			paramsJSON, _ := json.Marshal(s.Params)
			fmt.Fprintf(&sb, "\nUser preferences: %s", string(paramsJSON))
		}
		sb.WriteString("\n</skill>\n")
	}

	sb.WriteString("</available-skills>\n")

	// Skill management instructions
	sb.WriteString("\n<skill-management>\n")
	sb.WriteString("Users can ask you to create, modify, or delete custom skills.\n")
	sb.WriteString("Use skill_create/skill_update/skill_delete/skill_personalize tools for this.\n")
	sb.WriteString("When creating skills, compose from existing skills when possible.\n")
	sb.WriteString("For scheduled skills, set cron_expr (standard cron format).\n")
	sb.WriteString("Always confirm with the user before creating or modifying skills.\n")
	sb.WriteString("</skill-management>\n")

	return sb.String()
}

// ResolveManaged looks up a managed skill by slug or name (case-insensitive).
func (se *SkillEngine) ResolveManaged(nameOrSlug string) *ManagedSkill {
	lower := strings.ToLower(nameOrSlug)
	for _, s := range se.AllSkills() {
		if strings.ToLower(s.Slug) == lower || strings.ToLower(s.Name) == lower {
			return &s
		}
	}
	return nil
}

// fetchProjectSkills calls GET /api/internal/skills?project_id=X
func (se *SkillEngine) fetchProjectSkills(ctx context.Context) (*projectSkillsResponse, error) {
	url := se.serverURL + "/api/internal/skills?project_id=" + se.projectID

	reqCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-Webhook-Secret", se.secret)

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return nil, fmt.Errorf("server returned %d: %s", resp.StatusCode, string(body))
	}

	var result projectSkillsResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}
	return &result, nil
}

// projectSkillsResponse matches the server's response format.
type projectSkillsResponse struct {
	UserSkills       []userSkillRow       `json:"user_skills"`
	Personalizations []personalizationRow `json:"personalizations"`
}

type userSkillRow struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	Instruction string         `json:"instruction"`
	DependsOn   []string       `json:"depends_on"`
	Params      map[string]any `json:"params"`
	CronExpr    string         `json:"cron_expr,omitempty"`
	CronSession string         `json:"cron_session,omitempty"`
	Enabled     bool           `json:"enabled"`
}

type personalizationRow struct {
	SkillSlug         string         `json:"skill_slug"`
	Enabled           bool           `json:"enabled"`
	CustomParams      map[string]any `json:"custom_params"`
	CustomInstruction string         `json:"custom_instruction,omitempty"`
}

// mergeSkillParams merges custom params over defaults (shallow).
func mergeSkillParams(defaults, custom map[string]any) map[string]any {
	result := make(map[string]any)
	for k, v := range defaults {
		result[k] = v
	}
	for k, v := range custom {
		result[k] = v
	}
	return result
}

// managedSkillSlugify converts a name to a URL-safe slug.
func managedSkillSlugify(name string) string {
	s := strings.ToLower(strings.TrimSpace(name))
	s = strings.ReplaceAll(s, " ", "-")
	// Keep only alphanumeric and hyphens
	var sb strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '-' {
			sb.WriteRune(r)
		}
	}
	result := sb.String()
	if result == "" {
		return "skill"
	}
	return result
}
