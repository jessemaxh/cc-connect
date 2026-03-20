package core

import "github.com/chenhg5/cc-connect/mcp"

// SkillType constants
const (
	SkillTypePrompt    = "prompt"
	SkillTypeMCP       = "mcp"
	SkillTypeComposite = "composite"
)

// ManagedSkill represents a capability the AI agent can use.
// Named ManagedSkill to distinguish from the file-based Skill type in skill.go.
type ManagedSkill struct {
	// Identity
	Slug   string
	Name   string // display name (English, AI translates for user)
	Source string // "system" | "user"

	// Type & content
	SkillType   string // "prompt" | "mcp" | "composite"
	Instruction string // instructions for the AI (always English)

	// MCP integration (for mcp/composite types)
	MCPServers []mcp.ServerConfig

	// Parameters
	DefaultParams map[string]any // system defaults
	Params        map[string]any // merged (default + user personalization)

	// Control
	UserInvocable          bool // exposed as /skill <name>
	DisableModelInvocation bool // excluded from system prompt
	Enabled                bool

	// User skill fields
	ID          string   // DB id (user skills only)
	DependsOn   []string // system skill slugs this depends on
	CronExpr    string
	CronSession string
}

// SystemSkills defines all built-in skills.
// Instruction 用英文写，AI 会根据用户语言自动翻译回复。
var SystemSkills = []ManagedSkill{
	{
		Slug:      "weather",
		Name:      "Weather",
		Source:    "system",
		SkillType: SkillTypeMCP,
		Instruction: `When the user asks about weather, use the weather tool to query.
If no city is specified, use the default_city parameter.
Return: temperature, feels-like, weather conditions, air quality index, 3-day forecast.
Always respond in the user's language.`,
		MCPServers:    []mcp.ServerConfig{{Name: "weather", URL: "http://mcp-weather.easyclawbot-mcp.svc.cluster.local:8080"}},
		DefaultParams: map[string]any{"default_city": ""},
		UserInvocable: true,
		Enabled:       true,
	},
	{
		Slug:      "stocks",
		Name:      "Stocks",
		Source:    "system",
		SkillType: SkillTypeMCP,
		Instruction: `When the user asks about stock, fund, or cryptocurrency prices, use the stocks tool.
Support: A-shares (code like 600519), US stocks (AAPL), HK stocks (00700), crypto (BTC).
Return: current price, change percentage, volume.
If the user has a watchlist, query all items at once.
Always respond in the user's language.`,
		MCPServers:    []mcp.ServerConfig{{Name: "stocks", URL: "http://mcp-stocks.easyclawbot-mcp.svc.cluster.local:8080"}},
		DefaultParams: map[string]any{"watchlist": []any{}},
		UserInvocable: true,
		Enabled:       true,
	},
	{
		Slug:      "web-search",
		Name:      "Web Search",
		Source:    "system",
		SkillType: SkillTypeMCP,
		Instruction: `When the user needs up-to-date information, news, or facts you're unsure about, use the search tool.
Summarize the search results concisely and cite sources.
Always respond in the user's language.`,
		MCPServers:    []mcp.ServerConfig{{Name: "search", URL: "http://mcp-search.easyclawbot-mcp.svc.cluster.local:8080"}},
		DefaultParams: map[string]any{},
		UserInvocable: true,
		Enabled:       true,
	},
	{
		Slug:      "news",
		Name:      "News",
		Source:    "system",
		SkillType: SkillTypeMCP,
		Instruction: `When the user asks for news, use the news tool to get today's headlines.
Filter by user's preferred categories if set.
Return: title, brief summary, source. Limit to 5-10 items.
Always respond in the user's language.`,
		MCPServers:    []mcp.ServerConfig{{Name: "news", URL: "http://mcp-news.easyclawbot-mcp.svc.cluster.local:8080"}},
		DefaultParams: map[string]any{"categories": []any{"tech", "finance"}, "language": "en"},
		UserInvocable: true,
		Enabled:       true,
	},
	{
		Slug:      "traffic",
		Name:      "Traffic",
		Source:    "system",
		SkillType: SkillTypeMCP,
		Instruction: `When the user asks about traffic or commute, use the traffic tool.
If home and work addresses are set, query the commute route directly.
Return: estimated time, traffic conditions, alternative routes.
Always respond in the user's language.`,
		MCPServers:    []mcp.ServerConfig{{Name: "traffic", URL: "http://mcp-traffic.easyclawbot-mcp.svc.cluster.local:8080"}},
		DefaultParams: map[string]any{"home_address": "", "work_address": ""},
		UserInvocable: true,
		Enabled:       true,
	},
	{
		Slug:      "translator",
		Name:      "Translator",
		Source:    "system",
		SkillType: SkillTypePrompt,
		Instruction: `When the user asks for translation, detect the source language and translate to the target language.
If no target language is specified, use the user's preferred_language parameter.
Preserve the original formatting and tone.`,
		DefaultParams: map[string]any{"preferred_language": "en"},
		UserInvocable: true,
		Enabled:       true,
	},
	{
		Slug:      "calculator",
		Name:      "Calculator",
		Source:    "system",
		SkillType: SkillTypePrompt,
		Instruction: `When the user needs calculations, perform precise math.
Support: arithmetic expressions, unit conversion (temperature, length, weight, etc.), currency exchange.
For real-time exchange rates, note that you may need the web-search skill.
Show calculation steps.`,
		DefaultParams: map[string]any{},
		UserInvocable: true,
		Enabled:       true,
	},
	{
		Slug:      "reminder",
		Name:      "Reminder",
		Source:    "system",
		SkillType: SkillTypePrompt,
		Instruction: `When the user wants to set a reminder, parse the time and content.
Create a cron job using the skill_create tool with cron_expr set.
Support one-time and recurring reminders.
Confirm the time and content with the user before creating.
Always respond in the user's language.`,
		DefaultParams: map[string]any{},
		UserInvocable: true,
		Enabled:       true,
	},
}
