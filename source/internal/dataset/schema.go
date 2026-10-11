// Package dataset reads the JSONL output produced by the
// opencode-reasoning-extractor project and turns it into tokenized, loss-masked
// training examples for the Qwen3.8 trainer.
//
// The extractor's output is the canonical dataset. This package does not define
// an intermediate on-disk format; it streams the extractor records and renders
// them with the model's chat template on demand.
package dataset

import "encoding/json"

// ChatMessage mirrors the extractor's OpenAI-compatible chat message.
type ChatMessage struct {
	Role             string     `json:"role"`
	Content          string     `json:"content"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
	Name             string     `json:"name,omitempty"`
}

// ToolCall is an OpenAI-compatible tool invocation. Arguments is a compact JSON
// object encoded as a string.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction holds the resolved tool name and JSON-encoded arguments.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ToolSchema is a tool/function definition as exported by the extractor. It is
// only present once the extractor ships a static tool registry.
type ToolSchema struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// Tokens mirrors the token accounting stored on sessions.
type Tokens struct {
	Input      int64 `json:"input"`
	Output     int64 `json:"output"`
	Reasoning  int64 `json:"reasoning"`
	CacheRead  int64 `json:"cache_read"`
	CacheWrite int64 `json:"cache_write"`
}

// RecordMeta is the sidecar metadata attached to rich records.
type RecordMeta struct {
	Agent       string             `json:"agent,omitempty"`
	Model       string             `json:"model,omitempty"`
	Provider    string             `json:"provider,omitempty"`
	Subagent    bool               `json:"subagent"`
	ParentID    string             `json:"parent_id,omitempty"`
	Project     string             `json:"project,omitempty"`
	Directory   string             `json:"directory,omitempty"`
	TopicScores map[string]float64 `json:"topic_scores,omitempty"`
	Tokens      *Tokens            `json:"tokens,omitempty"`
	Cost        *float64           `json:"cost,omitempty"`
	TimeCreated int64              `json:"time_created,omitempty"`
	TimeUpdated int64              `json:"time_updated,omitempty"`
	Redactions  int                `json:"redactions,omitempty"`
	Tools       []ToolSchema       `json:"tools,omitempty"`
}

// SessionRecord is the per-session JSONL payload.
type SessionRecord struct {
	ID       string        `json:"id"`
	Topic    []string      `json:"topic"`
	Messages []ChatMessage `json:"messages"`
	Meta     RecordMeta    `json:"meta"`

	// Tree is "sft" or "subagents" and records where the session was read from.
	Tree string `json:"-"`
}

// TurnRecord is the per-assistant-turn JSONL payload.
type TurnRecord struct {
	ID       string        `json:"id"`
	Session  string        `json:"session"`
	Topic    []string      `json:"topic"`
	Messages []ChatMessage `json:"messages"`
	Meta     RecordMeta    `json:"meta"`
}

// Manifest describes an extraction run.
type Manifest struct {
	Version     string         `json:"version"`
	GeneratedAt string         `json:"generated_at"`
	Input       string         `json:"input"`
	CatalogHash string         `json:"catalog_hash"`
	Strict      bool           `json:"strict_openai"`
	Options     map[string]any `json:"options,omitempty"`
	Sessions    int            `json:"sessions_exported"`
	Exported    []string       `json:"exported_session_ids"`
	Topics      map[string]int `json:"topics"`
	Models      map[string]int `json:"models"`
	Agents      map[string]int `json:"agents"`
	Tools       []ToolSchema   `json:"tools,omitempty"`
}

// RLEntry is a per-session reward/quality record for a later RL stage.
type RLEntry struct {
	ID       string             `json:"id"`
	Topic    []string           `json:"topic"`
	Scores   map[string]float64 `json:"topic_scores,omitempty"`
	Agent    string             `json:"agent,omitempty"`
	Model    string             `json:"model,omitempty"`
	Provider string             `json:"provider,omitempty"`
	Subagent bool               `json:"subagent"`
	ParentID string             `json:"parent_id,omitempty"`
	Project  string             `json:"project,omitempty"`
	Reward   Reward             `json:"reward"`
}

// Reward holds derived, verifier-agnostic signals.
type Reward struct {
	ToolCalls       int            `json:"tool_calls"`
	ToolErrors      int            `json:"tool_errors"`
	ToolSuccessRate float64        `json:"tool_success_rate"`
	FinishReasons   map[string]int `json:"finish_reasons"`
	Patches         int            `json:"patches"`
	FilesTouched    int            `json:"files_touched"`
	Tokens          Tokens         `json:"tokens"`
	Cost            float64        `json:"cost"`
	HadError        bool           `json:"had_error"`
	FinalFinish     string         `json:"final_finish,omitempty"`
	QualityFlags    []string       `json:"quality_flags,omitempty"`
	Redactions      int            `json:"redactions"`
}
