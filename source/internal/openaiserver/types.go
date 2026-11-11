// Package openaiserver implements a minimal OpenAI-compatible chat completions
// API on top of the inference engine.
package openaiserver

import "encoding/json"

// ChatCompletionRequest is the accepted subset of the OpenAI request schema.
type ChatCompletionRequest struct {
	Model               string          `json:"model"`
	Messages            []ChatMessage   `json:"messages"`
	Stream              bool            `json:"stream"`
	StreamOptions       *StreamOptions  `json:"stream_options,omitempty"`
	MaxTokens           int             `json:"max_tokens,omitempty"`
	MaxCompletionTokens int             `json:"max_completion_tokens,omitempty"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	TopK                *int            `json:"top_k,omitempty"`
	Seed                *int64          `json:"seed,omitempty"`
	Stop                json.RawMessage `json:"stop,omitempty"`
	Tools               []Tool          `json:"tools,omitempty"`
	ToolChoice          json.RawMessage `json:"tool_choice,omitempty"`
	ParallelToolCalls   *bool           `json:"parallel_tool_calls,omitempty"`
	ReasoningEffort     string          `json:"reasoning_effort,omitempty"`
	ChatTemplateKwargs  map[string]any  `json:"chat_template_kwargs,omitempty"`
	EnableThinking      *bool           `json:"enable_thinking,omitempty"`
	N                   *int            `json:"n,omitempty"`
}

// StreamOptions controls streaming extras.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// ChatMessage is a request message. Content is raw so it accepts a string,
// null, or an array of text parts.
type ChatMessage struct {
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content,omitempty"`
	ReasoningContent string          `json:"reasoning_content,omitempty"`
	Name             string          `json:"name,omitempty"`
	ToolCallID       string          `json:"tool_call_id,omitempty"`
	ToolCalls        []ToolCall      `json:"tool_calls,omitempty"`
}

// Text returns the message's text content, joining text parts when content is
// an array. Non-text parts are ignored.
func (m ChatMessage) Text() string {
	if len(m.Content) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(m.Content, &parts); err == nil {
		var out string
		for _, p := range parts {
			if p.Type == "" || p.Type == "text" {
				out += p.Text
			}
		}
		return out
	}
	return ""
}

// Tool is an OpenAI function tool definition.
type Tool struct {
	Type     string      `json:"type"`
	Function FunctionDef `json:"function"`
}

// FunctionDef is the function schema inside a tool.
type FunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// ToolCall is a request or response tool invocation.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall holds the resolved tool name and JSON-encoded arguments.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// CompletionResponse is the non-streaming response body.
type CompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

// Choice is one completion choice.
type Choice struct {
	Index        int             `json:"index"`
	Message      ResponseMessage `json:"message"`
	FinishReason string          `json:"finish_reason"`
}

// ResponseMessage is the assistant message returned by a completion.
type ResponseMessage struct {
	Role             string     `json:"role"`
	Content          *string    `json:"content"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
}

// Chunk is one streamed chat completion chunk.
type Chunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []ChunkChoice `json:"choices"`
	Usage   *Usage        `json:"usage,omitempty"`
}

// ChunkChoice is one choice in a streamed chunk.
type ChunkChoice struct {
	Index        int        `json:"index"`
	Delta        ChunkDelta `json:"delta"`
	FinishReason *string    `json:"finish_reason"`
}

// ChunkDelta is the incremental content of a streamed chunk.
type ChunkDelta struct {
	Role             string          `json:"role,omitempty"`
	Content          string          `json:"content,omitempty"`
	ReasoningContent string          `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCallDelta `json:"tool_calls,omitempty"`
}

// ToolCallDelta adds the stream index to a tool call.
type ToolCallDelta struct {
	Index    int          `json:"index"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function FunctionCall `json:"function"`
}

// Usage reports token counts.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// Model is one entry in the /v1/models response.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelList is the /v1/models response body.
type ModelList struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// APIError is the OpenAI error envelope.
type APIError struct {
	Error ErrorBody `json:"error"`
}

// ErrorBody is the error details.
type ErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}
