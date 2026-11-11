package inference

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/dataset"
)

// ToolSchema is an OpenAI-style function definition without the outer
// {"type":"function"} wrapper.
type ToolSchema struct {
	Name        string
	Description string
	Parameters  json.RawMessage
}

// FunctionCall is a resolved tool invocation.
type FunctionCall struct {
	Name      string
	Arguments string
}

// ToolCall is an OpenAI-compatible tool call.
type ToolCall struct {
	ID       string
	Type     string
	Function FunctionCall
}

// Message is one chat message in the inference request.
type Message struct {
	Role             string
	Content          string
	ReasoningContent string
	ToolCalls        []ToolCall
	ToolCallID       string
	Name             string
}

// ChatRequest is a fully resolved inference request.
type ChatRequest struct {
	Messages         []Message
	Tools            []ToolSchema
	EnableThinking   bool
	ReasoningEffort  string
	PreserveThinking bool
	MaxTokens        int
	Temperature      float64
	TopP             float64
	TopK             int
	Seed             int64
	Stop             []string
}

// Delta is one streaming update from the decoder. Exactly one of
// ReasoningContent, Content, or ToolCalls is normally set.
type Delta struct {
	ReasoningContent string
	Content          string
	ToolCalls        []ToolCall
}

// Usage reports token accounting for a completed request.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
}

// Result is the assembled response of a completed request.
type Result struct {
	Content          string
	ReasoningContent string
	ToolCalls        []ToolCall
	FinishReason     string
	Usage            Usage
	// Tokens is the exact generated token sequence (excluding the final EOS).
	Tokens []int32
}

func (e *Engine) render(req ChatRequest) ([]int32, error) {
	msgs := messagesToDataset(req.Messages)
	tools := toolsToDataset(req.Tools)
	opt := dataset.RenderOptions{
		EnableThinking:      req.EnableThinking,
		ReasoningEffort:     req.ReasoningEffort,
		PreserveThinking:    req.PreserveThinking,
		AddGenerationPrompt: true,
		Tools:               tools,
	}
	segs, err := dataset.RenderSegments(msgs, opt)
	if err != nil {
		return nil, err
	}
	ids, _ := dataset.EncodeSegments(e.Vocab, segs)
	if len(ids) == 0 {
		return nil, fmt.Errorf("inference: rendered an empty prompt")
	}
	return ids, nil
}

func messagesToDataset(msgs []Message) []dataset.ChatMessage {
	out := make([]dataset.ChatMessage, len(msgs))
	for i, m := range msgs {
		out[i] = dataset.ChatMessage{
			Role:             m.Role,
			Content:          m.Content,
			ReasoningContent: m.ReasoningContent,
			ToolCallID:       m.ToolCallID,
			Name:             m.Name,
		}
		if len(m.ToolCalls) > 0 {
			out[i].ToolCalls = make([]dataset.ToolCall, len(m.ToolCalls))
			for j, tc := range m.ToolCalls {
				out[i].ToolCalls[j] = dataset.ToolCall{
					ID:   tc.ID,
					Type: firstNonEmpty(tc.Type, "function"),
					Function: dataset.ToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: tc.Function.Arguments,
					},
				}
			}
		}
	}
	return out
}

func toolsToDataset(tools []ToolSchema) []dataset.ToolSchema {
	if len(tools) == 0 {
		return nil
	}
	out := make([]dataset.ToolSchema, 0, len(tools))
	for _, t := range tools {
		if strings.TrimSpace(t.Name) == "" {
			continue
		}
		params := t.Parameters
		if len(params) == 0 {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		out = append(out, dataset.ToolSchema{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  params,
		})
	}
	return out
}

// toolArgTypes extracts parameter name -> JSON schema type for each tool so the
// output parser can coerce values back to their declared types.
func toolArgTypes(tools []ToolSchema) map[string]map[string]string {
	out := map[string]map[string]string{}
	for _, t := range tools {
		if len(t.Parameters) == 0 {
			continue
		}
		var schema struct {
			Properties map[string]struct {
				Type string `json:"type"`
			} `json:"properties"`
		}
		if err := json.Unmarshal(t.Parameters, &schema); err != nil {
			continue
		}
		props := map[string]string{}
		for name, p := range schema.Properties {
			props[name] = p.Type
		}
		out[t.Name] = props
	}
	return out
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}
