package inference

import (
	"encoding/json"
	"testing"
)

func TestMessagesToDataset(t *testing.T) {
	msgs := messagesToDataset([]Message{{
		Role:             "assistant",
		Content:          "answer",
		ReasoningContent: "thought",
		ToolCalls: []ToolCall{{
			ID:       "call_1",
			Function: FunctionCall{Name: "read", Arguments: `{"path":"/x"}`},
		}},
	}, {
		Role:       "tool",
		Content:    "contents",
		ToolCallID: "call_1",
		Name:       "read",
	}})
	if msgs[0].ReasoningContent != "thought" || msgs[0].ToolCalls[0].Type != "function" {
		t.Fatalf("assistant message = %+v", msgs[0])
	}
	if msgs[1].ToolCallID != "call_1" || msgs[1].Name != "read" {
		t.Fatalf("tool message = %+v", msgs[1])
	}
}

func TestToolsToDatasetAndArgTypes(t *testing.T) {
	in := []ToolSchema{{
		Name:        "read",
		Description: "Read a file",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"},"limit":{"type":"integer"}}}`),
	}}
	tools := toolsToDataset(in)
	if len(tools) != 1 || tools[0].Name != "read" {
		t.Fatalf("tools = %+v", tools)
	}
	types := toolArgTypes(in)
	if types["read"]["path"] != "string" || types["read"]["limit"] != "integer" {
		t.Fatalf("types = %+v", types)
	}
}
