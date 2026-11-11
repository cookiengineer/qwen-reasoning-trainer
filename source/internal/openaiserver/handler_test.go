package openaiserver

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/inference"
)

type fakeGenerator struct {
	req    inference.ChatRequest
	deltas []inference.Delta
	result *inference.Result
	err    error
}

func (f *fakeGenerator) Chat(_ context.Context, req inference.ChatRequest, emit func(inference.Delta) error) (*inference.Result, error) {
	f.req = req
	if f.err != nil {
		return nil, f.err
	}
	for _, d := range f.deltas {
		if emit != nil {
			if err := emit(d); err != nil {
				return nil, err
			}
		}
	}
	return f.result, nil
}

func testHandler(gen Generator, opts Options) http.Handler {
	if opts.ModelID == "" {
		opts.ModelID = "test-model"
	}
	return New(gen, opts)
}

func TestModels(t *testing.T) {
	h := testHandler(&fakeGenerator{}, Options{})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	var list ModelList
	if err := json.Unmarshal(rec.Body.Bytes(), &list); err != nil {
		t.Fatal(err)
	}
	if len(list.Data) != 1 || list.Data[0].ID != "test-model" {
		t.Fatalf("models = %+v", list)
	}
}

func TestChatCompletion(t *testing.T) {
	gen := &fakeGenerator{result: &inference.Result{
		Content:          "answer",
		ReasoningContent: "thought",
		FinishReason:     "stop",
		Usage:            inference.Usage{PromptTokens: 3, CompletionTokens: 2},
	}}
	h := testHandler(gen, Options{})
	body := `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	var resp CompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp.Object != "chat.completion" || len(resp.Choices) != 1 {
		t.Fatalf("response = %+v", resp)
	}
	msg := resp.Choices[0].Message
	if msg.Content == nil || *msg.Content != "answer" || msg.ReasoningContent != "thought" {
		t.Fatalf("message = %+v", msg)
	}
	if resp.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish = %q", resp.Choices[0].FinishReason)
	}
	if resp.Usage == nil || resp.Usage.TotalTokens != 5 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	if !gen.req.EnableThinking {
		t.Fatal("thinking should default to enabled")
	}
}

func TestChatCompletionToolCalls(t *testing.T) {
	gen := &fakeGenerator{result: &inference.Result{
		FinishReason: "tool_calls",
		ToolCalls: []inference.ToolCall{{
			ID: "call_1", Type: "function",
			Function: inference.FunctionCall{Name: "read", Arguments: `{"path":"/x"}`},
		}},
		Usage: inference.Usage{PromptTokens: 4, CompletionTokens: 5},
	}}
	h := testHandler(gen, Options{})
	body := `{"model":"test-model","messages":[{"role":"user","content":"read /x"}],"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	var resp CompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	msg := resp.Choices[0].Message
	if msg.Content != nil {
		t.Fatalf("content = %v, want null", *msg.Content)
	}
	if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].Function.Name != "read" {
		t.Fatalf("tool calls = %+v", msg.ToolCalls)
	}
	if len(gen.req.Tools) != 1 || gen.req.Tools[0].Name != "read" {
		t.Fatalf("request tools = %+v", gen.req.Tools)
	}
}

func TestChatCompletionStream(t *testing.T) {
	gen := &fakeGenerator{
		deltas: []inference.Delta{
			{ReasoningContent: "thought"},
			{Content: "answer"},
		},
		result: &inference.Result{
			Content: "answer", ReasoningContent: "thought", FinishReason: "stop",
			Usage: inference.Usage{PromptTokens: 2, CompletionTokens: 2},
		},
	}
	h := testHandler(gen, Options{})
	body := `{"model":"test-model","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}`
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content type = %q", ct)
	}
	events := sseEvents(rec.Body.String())
	if len(events) < 5 || events[len(events)-1] != "[DONE]" {
		t.Fatalf("events = %q", events)
	}
	var role, reasoning, content, final bool
	var usage *Usage
	for _, e := range events {
		if e == "[DONE]" {
			continue
		}
		var chunk Chunk
		if err := json.Unmarshal([]byte(e), &chunk); err != nil {
			t.Fatalf("chunk %q: %v", e, err)
		}
		switch {
		case chunk.Choices == nil:
		case len(chunk.Choices) == 0:
			usage = chunk.Usage
		default:
			c := chunk.Choices[0]
			if c.Delta.Role == "assistant" {
				role = true
			}
			if c.Delta.ReasoningContent == "thought" {
				reasoning = true
			}
			if c.Delta.Content == "answer" {
				content = true
			}
			if c.FinishReason != nil && *c.FinishReason == "stop" {
				final = true
			}
		}
	}
	if !role || !reasoning || !content || !final {
		t.Fatalf("role=%v reasoning=%v content=%v final=%v events=%q", role, reasoning, content, final, events)
	}
	if usage == nil || usage.TotalTokens != 4 {
		t.Fatalf("usage = %+v", usage)
	}
}

func TestAuth(t *testing.T) {
	h := testHandler(&fakeGenerator{}, Options{APIKey: "secret"})
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized status = %d", rec.Code)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer secret")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("authorized status = %d", rec.Code)
	}
}

func TestParseStop(t *testing.T) {
	cases := []struct {
		raw  string
		want []string
	}{
		{``, nil},
		{`null`, nil},
		{`"x"`, []string{"x"}},
		{`["a","","b"]`, []string{"a", "b"}},
	}
	for _, tc := range cases {
		got, err := parseStop(json.RawMessage(tc.raw))
		if err != nil {
			t.Fatalf("%s: %v", tc.raw, err)
		}
		if len(got) != len(tc.want) {
			t.Fatalf("%s: got %v want %v", tc.raw, got, tc.want)
		}
		for i := range got {
			if got[i] != tc.want[i] {
				t.Fatalf("%s: got %v want %v", tc.raw, got, tc.want)
			}
		}
	}
	if _, err := parseStop(json.RawMessage(`123`)); err == nil {
		t.Fatal("expected error for numeric stop")
	}
}

func sseEvents(body string) []string {
	var out []string
	for _, block := range strings.Split(body, "\n\n") {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		for _, line := range strings.Split(block, "\n") {
			if strings.HasPrefix(line, "data: ") {
				out = append(out, strings.TrimSpace(strings.TrimPrefix(line, "data: ")))
			}
		}
	}
	return out
}
