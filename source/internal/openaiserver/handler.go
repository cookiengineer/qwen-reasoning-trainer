package openaiserver

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/inference"
)

// Generator is the part of *inference.Engine used by the HTTP layer.
type Generator interface {
	Chat(ctx context.Context, req inference.ChatRequest, emit func(inference.Delta) error) (*inference.Result, error)
}

// Options configures the OpenAI-compatible server.
type Options struct {
	ModelID            string
	Created            int64
	APIKey             string
	DefaultMaxTokens   int
	DefaultTemperature float64
	DefaultTopP        float64
	DefaultTopK        int
	Logger             *log.Logger
}

type server struct {
	gen  Generator
	opts Options
}

// New returns the HTTP handler for the server.
func New(gen Generator, opts Options) http.Handler {
	if opts.Created == 0 {
		opts.Created = time.Now().Unix()
	}
	if opts.DefaultMaxTokens <= 0 {
		opts.DefaultMaxTokens = 1024
	}
	if opts.DefaultTemperature < 0 {
		opts.DefaultTemperature = 0
	}
	if opts.DefaultTopP <= 0 || opts.DefaultTopP > 1 {
		opts.DefaultTopP = 1
	}
	if opts.DefaultTopK < 0 {
		opts.DefaultTopK = 0
	}
	s := &server{gen: gen, opts: opts}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /v1/models", s.auth(s.handleModels))
	mux.HandleFunc("GET /v1/models/{id}", s.auth(s.handleModel))
	mux.HandleFunc("POST /v1/chat/completions", s.auth(s.handleChat))
	return mux
}

func (s *server) auth(next http.HandlerFunc) http.HandlerFunc {
	if s.opts.APIKey == "" {
		return next
	}
	return func(w http.ResponseWriter, r *http.Request) {
		token := strings.TrimSpace(r.Header.Get("Authorization"))
		token = strings.TrimPrefix(token, "Bearer ")
		if subtle.ConstantTimeCompare([]byte(token), []byte(s.opts.APIKey)) != 1 {
			s.writeError(w, http.StatusUnauthorized, "authentication_error", "invalid_api_key",
				"invalid or missing API key")
			return
		}
		next(w, r)
	}
}

func (s *server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *server) handleModels(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, ModelList{
		Object: "list",
		Data: []Model{{
			ID:      s.opts.ModelID,
			Object:  "model",
			Created: s.opts.Created,
			OwnedBy: "local",
		}},
	})
}

func (s *server) handleModel(w http.ResponseWriter, r *http.Request) {
	if r.PathValue("id") != s.opts.ModelID {
		s.writeError(w, http.StatusNotFound, "not_found_error", "model_not_found",
			fmt.Sprintf("model %q not found", r.PathValue("id")))
		return
	}
	writeJSON(w, http.StatusOK, Model{
		ID:      s.opts.ModelID,
		Object:  "model",
		Created: s.opts.Created,
		OwnedBy: "local",
	})
}

func (s *server) handleChat(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 32<<20)
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_json",
			"could not decode request: "+err.Error())
		return
	}
	if req.Model != "" && req.Model != s.opts.ModelID {
		s.writeError(w, http.StatusNotFound, "not_found_error", "model_not_found",
			fmt.Sprintf("model %q not found", req.Model))
		return
	}
	if len(req.Messages) == 0 {
		s.writeError(w, http.StatusBadRequest, "invalid_request_error", "missing_messages",
			"messages is required")
		return
	}
	if req.N != nil && *req.N > 1 {
		s.writeError(w, http.StatusBadRequest, "invalid_request_error", "unsupported_parameter",
			"only n=1 is supported")
		return
	}

	in, err := s.buildRequest(&req)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_request", err.Error())
		return
	}
	id := newID("chatcmpl")
	created := time.Now().Unix()
	includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage

	if req.Stream {
		s.streamChat(w, r, id, created, in, includeUsage)
		return
	}

	res, err := s.gen.Chat(r.Context(), in, nil)
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, "server_error", "generation_failed", err.Error())
		return
	}
	content := res.Content
	var contentPtr *string = &content
	if res.Content == "" && len(res.ToolCalls) > 0 {
		contentPtr = nil
	}
	resp := CompletionResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: created,
		Model:   s.opts.ModelID,
		Choices: []Choice{{
			Index:        0,
			Message:      responseMessage(res, contentPtr),
			FinishReason: res.FinishReason,
		}},
		Usage: usagePtr(res.Usage),
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *server) buildRequest(req *ChatCompletionRequest) (inference.ChatRequest, error) {
	msgs := make([]inference.Message, 0, len(req.Messages))
	for _, m := range req.Messages {
		if strings.TrimSpace(m.Role) == "" {
			return inference.ChatRequest{}, fmt.Errorf("every message must have a role")
		}
		msg := inference.Message{
			Role:             m.Role,
			Content:          m.Text(),
			ReasoningContent: m.ReasoningContent,
			ToolCallID:       m.ToolCallID,
			Name:             m.Name,
		}
		for _, tc := range m.ToolCalls {
			msg.ToolCalls = append(msg.ToolCalls, inference.ToolCall{
				ID:   tc.ID,
				Type: firstNonEmpty(tc.Type, "function"),
				Function: inference.FunctionCall{
					Name:      tc.Function.Name,
					Arguments: tc.Function.Arguments,
				},
			})
		}
		msgs = append(msgs, msg)
	}

	tools := toInferenceTools(req.Tools)
	if toolChoiceNone(req.ToolChoice) {
		tools = nil
	}

	thinking := true
	if req.EnableThinking != nil {
		thinking = *req.EnableThinking
	} else if v, ok := req.ChatTemplateKwargs["enable_thinking"].(bool); ok {
		thinking = v
	}
	effort := req.ReasoningEffort
	if effort == "" {
		if v, ok := req.ChatTemplateKwargs["reasoning_effort"].(string); ok {
			effort = v
		}
	}
	switch effort {
	case "minimal":
		effort = "low"
	case "none":
		thinking = false
	}
	preserve := false
	if v, ok := req.ChatTemplateKwargs["preserve_thinking"].(bool); ok {
		preserve = v
	}

	stop, err := parseStop(req.Stop)
	if err != nil {
		return inference.ChatRequest{}, err
	}

	maxTokens := req.MaxCompletionTokens
	if maxTokens <= 0 {
		maxTokens = req.MaxTokens
	}
	if maxTokens <= 0 {
		maxTokens = s.opts.DefaultMaxTokens
	}
	temperature := s.opts.DefaultTemperature
	if req.Temperature != nil {
		temperature = *req.Temperature
	}
	topP := s.opts.DefaultTopP
	if req.TopP != nil {
		topP = *req.TopP
	}
	topK := s.opts.DefaultTopK
	if req.TopK != nil {
		topK = *req.TopK
	}
	var seed int64
	if req.Seed != nil {
		seed = *req.Seed
	}

	return inference.ChatRequest{
		Messages:         msgs,
		Tools:            tools,
		EnableThinking:   thinking,
		ReasoningEffort:  effort,
		PreserveThinking: preserve,
		MaxTokens:        maxTokens,
		Temperature:      temperature,
		TopP:             topP,
		TopK:             topK,
		Seed:             seed,
		Stop:             stop,
	}, nil
}

func (s *server) streamChat(w http.ResponseWriter, r *http.Request, id string, created int64, req inference.ChatRequest, includeUsage bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		s.writeError(w, http.StatusInternalServerError, "server_error", "stream_unsupported",
			"response writer does not support flushing")
		return
	}
	started := false
	toolIndex := 0

	send := func(delta ChunkDelta, finish *string, usage *Usage) error {
		if !started {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			w.Header().Set("X-Accel-Buffering", "no")
			w.WriteHeader(http.StatusOK)
			started = true
			role := Chunk{
				ID: id, Object: "chat.completion.chunk", Created: created, Model: s.opts.ModelID,
				Choices: []ChunkChoice{{Index: 0, Delta: ChunkDelta{Role: "assistant"}}},
			}
			if err := writeSSE(w, role); err != nil {
				return err
			}
			flusher.Flush()
		}
		chunk := Chunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: s.opts.ModelID,
			Choices: []ChunkChoice{{Index: 0, Delta: delta, FinishReason: finish}},
			Usage:   usage,
		}
		if err := writeSSE(w, chunk); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}

	emit := func(d inference.Delta) error {
		if d.ReasoningContent != "" {
			return send(ChunkDelta{ReasoningContent: d.ReasoningContent}, nil, nil)
		}
		if d.Content != "" {
			return send(ChunkDelta{Content: d.Content}, nil, nil)
		}
		for _, tc := range d.ToolCalls {
			delta := ToolCallDelta{
				Index:    toolIndex,
				ID:       tc.ID,
				Type:     firstNonEmpty(tc.Type, "function"),
				Function: FunctionCall{Name: tc.Function.Name, Arguments: tc.Function.Arguments},
			}
			toolIndex++
			if err := send(ChunkDelta{ToolCalls: []ToolCallDelta{delta}}, nil, nil); err != nil {
				return err
			}
		}
		return nil
	}

	res, err := s.gen.Chat(r.Context(), req, emit)
	if err != nil {
		if !started {
			s.writeError(w, http.StatusInternalServerError, "server_error", "generation_failed", err.Error())
			return
		}
		_ = writeSSE(w, APIError{Error: ErrorBody{
			Message: err.Error(), Type: "server_error", Code: "generation_failed",
		}})
		flusher.Flush()
		writeDone(w)
		flusher.Flush()
		return
	}

	finish := res.FinishReason
	if err := send(ChunkDelta{}, &finish, nil); err != nil {
		return
	}
	if includeUsage {
		u := usageValue(res.Usage)
		chunk := Chunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: s.opts.ModelID,
			Choices: []ChunkChoice{}, Usage: &u,
		}
		if err := writeSSE(w, chunk); err != nil {
			return
		}
		flusher.Flush()
	}
	writeDone(w)
	flusher.Flush()
}

func (s *server) writeError(w http.ResponseWriter, status int, typ, code, msg string) {
	writeJSON(w, status, APIError{Error: ErrorBody{Message: msg, Type: typ, Code: code}})
}

func responseMessage(res *inference.Result, content *string) ResponseMessage {
	msg := ResponseMessage{
		Role:             "assistant",
		Content:          content,
		ReasoningContent: res.ReasoningContent,
	}
	for _, tc := range res.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, ToolCall{
			ID:   tc.ID,
			Type: firstNonEmpty(tc.Type, "function"),
			Function: FunctionCall{
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			},
		})
	}
	return msg
}

func toInferenceTools(tools []Tool) []inference.ToolSchema {
	if len(tools) == 0 {
		return nil
	}
	out := make([]inference.ToolSchema, 0, len(tools))
	for _, t := range tools {
		if strings.TrimSpace(t.Function.Name) == "" {
			continue
		}
		out = append(out, inference.ToolSchema{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			Parameters:  t.Function.Parameters,
		})
	}
	return out
}

func toolChoiceNone(raw json.RawMessage) bool {
	if len(raw) == 0 {
		return false
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s == "none"
	}
	return false
}

func parseStop(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		if s == "" {
			return nil, nil
		}
		return []string{s}, nil
	}
	var list []string
	if err := json.Unmarshal(raw, &list); err != nil {
		return nil, fmt.Errorf("stop must be a string or an array of strings")
	}
	out := list[:0]
	for _, v := range list {
		if v != "" {
			out = append(out, v)
		}
	}
	return out, nil
}

func usageValue(u inference.Usage) Usage {
	return Usage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.PromptTokens + u.CompletionTokens,
	}
}

func usagePtr(u inference.Usage) *Usage {
	v := usageValue(u)
	return &v
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeSSE(w http.ResponseWriter, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", data)
	return err
}

func writeDone(w http.ResponseWriter) {
	_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
}

func newID(prefix string) string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
	}
	return prefix + "-" + hex.EncodeToString(b[:])
}
