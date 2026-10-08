package dataset

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/tokenizer"
)

// Segment is a contiguous piece of rendered text with a uniform loss flag.
type Segment struct {
	Text string
	Loss bool
}

// RenderOptions controls chat-template rendering. Use DefaultRenderOptions and
// override fields as needed.
type RenderOptions struct {
	// EnableThinking mirrors the template's enable_thinking variable.
	EnableThinking bool
	// ReasoningEffort is one of "", "xhigh", "medium", "low". Empty defaults to
	// "xhigh"; "high" is an alias for "xhigh".
	ReasoningEffort string
	// PreserveThinking renders <think> blocks for earlier assistant turns too.
	PreserveThinking bool
	// AddGenerationPrompt appends the assistant generation prefix.
	AddGenerationPrompt bool
	// Tools is the tool/function registry used to render the tools preamble.
	Tools []ToolSchema
}

// DefaultRenderOptions returns the model template's defaults.
func DefaultRenderOptions() RenderOptions {
	return RenderOptions{
		EnableThinking:   true,
		ReasoningEffort:  "xhigh",
		PreserveThinking: true,
	}
}

const xhighInstruction = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer."
const lowInstruction = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."

const toolsPreamblePrefix = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
const toolsPreambleSuffix = "\n</tools>" +
	"\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n" +
	"<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n" +
	"<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n" +
	"</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n" +
	"- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n" +
	"- Required parameters MUST be specified\n" +
	"- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n" +
	"- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n" +
	"</IMPORTANT>"

// RenderSegments renders a chat transcript with the Qwen3.8 GGUF chat template.
// Every assistant message becomes a Loss=true segment; system, user, and tool
// messages are Loss=false context.
func RenderSegments(msgs []ChatMessage, opt RenderOptions) ([]Segment, error) {
	if len(msgs) == 0 {
		return nil, fmt.Errorf("dataset: no messages")
	}
	instructions, err := reasoningInstructions(opt)
	if err != nil {
		return nil, err
	}

	numSys, mergedSystem := mergeLeadingSystem(msgs)

	lastQuery, err := lastQueryIndex(msgs)
	if err != nil {
		return nil, err
	}

	var segs []Segment
	emit := func(loss bool, text string) {
		if text == "" {
			return
		}
		segs = append(segs, Segment{Text: text, Loss: loss})
	}

	// System preamble.
	if len(opt.Tools) > 0 {
		var b strings.Builder
		b.WriteString("<|im_start|>system\n")
		if instructions != "" {
			b.WriteString(instructions + "\n\n")
		}
		b.WriteString(toolsPreamblePrefix)
		for _, tool := range opt.Tools {
			b.WriteString("\n" + renderTool(tool))
		}
		b.WriteString(toolsPreambleSuffix)
		if mergedSystem != "" {
			b.WriteString("\n\n" + mergedSystem)
		}
		b.WriteString("<|im_end|>\n")
		emit(false, b.String())
	} else if mergedSystem != "" {
		var b strings.Builder
		b.WriteString("<|im_start|>system\n")
		if instructions != "" {
			b.WriteString(instructions + "\n\n")
		}
		b.WriteString(mergedSystem + "<|im_end|>\n")
		emit(false, b.String())
	} else if instructions != "" {
		emit(false, "<|im_start|>system\n"+instructions+"<|im_end|>\n")
	}

	for i := numSys; i < len(msgs); i++ {
		msg := &msgs[i]
		content := strings.TrimSpace(msg.Content)
		switch msg.Role {
		case "system", "developer":
			return nil, fmt.Errorf("dataset: system message must be at the beginning")
		case "user":
			emit(false, "<|im_start|>user\n"+content+"<|im_end|>\n")
		case "assistant":
			text, err := renderAssistant(msg, content, i, lastQuery, opt)
			if err != nil {
				return nil, err
			}
			emit(true, text)
		case "tool":
			if i > 0 && msgs[i-1].Role == "tool" {
				continue // part of a run already emitted
			}
			var b strings.Builder
			if i > 0 {
				b.WriteString("<|im_start|>user")
			}
			for j := i; j < len(msgs) && msgs[j].Role == "tool"; j++ {
				c := strings.TrimSpace(msgs[j].Content)
				b.WriteString("\n<tool_response>\n" + c + "\n</tool_response>")
			}
			b.WriteString("<|im_end|>\n")
			emit(false, b.String())
		default:
			return nil, fmt.Errorf("dataset: unexpected message role %q", msg.Role)
		}
	}

	if opt.AddGenerationPrompt {
		if !opt.EnableThinking {
			emit(false, "<|im_start|>assistant\n<think>\n\n</think>\n\n")
		} else {
			emit(false, "<|im_start|>assistant\n<think>\n")
		}
	}
	return segs, nil
}

func renderAssistant(msg *ChatMessage, content string, i, lastQuery int, opt RenderOptions) (string, error) {
	var b strings.Builder
	b.WriteString("<|im_start|>assistant\n")

	includeThink := opt.PreserveThinking || i > lastQuery
	if includeThink {
		reasoning := strings.TrimSpace(msg.ReasoningContent)
		b.WriteString("<think>\n" + reasoning + "\n</think>\n\n" + content)
	} else {
		b.WriteString(content)
	}

	for j, tc := range msg.ToolCalls {
		fn := tc.Function
		if fn.Name == "" {
			return "", fmt.Errorf("dataset: tool call missing a function name")
		}
		if j == 0 {
			if strings.TrimSpace(content) != "" {
				b.WriteString("\n\n<tool_call>\n<function=" + fn.Name + ">\n")
			} else {
				b.WriteString("<tool_call>\n<function=" + fn.Name + ">\n")
			}
		} else {
			b.WriteString("\n<tool_call>\n<function=" + fn.Name + ">\n")
		}
		params, err := orderedArgs(fn.Arguments)
		if err != nil {
			return "", fmt.Errorf("dataset: tool %s: %w", fn.Name, err)
		}
		for _, p := range params {
			b.WriteString("<parameter=" + p.key + ">\n" + p.val + "\n</parameter>\n")
		}
		b.WriteString("</function>\n</tool_call>")
	}
	b.WriteString("<|im_end|>\n")
	return b.String(), nil
}

// mergeLeadingSystem mirrors the template's merged system/developer messages.
func mergeLeadingSystem(msgs []ChatMessage) (int, string) {
	count := 0
	text := ""
	for i, m := range msgs {
		if i != count {
			break
		}
		if m.Role != "system" && m.Role != "developer" {
			break
		}
		c := strings.TrimSpace(m.Content)
		if c != "" {
			if text != "" {
				text += "\n"
			}
			text += c
		}
		count++
	}
	return count, text
}

// lastQueryIndex returns the index of the last genuine user query, matching the
// template's multi_step_tool scan.
func lastQueryIndex(msgs []ChatMessage) (int, error) {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role != "user" {
			continue
		}
		c := strings.TrimSpace(msgs[i].Content)
		if strings.HasPrefix(c, "<tool_response>") && strings.HasSuffix(c, "</tool_response>") {
			continue
		}
		return i, nil
	}
	return 0, fmt.Errorf("dataset: no user query found in messages")
}

func reasoningInstructions(opt RenderOptions) (string, error) {
	if !opt.EnableThinking {
		return "", nil
	}
	effort := opt.ReasoningEffort
	if effort == "" {
		effort = "xhigh"
	}
	if effort == "high" {
		effort = "xhigh"
	}
	switch effort {
	case "xhigh":
		return xhighInstruction, nil
	case "low":
		return lowInstruction, nil
	case "medium":
		return "", nil
	default:
		return "", fmt.Errorf("dataset: unexpected reasoning effort %q", opt.ReasoningEffort)
	}
}

// renderTool serializes a tool schema as an OpenAI function definition, the
// shape HF chat templates receive in their `tools` argument. Note the encoder
// emits compact JSON; Python's Jinja `tojson` adds insignificant whitespace.
func renderTool(t ToolSchema) string {
	type function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters,omitempty"`
	}
	wrapped := struct {
		Type     string   `json:"type"`
		Function function `json:"function"`
	}{
		Type: "function",
		Function: function{
			Name:        t.Name,
			Description: t.Description,
			Parameters:  t.Parameters,
		},
	}
	b, err := json.Marshal(wrapped)
	if err != nil {
		return "{}"
	}
	return string(b)
}

type argKV struct{ key, val string }

// orderedArgs parses a tool arguments JSON string, preserving key order. String
// values are emitted verbatim; other values are emitted as compact JSON.
func orderedArgs(raw string) ([]argKV, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" || raw == "{}" {
		return nil, nil
	}
	dec := json.NewDecoder(strings.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if d, ok := tok.(json.Delim); !ok || d != '{' {
		return nil, fmt.Errorf("arguments must be a JSON object")
	}
	var out []argKV
	for dec.More() {
		ktok, err := dec.Token()
		if err != nil {
			return nil, err
		}
		key, ok := ktok.(string)
		if !ok {
			return nil, fmt.Errorf("arguments key is not a string")
		}
		var val json.RawMessage
		if err := dec.Decode(&val); err != nil {
			return nil, err
		}
		var s string
		if len(val) > 0 && val[0] == '"' {
			if err := json.Unmarshal(val, &s); err != nil {
				return nil, err
			}
		} else {
			s = string(val)
		}
		out = append(out, argKV{key: key, val: s})
	}
	if _, err := dec.Token(); err != nil {
		return nil, err
	}
	return out, nil
}

// EncodeSegments tokenizes rendered segments and returns token ids plus a
// per-token loss mask (1 = train, 0 = context).
func EncodeSegments(v *tokenizer.Vocab, segs []Segment) (ids []int32, mask []uint8) {
	for _, s := range segs {
		if s.Text == "" {
			continue
		}
		t := v.Encode(s.Text)
		ids = append(ids, t...)
		var m uint8
		if s.Loss {
			m = 1
		}
		for range t {
			mask = append(mask, m)
		}
	}
	return ids, mask
}
