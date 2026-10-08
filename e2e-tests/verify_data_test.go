package e2e

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/dataset"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/tokenizer"
)

// TestReferenceData cross-checks the Go dataset code against independent
// reference implementations. Only runs when QWEN38_VERIFY_DATA=1:
//
//	TestReferenceData/template       Jinja2 render of the model's chat template
//	TestReferenceData/pretokenizer   Python `regex` evaluation of the qwen35 regex
func TestReferenceData(t *testing.T) {
	if os.Getenv("QWEN38_VERIFY_DATA") == "" {
		t.Skip("set QWEN38_VERIFY_DATA=1 to run the dataset reference checks")
	}
	t.Run("template", testReferenceTemplate)
	t.Run("pretokenizer", testReferencePretokenizer)
}

// requirePythonModule fails the test when an explicitly requested reference
// cannot run because a Python module is missing. Skips are only for the default
// (env-var unset) case, never for `make verify-data`.
func requirePythonModule(t *testing.T, mod string) {
	t.Helper()
	if err := exec.Command(refPython(), "-c", "import "+mod).Run(); err != nil {
		t.Fatalf("%s has no %q module; install it or point QWEN38_REF_PYTHON at a venv that has it", refPython(), mod)
	}
}

func requireModelPath(t *testing.T) string {
	t.Helper()
	candidates := []string{
		filepath.Join("..", "models", "Qwen3.8-27B-UD-Q4_K_M.gguf"),
		filepath.Join("..", "models", "abliterated.gguf"),
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	t.Fatalf("no model present (looked in %v); the template reference needs the GGUF", candidates)
	return ""
}

func testReferenceTemplate(t *testing.T) {
	requirePythonModule(t, "jinja2")
	model := requireModelPath(t)

	cases := templateCases()
	type refCase struct {
		Messages []map[string]any `json:"messages"`
		Options  map[string]any   `json:"options"`
	}
	refInput := make([]refCase, len(cases))
	for i, c := range cases {
		refInput[i] = refCase{Messages: toRefMessages(c.msgs), Options: c.pyOpts}
	}
	tmp, err := os.CreateTemp(t.TempDir(), "cases-*.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.NewEncoder(tmp).Encode(refInput); err != nil {
		t.Fatal(err)
	}
	tmp.Close()

	cmd := exec.Command(refPython(), scriptPath(t, "render_template.py"), model, tmp.Name())
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("render_template.py: %v\n%s", err, stderr.String())
	}
	var want []string
	if err := json.Unmarshal(stdout.Bytes(), &want); err != nil {
		t.Fatalf("parse reference output: %v", err)
	}
	if len(want) != len(cases) {
		t.Fatalf("reference returned %d cases, want %d", len(want), len(cases))
	}

	for i, c := range cases {
		segs, err := dataset.RenderSegments(c.msgs, c.goOpts)
		if err != nil {
			t.Fatalf("case %s: RenderSegments: %v", c.name, err)
		}
		var b strings.Builder
		for _, s := range segs {
			b.WriteString(s.Text)
		}
		if got := b.String(); got != want[i] {
			t.Errorf("case %s mismatch\n got: %q\nwant: %q", c.name, got, want[i])
		}
	}
}

func testReferencePretokenizer(t *testing.T) {
	requirePythonModule(t, "regex")
	corpus := []string{
		"Hello, world!",
		"don't we've I'm you'll he'd she's",
		"a  b",
		"a   ",
		" leading space",
		"line1\nline2\r\nline3",
		"123 456.789",
		"élan naïve Ångström café",
		"日本語のテキスト",
		"Devanagari: हिन्दी",
		"combining: e\u0301 a\u0308 x\u0327",
		"emoji 🚀🔥 ok",
		"tabs\tand\tspaces",
		"mixed<punct>{}[]|\\/",
		"\u00a0nbsp\u2003em",
		"trailing   ",
		"multiple\n\n\nnewlines",
		"a'b'c",
		"O'Brien's",
		"",
	}
	body, err := json.Marshal(corpus)
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(refPython(), scriptPath(t, "split_qwen35.py"))
	cmd.Stdin = bytes.NewReader(body)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("split_qwen35.py: %v\n%s", err, stderr.String())
	}
	var want [][]string
	if err := json.Unmarshal(stdout.Bytes(), &want); err != nil {
		t.Fatalf("parse reference output: %v", err)
	}
	if len(want) != len(corpus) {
		t.Fatalf("reference returned %d entries, want %d", len(want), len(corpus))
	}
	for i, text := range corpus {
		got := tokenizer.SplitQwen35(text)
		if len(got) == 0 && len(want[i]) == 0 {
			continue
		}
		if !equalStrings(got, want[i]) {
			t.Errorf("SplitQwen35(%q) = %q, reference = %q", text, got, want[i])
		}
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

type tplCase struct {
	name   string
	msgs   []dataset.ChatMessage
	goOpts dataset.RenderOptions
	pyOpts map[string]any
}

func templateCases() []tplCase {
	base := func() (dataset.RenderOptions, map[string]any) {
		return dataset.DefaultRenderOptions(), map[string]any{
			"enable_thinking":       true,
			"reasoning_effort":      "xhigh",
			"preserve_thinking":     true,
			"add_generation_prompt": false,
		}
	}
	mk := func(name string, msgs []dataset.ChatMessage, tweak func(*dataset.RenderOptions, map[string]any)) tplCase {
		g, p := base()
		if tweak != nil {
			tweak(&g, p)
		}
		return tplCase{name: name, msgs: msgs, goOpts: g, pyOpts: p}
	}

	return []tplCase{
		mk("simple", []dataset.ChatMessage{
			{Role: "user", Content: "hello"},
			{Role: "assistant", Content: "hi", ReasoningContent: "think."},
		}, nil),
		mk("merged-system", []dataset.ChatMessage{
			{Role: "system", Content: "sys one"},
			{Role: "system", Content: "sys two"},
			{Role: "user", Content: "u"},
			{Role: "assistant", Content: "a"},
		}, nil),
		mk("tool-call", []dataset.ChatMessage{
			{Role: "user", Content: "run ls"},
			{Role: "assistant", ReasoningContent: "r", ToolCalls: []dataset.ToolCall{{
				ID: "c1", Type: "function",
				Function: dataset.ToolCallFunction{Name: "bash", Arguments: `{"command":"ls","note":"x"}`},
			}}},
			{Role: "tool", ToolCallID: "c1", Name: "bash", Content: "out"},
			{Role: "assistant", Content: "done", ReasoningContent: "ok"},
		}, nil),
		mk("consecutive-tools", []dataset.ChatMessage{
			{Role: "user", Content: "u"},
			{Role: "assistant", Content: "c", ReasoningContent: "r", ToolCalls: []dataset.ToolCall{
				{Function: dataset.ToolCallFunction{Name: "a", Arguments: "{}"}},
				{Function: dataset.ToolCallFunction{Name: "b", Arguments: ""}},
			}},
			{Role: "tool", Content: "o1"},
			{Role: "tool", Content: "o2"},
			{Role: "assistant", Content: "done", ReasoningContent: "r2"},
		}, nil),
		mk("embedded-think", []dataset.ChatMessage{
			{Role: "user", Content: "u"},
			{Role: "assistant", Content: "<think>\nreasoning here\n</think>\nanswer"},
		}, nil),
		mk("no-thinking-gen", []dataset.ChatMessage{
			{Role: "user", Content: "u"},
		}, func(g *dataset.RenderOptions, p map[string]any) {
			g.EnableThinking = false
			g.AddGenerationPrompt = true
			p["enable_thinking"] = false
			p["add_generation_prompt"] = true
		}),
		mk("medium-gen", []dataset.ChatMessage{
			{Role: "user", Content: "u"},
		}, func(g *dataset.RenderOptions, p map[string]any) {
			g.ReasoningEffort = "medium"
			g.AddGenerationPrompt = true
			p["reasoning_effort"] = "medium"
			p["add_generation_prompt"] = true
		}),
		mk("no-preserve", []dataset.ChatMessage{
			{Role: "user", Content: "u1"},
			{Role: "assistant", Content: "a1", ReasoningContent: "r1"},
			{Role: "user", Content: "u2"},
			{Role: "assistant", Content: "a2", ReasoningContent: "r2"},
		}, func(g *dataset.RenderOptions, p map[string]any) {
			g.PreserveThinking = false
			p["preserve_thinking"] = false
		}),
	}
}

// toRefMessages converts extractor messages to the shape the real template
// expects: tool-call arguments must be an object, not a JSON string.
func toRefMessages(msgs []dataset.ChatMessage) []map[string]any {
	out := make([]map[string]any, 0, len(msgs))
	for _, m := range msgs {
		mm := map[string]any{"role": m.Role, "content": m.Content}
		if m.ReasoningContent != "" {
			mm["reasoning_content"] = m.ReasoningContent
		}
		if m.ToolCallID != "" {
			mm["tool_call_id"] = m.ToolCallID
		}
		if m.Name != "" {
			mm["name"] = m.Name
		}
		if len(m.ToolCalls) > 0 {
			tcs := make([]map[string]any, 0, len(m.ToolCalls))
			for _, tc := range m.ToolCalls {
				args := any(map[string]any{})
				if strings.TrimSpace(tc.Function.Arguments) != "" {
					_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
				}
				tcs = append(tcs, map[string]any{
					"id":   tc.ID,
					"type": tc.Type,
					"function": map[string]any{
						"name":      tc.Function.Name,
						"arguments": args,
					},
				})
			}
			mm["tool_calls"] = tcs
		}
		out = append(out, mm)
	}
	return out
}
