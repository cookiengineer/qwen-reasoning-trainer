// Command verify-data cross-checks the Go dataset code against independent
// reference implementations:
//
//   - Jinja2 rendering of the model's chat template (render_template.py)
//   - Python `regex` evaluation of the qwen35 pre-tokenizer (split_qwen35.py)
//
// It is an opt-in external check, run as:
//
//	go run ./e2e-tests/verify-data          # or: make verify-data
//
// It needs the target GGUF (for tokenizer.chat_template) and the Python modules
// jinja2 + regex; the interpreter defaults to python3 (set QWEN38_REF_PYTHON to
// use the project venv). It exits non-zero on any mismatch.
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/dataset"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/tokenizer"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "verify-data:", err)
		os.Exit(1)
	}
	fmt.Println("verify-data: OK (template + pre-tokenizer match the references)")
}

func run() error {
	if err := checkPythonModule("jinja2"); err != nil {
		return err
	}
	if err := checkPythonModule("regex"); err != nil {
		return err
	}
	if err := checkTemplate(); err != nil {
		return err
	}
	if err := checkPretokenizer(); err != nil {
		return err
	}
	return nil
}

// checkPythonModule fails when a requested reference cannot run because its
// Python module is missing.
func checkPythonModule(mod string) error {
	if err := exec.Command(refPython(), "-c", "import "+mod).Run(); err != nil {
		return fmt.Errorf("%s has no %q module; install it or point QWEN38_REF_PYTHON at a venv that has it", refPython(), mod)
	}
	return nil
}

func checkTemplate() error {
	model, err := findModelPath()
	if err != nil {
		return err
	}

	cases := templateCases()
	type refCase struct {
		Messages []map[string]any `json:"messages"`
		Options  map[string]any   `json:"options"`
	}
	refInput := make([]refCase, len(cases))
	for i, c := range cases {
		refInput[i] = refCase{Messages: toRefMessages(c.msgs), Options: c.pyOpts}
	}
	tmp, err := os.CreateTemp("", "verify-data-cases-*.json")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if err := json.NewEncoder(tmp).Encode(refInput); err != nil {
		tmp.Close()
		return err
	}
	tmp.Close()

	cmd := exec.Command(refPython(), scriptPath("render_template.py"), model, tmp.Name())
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("render_template.py: %v\n%s", err, stderr.String())
	}
	var want []string
	if err := json.Unmarshal(stdout.Bytes(), &want); err != nil {
		return fmt.Errorf("parse reference output: %w", err)
	}
	if len(want) != len(cases) {
		return fmt.Errorf("reference returned %d cases, want %d", len(want), len(cases))
	}

	var mismatches []string
	for i, c := range cases {
		segs, err := dataset.RenderSegments(c.msgs, c.goOpts)
		if err != nil {
			return fmt.Errorf("case %s: RenderSegments: %w", c.name, err)
		}
		var b strings.Builder
		for _, s := range segs {
			b.WriteString(s.Text)
		}
		if got := b.String(); got != want[i] {
			mismatches = append(mismatches, fmt.Sprintf("case %s mismatch\n got: %q\nwant: %q", c.name, got, want[i]))
		}
	}
	if len(mismatches) > 0 {
		return fmt.Errorf("template:\n%s", strings.Join(mismatches, "\n"))
	}
	fmt.Printf("template: %d cases match the Jinja2 render\n", len(cases))
	return nil
}

func checkPretokenizer() error {
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
		return err
	}
	cmd := exec.Command(refPython(), scriptPath("split_qwen35.py"))
	cmd.Stdin = bytes.NewReader(body)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("split_qwen35.py: %v\n%s", err, stderr.String())
	}
	var want [][]string
	if err := json.Unmarshal(stdout.Bytes(), &want); err != nil {
		return fmt.Errorf("parse reference output: %w", err)
	}
	if len(want) != len(corpus) {
		return fmt.Errorf("reference returned %d entries, want %d", len(want), len(corpus))
	}
	var mismatches []string
	for i, text := range corpus {
		got := tokenizer.SplitQwen35(text)
		if len(got) == 0 && len(want[i]) == 0 {
			continue
		}
		if !equalStrings(got, want[i]) {
			mismatches = append(mismatches, fmt.Sprintf("SplitQwen35(%q) = %q, reference = %q", text, got, want[i]))
		}
	}
	if len(mismatches) > 0 {
		return fmt.Errorf("pre-tokenizer:\n%s", strings.Join(mismatches, "\n"))
	}
	fmt.Printf("pre-tokenizer: %d strings match the Python regex\n", len(corpus))
	return nil
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

// refPython returns the interpreter to use for the reference scripts.
func refPython() string {
	if p := os.Getenv("QWEN38_REF_PYTHON"); p != "" {
		return p
	}
	return "python3"
}

// scriptPath resolves a reference script next to this source file.
func scriptPath(name string) string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), name)
}

// findModelPath locates the target GGUF under the repository's models/ dir.
func findModelPath() (string, error) {
	root := repoRoot()
	candidates := []string{
		filepath.Join(root, "models", "Qwen3.8-27B-UD-Q4_K_M.gguf"),
		filepath.Join(root, "models", "abliterated.gguf"),
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}
	return "", fmt.Errorf("no model present (looked in %v); the template reference needs the GGUF", candidates)
}

// repoRoot returns the repository root (the source file is at e2e-tests/verify-data).
func repoRoot() string {
	_, thisFile, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(thisFile), "..", "..")
}
