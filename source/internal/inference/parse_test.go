package inference

import (
	"encoding/json"
	"strings"
	"testing"
)

func collectParser(t *testing.T, thinking bool, chunks ...string) ([]Delta, Result) {
	t.Helper()
	var deltas []Delta
	p := newStreamParser(thinking, nil, func(d Delta) error {
		deltas = append(deltas, d)
		return nil
	})
	for _, c := range chunks {
		if err := p.Feed(c); err != nil {
			t.Fatal(err)
		}
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	return deltas, p.Result()
}

func TestParserThinkingAndContent(t *testing.T) {
	deltas, res := collectParser(t, true,
		"Hello <thi",
		"nk>world</think>",
		"Answer",
	)
	if got, want := res.ReasoningContent, "Hello world"; got != want {
		t.Fatalf("reasoning = %q, want %q", got, want)
	}
	if got, want := res.Content, "Answer"; got != want {
		t.Fatalf("content = %q, want %q", got, want)
	}
	var reasoning, content strings.Builder
	for _, d := range deltas {
		reasoning.WriteString(d.ReasoningContent)
		content.WriteString(d.Content)
	}
	if reasoning.String() != res.ReasoningContent || content.String() != res.Content {
		t.Fatalf("streamed deltas do not match result: %+v", deltas)
	}
}

func TestParserHoldsPartialMarkers(t *testing.T) {
	var deltas []Delta
	p := newStreamParser(true, nil, func(d Delta) error {
		deltas = append(deltas, d)
		return nil
	})
	if err := p.Feed("<thi"); err != nil {
		t.Fatal(err)
	}
	if len(deltas) != 0 {
		t.Fatalf("emitted partial marker: %+v", deltas)
	}
	if err := p.Feed("nk>hi"); err != nil {
		t.Fatal(err)
	}
	if len(deltas) != 1 || deltas[0].ReasoningContent != "hi" {
		t.Fatalf("deltas = %+v, want reasoning hi", deltas)
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestParserToolCall(t *testing.T) {
	var deltas []Delta
	p := newStreamParser(false, map[string]map[string]string{
		"read": {"filePath": "string", "limit": "number"},
	}, func(d Delta) error {
		deltas = append(deltas, d)
		return nil
	})
	for _, c := range []string{"<tool_", "call>\n<function=read>\n<parameter=filePath>\n/x\n</parameter>\n<parameter=limit>\n42\n</parameter>\n</function>\n</tool_", "call>"} {
		if err := p.Feed(c); err != nil {
			t.Fatal(err)
		}
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	res := p.Result()
	if len(res.ToolCalls) != 1 {
		t.Fatalf("tool calls = %+v", res.ToolCalls)
	}
	tc := res.ToolCalls[0]
	if tc.Function.Name != "read" || tc.ID == "" || tc.Type != "function" {
		t.Fatalf("unexpected tool call: %+v", tc)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
		t.Fatalf("arguments %q: %v", tc.Function.Arguments, err)
	}
	if args["filePath"] != "/x" {
		t.Fatalf("filePath = %v", args["filePath"])
	}
	if args["limit"] != float64(42) {
		t.Fatalf("limit = %v (%T), want number 42", args["limit"], args["limit"])
	}
	if len(deltas) != 1 || len(deltas[0].ToolCalls) != 1 {
		t.Fatalf("tool deltas = %+v", deltas)
	}
}

func TestParserSchemaCoercion(t *testing.T) {
	types := map[string]map[string]string{"f": {
		"n": "number", "i": "integer", "b": "boolean", "o": "object", "s": "string",
	}}
	block := "<function=f>" +
		"<parameter=n>\n4.5\n</parameter>" +
		"<parameter=i>\n7\n</parameter>" +
		"<parameter=b>\ntrue\n</parameter>" +
		"<parameter=o>\n{\"a\":1}\n</parameter>" +
		"<parameter=s>\n hello \n</parameter>" +
		"</function>"
	calls := parseToolCalls(block, types)
	if len(calls) != 1 {
		t.Fatalf("calls = %+v", calls)
	}
	var args map[string]any
	if err := json.Unmarshal([]byte(calls[0].Function.Arguments), &args); err != nil {
		t.Fatal(err)
	}
	if args["n"] != 4.5 || args["i"] != float64(7) || args["b"] != true {
		t.Fatalf("numeric/bool coercion failed: %+v", args)
	}
	obj, ok := args["o"].(map[string]any)
	if !ok || obj["a"] != float64(1) {
		t.Fatalf("object coercion failed: %+v", args["o"])
	}
	if args["s"] != " hello " {
		t.Fatalf("string value = %q", args["s"])
	}
}

func TestParserMultipleToolCalls(t *testing.T) {
	block := "<function=one><parameter=a>\nx\n</parameter></function>" +
		"<function=two><parameter=b>\ny\n</parameter></function>"
	calls := parseToolCalls(block, nil)
	if len(calls) != 2 || calls[0].Function.Name != "one" || calls[1].Function.Name != "two" {
		t.Fatalf("calls = %+v", calls)
	}
}
