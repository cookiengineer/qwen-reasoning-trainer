package dataset

import (
	"reflect"
	"strings"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/tokenizer"
)

func TestRenderSimple(t *testing.T) {
	msgs := []ChatMessage{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi", ReasoningContent: "think."},
	}
	segs, err := RenderSegments(msgs, DefaultRenderOptions())
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) != 3 {
		t.Fatalf("got %d segments: %+v", len(segs), segs)
	}
	if segs[0].Loss || !strings.HasPrefix(segs[0].Text, "<|im_start|>system\n"+xhighInstruction) {
		t.Fatalf("system seg = %+v", segs[0])
	}
	if segs[1].Loss || segs[1].Text != "<|im_start|>user\nhello<|im_end|>\n" {
		t.Fatalf("user seg = %+v", segs[1])
	}
	want := "<|im_start|>assistant\n<think>\nthink.\n</think>\n\nhi<|im_end|>\n"
	if !segs[2].Loss || segs[2].Text != want {
		t.Fatalf("assistant seg = %q, want %q", segs[2].Text, want)
	}
}

func TestRenderMergedSystem(t *testing.T) {
	msgs := []ChatMessage{
		{Role: "system", Content: "sys one"},
		{Role: "system", Content: "sys two"},
		{Role: "user", Content: "u"},
		{Role: "assistant", Content: "a"},
	}
	segs, err := RenderSegments(msgs, DefaultRenderOptions())
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>system\n" + xhighInstruction + "\n\nsys one\nsys two<|im_end|>\n"
	if segs[0].Text != want {
		t.Fatalf("preamble = %q, want %q", segs[0].Text, want)
	}
}

func TestRenderToolCall(t *testing.T) {
	msgs := []ChatMessage{
		{Role: "user", Content: "run ls"},
		{Role: "assistant", ReasoningContent: "r", ToolCalls: []ToolCall{{
			ID: "c1", Type: "function",
			Function: ToolCallFunction{Name: "bash", Arguments: `{"command":"ls","note":"x"}`},
		}}},
		{Role: "tool", ToolCallID: "c1", Name: "bash", Content: "out"},
	}
	segs, err := RenderSegments(msgs, DefaultRenderOptions())
	if err != nil {
		t.Fatal(err)
	}
	// system, user, assistant, tool
	if len(segs) != 4 {
		t.Fatalf("got %d segments: %+v", len(segs), segs)
	}
	wantAssistant := "<|im_start|>assistant\n<think>\nr\n</think>\n\n" +
		"<tool_call>\n<function=bash>\n" +
		"<parameter=command>\nls\n</parameter>\n" +
		"<parameter=note>\nx\n</parameter>\n" +
		"</function>\n</tool_call>" +
		"<|im_end|>\n"
	if !segs[2].Loss || segs[2].Text != wantAssistant {
		t.Fatalf("assistant = %q", segs[2].Text)
	}
	wantTool := "<|im_start|>user\n<tool_response>\nout\n</tool_response><|im_end|>\n"
	if segs[3].Loss || segs[3].Text != wantTool {
		t.Fatalf("tool = %q, want %q", segs[3].Text, wantTool)
	}
}

func TestRenderConsecutiveTools(t *testing.T) {
	msgs := []ChatMessage{
		{Role: "user", Content: "u"},
		{Role: "assistant", Content: "c", ReasoningContent: "r", ToolCalls: []ToolCall{
			{Function: ToolCallFunction{Name: "a", Arguments: "{}"}},
			{Function: ToolCallFunction{Name: "b", Arguments: ""}},
		}},
		{Role: "tool", Content: "o1"},
		{Role: "tool", Content: "o2"},
		{Role: "assistant", Content: "done", ReasoningContent: "r2"},
	}
	segs, err := RenderSegments(msgs, DefaultRenderOptions())
	if err != nil {
		t.Fatal(err)
	}
	// system, user, assistant, tool-group, assistant
	if len(segs) != 5 {
		t.Fatalf("got %d segments: %+v", len(segs), segs)
	}
	if !strings.Contains(segs[2].Text, "\n\n<tool_call>\n<function=a>") {
		t.Fatalf("first tool_call spacing wrong: %q", segs[2].Text)
	}
	if !strings.Contains(segs[2].Text, "\n<tool_call>\n<function=b>") {
		t.Fatalf("second tool_call spacing wrong: %q", segs[2].Text)
	}
	if !strings.Contains(segs[2].Text, "<function=a>\n</function>") {
		t.Fatalf("empty args should emit no parameters: %q", segs[2].Text)
	}
	wantTools := "<|im_start|>user\n<tool_response>\no1\n</tool_response>\n<tool_response>\no2\n</tool_response><|im_end|>\n"
	if segs[3].Text != wantTools {
		t.Fatalf("tool group = %q, want %q", segs[3].Text, wantTools)
	}
	if !segs[4].Loss {
		t.Fatalf("final assistant should be trained: %+v", segs[4])
	}
}

func TestRenderEmbeddedThink(t *testing.T) {
	// The GGUF template does not split an embedded <think> block out of content;
	// it emits an empty think block and the content verbatim. (The separate
	// references/.../Qwen3.5-4B.jinja does split, but that is not this model's
	// template.)
	msgs := []ChatMessage{
		{Role: "user", Content: "u"},
		{Role: "assistant", Content: "<think>\nreasoning here\n</think>\nanswer"},
	}
	segs, err := RenderSegments(msgs, DefaultRenderOptions())
	if err != nil {
		t.Fatal(err)
	}
	want := "<|im_start|>assistant\n<think>\n\n</think>\n\n<think>\nreasoning here\n</think>\nanswer<|im_end|>\n"
	if segs[len(segs)-1].Text != want {
		t.Fatalf("embedded think = %q, want %q", segs[len(segs)-1].Text, want)
	}
}

func TestRenderNoUserQuery(t *testing.T) {
	msgs := []ChatMessage{{Role: "assistant", Content: "a"}}
	if _, err := RenderSegments(msgs, DefaultRenderOptions()); err == nil {
		t.Fatal("expected error for missing user query")
	}
}

func TestReasoningEffort(t *testing.T) {
	msgs := []ChatMessage{{Role: "user", Content: "u"}, {Role: "assistant", Content: "a"}}
	opt := DefaultRenderOptions()
	opt.ReasoningEffort = "medium"
	segs, err := RenderSegments(msgs, opt)
	if err != nil {
		t.Fatal(err)
	}
	// medium emits no instruction, so no system preamble.
	if segs[0].Text != "<|im_start|>user\nu<|im_end|>\n" {
		t.Fatalf("medium preamble = %q", segs[0].Text)
	}

	opt.ReasoningEffort = "bogus"
	if _, err := RenderSegments(msgs, opt); err == nil {
		t.Fatal("expected error for bogus effort")
	}
}

func TestOrderedArgs(t *testing.T) {
	got, err := orderedArgs(`{"b":"1","a":2,"c":[1,2],"d":null,"e":true}`)
	if err != nil {
		t.Fatal(err)
	}
	want := []argKV{{"b", "1"}, {"a", "2"}, {"c", "[1,2]"}, {"d", "null"}, {"e", "true"}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("orderedArgs = %+v, want %+v", got, want)
	}
	if _, err := orderedArgs(`[1,2]`); err == nil {
		t.Fatal("expected error for non-object arguments")
	}
}

func TestEncodeSegmentsMask(t *testing.T) {
	v := tokenizer.NewVocab([]string{"a", "b", "c"})
	segs := []Segment{{Text: "ab", Loss: true}, {Text: "c", Loss: false}}
	ids, mask := EncodeSegments(v, segs)
	if !reflect.DeepEqual(ids, []int32{0, 1, 2}) {
		t.Fatalf("ids = %v", ids)
	}
	if !reflect.DeepEqual(mask, []uint8{1, 1, 0}) {
		t.Fatalf("mask = %v", mask)
	}
}
