package dataset

import (
	"path/filepath"
	"testing"
)

func TestOpenAndSessions(t *testing.T) {
	d, err := Open(filepath.Join("testdata", "extractor"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if d.Manifest.Version != "0.1.0" {
		t.Fatalf("version = %q", d.Manifest.Version)
	}
	if len(d.Manifest.Tools) != 1 || d.Manifest.Tools[0].Name != "bash" {
		t.Fatalf("manifest tools = %+v", d.Manifest.Tools)
	}

	// ses_a is written to two topic files; it must be deduplicated.
	sessions, err := d.Sessions()
	if err != nil {
		t.Fatalf("Sessions: %v", err)
	}
	if len(sessions) != 2 {
		t.Fatalf("got %d sessions, want 2", len(sessions))
	}

	byID := map[string]SessionRecord{}
	for _, s := range sessions {
		byID[s.ID] = s
	}
	a, ok := byID["ses_a"]
	if !ok {
		t.Fatalf("missing ses_a")
	}
	if len(a.Messages) != 4 {
		t.Fatalf("ses_a messages = %d, want 4", len(a.Messages))
	}
	if a.Messages[1].ReasoningContent != "Let me read the parser first." {
		t.Fatalf("reasoning = %q", a.Messages[1].ReasoningContent)
	}
	if len(a.Messages[1].ToolCalls) != 1 {
		t.Fatalf("tool calls = %d", len(a.Messages[1].ToolCalls))
	}
	tc := a.Messages[1].ToolCalls[0]
	if tc.Function.Name != "read" || tc.Function.Arguments != `{"filePath":"/work/parser.go"}` {
		t.Fatalf("tool call = %+v", tc)
	}
	if a.Messages[2].Role != "tool" || a.Messages[2].ToolCallID != "call_1" {
		t.Fatalf("tool result = %+v", a.Messages[2])
	}
	if a.Meta.Tokens == nil || a.Meta.Tokens.Input != 100 {
		t.Fatalf("meta tokens = %+v", a.Meta.Tokens)
	}
	if a.Meta.Cost == nil || *a.Meta.Cost != 0.5 {
		t.Fatalf("meta cost = %+v", a.Meta.Cost)
	}
	if a.Tree != "sft" {
		t.Fatalf("tree = %q", a.Tree)
	}
}

func TestWalkRL(t *testing.T) {
	d, err := Open(filepath.Join("testdata", "extractor"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	var entries []RLEntry
	if err := d.WalkRL(func(e *RLEntry) error {
		entries = append(entries, *e)
		return nil
	}); err != nil {
		t.Fatalf("WalkRL: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("got %d rl entries, want 1", len(entries))
	}
	if entries[0].Reward.ToolSuccessRate != 1 || len(entries[0].Reward.QualityFlags) != 1 {
		t.Fatalf("reward = %+v", entries[0].Reward)
	}
}

func TestStrictDedupByHash(t *testing.T) {
	d, err := Open(filepath.Join("testdata", "strict"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	sessions, err := d.Sessions()
	if err != nil {
		t.Fatalf("Sessions: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("got %d strict sessions, want 1", len(sessions))
	}
	if sessions[0].ID != "" {
		t.Fatalf("strict session id = %q, want empty", sessions[0].ID)
	}
}

func TestOpenMissingManifest(t *testing.T) {
	if _, err := Open(t.TempDir()); err == nil {
		t.Fatal("expected error for missing manifest")
	}
}

func TestSubagentsOptIn(t *testing.T) {
	d, err := Open(filepath.Join("testdata", "extractor"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	d2, err := Open(filepath.Join("testdata", "extractor"), WithSubagents(true))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	// No subagents/ tree exists in the fixture, so both yield the same count.
	s1, _ := d.Sessions()
	s2, _ := d2.Sessions()
	if len(s1) != len(s2) {
		t.Fatalf("subagents opt-in changed count: %d vs %d", len(s1), len(s2))
	}
}
