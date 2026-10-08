package dataset

import (
	"path/filepath"
	"testing"
)

// TestReaderExtractorWriterOutput parses a fixture produced by the
// opencode-reasoning-extractor's own format.Output writer (not hand-written),
// so the reader is verified against the real emitter contract.
func TestReaderExtractorWriterOutput(t *testing.T) {
	d, err := Open(filepath.Join("testdata", "extractor-writer"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if d.Manifest.Version != "0.1.0" || d.Manifest.Sessions != 1 {
		t.Fatalf("manifest = %+v", d.Manifest)
	}

	sessions, err := d.Sessions()
	if err != nil {
		t.Fatalf("Sessions: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("got %d sessions, want 1", len(sessions))
	}
	rec := sessions[0]
	if rec.ID != "ses_writer" || rec.Meta.Agent != "build" || rec.Meta.Model != "deepseek-flash" {
		t.Fatalf("record = %+v", rec)
	}
	if rec.Meta.Tokens == nil || rec.Meta.Tokens.Reasoning != 3 {
		t.Fatalf("tokens = %+v", rec.Meta.Tokens)
	}
	if len(rec.Messages) != 5 {
		t.Fatalf("messages = %d", len(rec.Messages))
	}
	if rec.Messages[1].Role != "user" || rec.Messages[1].Content != "hello" {
		t.Fatalf("user = %+v", rec.Messages[1])
	}
	tc := rec.Messages[2].ToolCalls[0]
	if tc.Function.Name != "read" || tc.Function.Arguments != `{"filePath":"/x"}` {
		t.Fatalf("tool call = %+v", tc)
	}
	if rec.Messages[3].Role != "tool" || rec.Messages[3].Content != "contents" {
		t.Fatalf("tool result = %+v", rec.Messages[3])
	}

	var rewards []RLEntry
	if err := d.WalkRL(func(e *RLEntry) error {
		rewards = append(rewards, *e)
		return nil
	}); err != nil {
		t.Fatalf("WalkRL: %v", err)
	}
	if len(rewards) != 1 || rewards[0].Reward.ToolCalls != 1 || rewards[0].Reward.ToolSuccessRate != 1 {
		t.Fatalf("rewards = %+v", rewards)
	}
}
