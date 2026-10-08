package tokenizer

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
)

// TestFromGGUFRealModel validates special-token handling against the real
// Qwen3.8 GGUF. It skips when no model is present so `make check` stays
// model-free.
func TestFromGGUFRealModel(t *testing.T) {
	candidates := []string{
		filepath.Join("..", "..", "models", "Qwen3.8-27B-UD-Q4_K_M.gguf"),
		filepath.Join("..", "..", "models", "abliterated.gguf"),
	}
	var path string
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			path = c
			break
		}
	}
	if path == "" {
		t.Skip("no model present")
	}

	g, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("open model: %v", err)
	}
	defer g.Close()

	v, err := FromGGUF(g)
	if err != nil {
		t.Fatalf("FromGGUF: %v", err)
	}
	if v.Pre != "qwen35" {
		t.Fatalf("Pre = %q, want qwen35", v.Pre)
	}

	checks := map[string]int32{
		"<|im_start|>":     248045,
		"<|im_end|>":       248046,
		"<tool_call>":      248058,
		"</tool_call>":     248059,
		"<tool_response>":  248066,
		"</tool_response>": 248067,
		"<think>":          248068,
		"</think>":         248069,
	}
	for s, id := range checks {
		got := v.Encode(s)
		if len(got) != 1 || got[0] != id {
			t.Errorf("Encode(%q) = %v, want [%d]", s, got, id)
		}
	}

	// A rendered assistant prefix must begin with the im_start control token and
	// contain the think control token.
	ids := v.Encode("<|im_start|>assistant\n<think>\n")
	if len(ids) < 2 || ids[0] != 248045 {
		t.Fatalf("assistant prefix encoding = %v", ids)
	}
	found := false
	for _, id := range ids {
		if id == 248068 {
			found = true
		}
	}
	if !found {
		t.Fatalf("assistant prefix missing <think>: %v", ids)
	}
}
