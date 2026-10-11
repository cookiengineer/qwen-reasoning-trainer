package train

import (
	"strings"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

func TestOverridesCoverAdapters(t *testing.T) {
	cfg := tinyCfg()
	w := qwen38.NewRandom(cfg, 5)
	tm := NewQwenModel(cfg, w, DefaultLoRA(), 19)

	ov, err := tm.Overrides()
	if err != nil {
		t.Fatal(err)
	}
	if len(ov) != len(tm.entries) {
		t.Fatalf("overrides=%d entries=%d", len(ov), len(tm.entries))
	}
	for name, o := range ov {
		if !strings.HasPrefix(name, "blk.") || !strings.HasSuffix(name, ".weight") {
			t.Fatalf("bad tensor name %q", name)
		}
		if len(o.Raw) == 0 {
			t.Fatalf("empty raw for %s", name)
		}
	}
	// The full-attention layer (index 3) must expose attn_q/attn_output.
	for _, want := range []string{"blk.3.attn_q.weight", "blk.3.attn_output.weight", "blk.3.ffn_down.weight"} {
		if _, ok := ov[want]; !ok {
			t.Errorf("missing override %q", want)
		}
	}
	// A linear layer must expose the fused qkv and ssm_out.
	if _, ok := ov["blk.0.attn_qkv.weight"]; !ok {
		t.Errorf("missing linear-attention override blk.0.attn_qkv.weight")
	}
	if _, ok := ov["blk.0.ssm_out.weight"]; !ok {
		t.Errorf("missing linear-attention override blk.0.ssm_out.weight")
	}
}
