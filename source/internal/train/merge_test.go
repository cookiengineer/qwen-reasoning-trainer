package train

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
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

// TestWriteMergedReadsBaseFromFile verifies the device-path merge contract: the
// frozen base bytes are read back from the source GGUF, so the host copy of the
// weights can be released (FreeRaw) before the merge runs.
func TestWriteMergedReadsBaseFromFile(t *testing.T) {
	cfg := tinyCfg()
	w := qwen38.NewRandom(cfg, 5)
	tm := NewQwenModel(cfg, w, DefaultLoRA(), 19)

	kvs := []gguf.KV{
		{Key: "general.architecture", Value: gguf.StringValue("qwen35")},
		{Key: "general.alignment", Value: gguf.Uint32Value(32)},
	}
	tensors := make([]gguf.WriteTensor, 0, len(tm.entries))
	for _, e := range tm.entries {
		dims := make([]uint64, len(e.w.Dims))
		for i, d := range e.w.Dims {
			dims[i] = uint64(d)
		}
		tensors = append(tensors, gguf.WriteTensor{Name: e.name, Dims: dims, Type: e.w.Typ, Data: e.w.Raw})
	}
	buf, err := gguf.Encode(kvs, tensors)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	basePath := filepath.Join(dir, "base.gguf")
	if err := os.WriteFile(basePath, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(basePath)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()

	// Drop the host base bytes: the merge must not depend on them.
	w.FreeRaw()

	out := filepath.Join(dir, "merged.gguf")
	if err := tm.WriteMerged(g, out); err != nil {
		t.Fatal(err)
	}
	og, err := gguf.Open(out)
	if err != nil {
		t.Fatal(err)
	}
	defer og.Close()
	if len(og.Tensors) != len(tensors) {
		t.Fatalf("merged tensors = %d, want %d", len(og.Tensors), len(tensors))
	}
	for i := range tensors {
		if og.Tensors[i].Type != tensors[i].Type {
			t.Fatalf("tensor %s type = %s, want %s", tensors[i].Name, og.Tensors[i].Type, tensors[i].Type)
		}
	}
}
