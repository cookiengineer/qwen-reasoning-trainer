package qwen38

import (
	"bytes"
	"encoding/binary"
	"math"
	"path/filepath"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

func TestRewriteGGUF(t *testing.T) {
	kvs := []gguf.KV{
		{Key: "general.architecture", Value: gguf.StringValue("qwen35")},
		{Key: "general.alignment", Value: gguf.Uint32Value(32)},
	}
	origA := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	origB := []float32{9, 10}
	aData := make([]byte, 8*4)
	for i, v := range origA {
		binary.LittleEndian.PutUint32(aData[i*4:], math.Float32bits(v))
	}
	bData := make([]byte, 2*4)
	for i, v := range origB {
		binary.LittleEndian.PutUint32(bData[i*4:], math.Float32bits(v))
	}
	buf, err := gguf.Encode(kvs, []gguf.WriteTensor{
		{Name: "a", Dims: []uint64{4, 2}, Type: quant.TypeF32, Data: aData},
		{Name: "b", Dims: []uint64{2}, Type: quant.TypeF32, Data: bData},
	})
	if err != nil {
		t.Fatal(err)
	}
	base, err := gguf.ReadFrom(bytes.NewReader(buf), int64(len(buf)))
	if err != nil {
		t.Fatal(err)
	}

	newA := []float32{-1, -2, -3, -4}
	rawA, err := quant.Quantize(quant.TypeF32, newA)
	if err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "out.gguf")
	if err := RewriteGGUF(base, map[string]Override{"a": {Type: quant.TypeF32, Raw: rawA}}, out); err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(out)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	ta, _ := g.Tensor("a")
	gotA, err := g.ReadTensorF32(ta)
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range newA {
		if gotA[i] != w {
			t.Fatalf("a[%d] = %v, want %v", i, gotA[i], w)
		}
	}
	tb, _ := g.Tensor("b")
	gotB, err := g.ReadTensorF32(tb)
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range origB {
		if gotB[i] != w {
			t.Fatalf("b[%d] = %v, want %v", i, gotB[i], w)
		}
	}
}

// mapSource is a test OverrideSource.
type mapSource struct{ m map[string]Override }

func (s mapSource) OverrideMeta(name string, dims []uint64, baseType quant.Type) (quant.Type, int64, bool) {
	ov, ok := s.m[name]
	if !ok {
		return 0, 0, false
	}
	return ov.Type, int64(len(ov.Raw)), true
}
func (s mapSource) OverrideBytes(name string) ([]byte, error) { return s.m[name].Raw, nil }

func TestRewriteGGUFStream(t *testing.T) {
	kvs := []gguf.KV{
		{Key: "general.architecture", Value: gguf.StringValue("qwen35")},
		{Key: "general.alignment", Value: gguf.Uint32Value(32)},
	}
	aData := make([]byte, 8*4)
	for i, v := range []float32{1, 2, 3, 4, 5, 6, 7, 8} {
		binary.LittleEndian.PutUint32(aData[i*4:], math.Float32bits(v))
	}
	bData := make([]byte, 2*4)
	for i, v := range []float32{9, 10} {
		binary.LittleEndian.PutUint32(bData[i*4:], math.Float32bits(v))
	}
	buf, err := gguf.Encode(kvs, []gguf.WriteTensor{
		{Name: "a", Dims: []uint64{4, 2}, Type: quant.TypeF32, Data: aData},
		{Name: "b", Dims: []uint64{2}, Type: quant.TypeF32, Data: bData},
	})
	if err != nil {
		t.Fatal(err)
	}
	base, err := gguf.ReadFrom(bytes.NewReader(buf), int64(len(buf)))
	if err != nil {
		t.Fatal(err)
	}
	newA := []float32{-1, -2, -3, -4}
	rawA, err := quant.Quantize(quant.TypeF32, newA)
	if err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "out.gguf")
	src := mapSource{map[string]Override{"a": {Type: quant.TypeF32, Raw: rawA}}}
	if err := RewriteGGUFStream(base, src, out); err != nil {
		t.Fatal(err)
	}
	g, err := gguf.Open(out)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	ta, _ := g.Tensor("a")
	gotA, err := g.ReadTensorF32(ta)
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range newA {
		if gotA[i] != w {
			t.Fatalf("a[%d] = %v, want %v", i, gotA[i], w)
		}
	}
	tb, _ := g.Tensor("b")
	gotB, err := g.ReadTensorF32(tb)
	if err != nil {
		t.Fatal(err)
	}
	for i, w := range []float32{9, 10} {
		if gotB[i] != w {
			t.Fatalf("b[%d] = %v, want %v", i, gotB[i], w)
		}
	}
}

func TestTargetWeights(t *testing.T) {
	w := NewRandom(tinyConfig(), 7)
	targets := w.TargetWeights()
	overrides := w.Overrides()

	if len(targets) != len(overrides) {
		t.Fatalf("targets %d != overrides %d", len(targets), len(overrides))
	}
	seen := map[string]bool{}
	for _, tt := range targets {
		ov, ok := overrides[tt.Name]
		if !ok {
			t.Fatalf("target %q has no override", tt.Name)
		}
		if ov.Type != tt.W.Typ || len(ov.Raw) != len(tt.W.Raw) {
			t.Fatalf("target %q override does not match weight", tt.Name)
		}
		if seen[tt.Name] {
			t.Fatalf("duplicate target %q", tt.Name)
		}
		seen[tt.Name] = true
	}
	// 3 linear ssm_out + 1 full attn_output + 4 ffn_down.
	if len(targets) != 8 {
		t.Fatalf("target count = %d, want 8", len(targets))
	}
}
