package dataset

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/tokenizer"
)

// findModelPath returns a local GGUF path or "" if none is present.
func findModelPath() string {
	candidates := []string{
		filepath.Join("..", "..", "models", "Qwen3.8-27B-UD-Q4_K_M.gguf"),
		filepath.Join("..", "..", "models", "abliterated.gguf"),
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

func testVocab(t *testing.T) *tokenizer.Vocab {
	t.Helper()
	path := findModelPath()
	if path == "" {
		t.Skip("no model present")
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatalf("open model: %v", err)
	}
	defer g.Close()
	v, err := tokenizer.FromGGUF(g)
	if err != nil {
		t.Fatalf("FromGGUF: %v", err)
	}
	return v
}

func TestBuildExamplesRealVocab(t *testing.T) {
	v := testVocab(t)
	d, err := Open(filepath.Join("testdata", "extractor"))
	if err != nil {
		t.Fatal(err)
	}
	recs, err := d.Sessions()
	if err != nil {
		t.Fatal(err)
	}
	exs, err := BuildExamples(v, recs, DefaultBuildOptions())
	if err != nil {
		t.Fatal(err)
	}
	if len(exs) != 2 {
		t.Fatalf("got %d examples, want 2", len(exs))
	}
	for _, e := range exs {
		if len(e.IDs) == 0 || len(e.IDs) != len(e.Mask) {
			t.Fatalf("example %s: len(ids)=%d len(mask)=%d", e.Source, len(e.IDs), len(e.Mask))
		}
		if e.LossTokens == 0 {
			t.Fatalf("example %s has no loss tokens", e.Source)
		}
		if e.LossTokens == len(e.IDs) {
			t.Fatalf("example %s has no masked context", e.Source)
		}
	}
}

func TestBuildExampleTruncation(t *testing.T) {
	v := testVocab(t)
	d, err := Open(filepath.Join("testdata", "extractor"))
	if err != nil {
		t.Fatal(err)
	}
	recs, err := d.Sessions()
	if err != nil {
		t.Fatal(err)
	}
	opt := DefaultBuildOptions()
	opt.MaxSeqLen = 32
	exs, err := BuildExamples(v, recs, opt)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range exs {
		if len(e.IDs) > 32 {
			t.Fatalf("example %s not truncated: %d tokens", e.Source, len(e.IDs))
		}
		if e.LossTokens == 0 {
			t.Fatalf("example %s lost all targets under truncation", e.Source)
		}
	}
}

func TestExamplesAutoLoadTools(t *testing.T) {
	v := testVocab(t)
	d, err := Open(filepath.Join("testdata", "extractor"))
	if err != nil {
		t.Fatal(err)
	}
	recs, err := d.Sessions()
	if err != nil {
		t.Fatal(err)
	}
	// The fixture records carry no per-session tools, so the free builder with
	// empty options renders no tools preamble.
	free, err := BuildExamples(v, recs, DefaultBuildOptions())
	if err != nil {
		t.Fatal(err)
	}
	// Dataset.Examples auto-loads the manifest registry, which adds the tools
	// preamble and therefore more tokens.
	auto, err := d.Examples(v, DefaultBuildOptions())
	if err != nil {
		t.Fatal(err)
	}
	if len(auto) != len(free) {
		t.Fatalf("example count changed: %d vs %d", len(auto), len(free))
	}
	if TotalTokens(auto) <= TotalTokens(free) {
		t.Fatalf("auto-loaded tools did not add tokens: %d <= %d", TotalTokens(auto), TotalTokens(free))
	}
}

func TestIterMatchesBuild(t *testing.T) {
	v := testVocab(t)
	d, err := Open(filepath.Join("testdata", "extractor"))
	if err != nil {
		t.Fatal(err)
	}
	recs, _ := d.Sessions()
	want, err := BuildExamples(v, recs, DefaultBuildOptions())
	if err != nil {
		t.Fatal(err)
	}

	out, errs := Iter(context.Background(), d, v, DefaultBuildOptions())
	var got []Example
	for e := range out {
		got = append(got, e)
	}
	if err := <-errs; err != nil {
		t.Fatal(err)
	}
	if len(got) != len(want) {
		t.Fatalf("iter got %d, build got %d", len(got), len(want))
	}
}

func TestShuffleDeterministic(t *testing.T) {
	ex := make([]Example, 20)
	for i := range ex {
		ex[i] = Example{Source: string(rune('a' + i))}
	}
	a := append([]Example(nil), ex...)
	b := append([]Example(nil), ex...)
	Shuffle(a, 42)
	Shuffle(b, 42)
	if !reflect.DeepEqual(a, b) {
		t.Fatal("shuffle not deterministic for a fixed seed")
	}
	if reflect.DeepEqual(a, ex) {
		t.Fatal("shuffle produced no change")
	}
}

func TestSplitByHash(t *testing.T) {
	var ex []Example
	for i := 0; i < 1000; i++ {
		ex = append(ex, Example{Source: string(rune(i)) + "-session"})
	}
	train, val := Split(ex, 0.1)
	if len(train)+len(val) != len(ex) {
		t.Fatalf("split lost examples: %d + %d != %d", len(train), len(val), len(ex))
	}
	// Deterministic: same input yields the same partitions.
	train2, val2 := Split(ex, 0.1)
	if len(train) != len(train2) || len(val) != len(val2) {
		t.Fatal("split not deterministic")
	}
	if len(val) == 0 || len(val) == len(ex) {
		t.Fatalf("val partition degenerate: %d/%d", len(val), len(ex))
	}
}
