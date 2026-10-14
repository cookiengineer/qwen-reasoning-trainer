package gguf

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

func TestWriteStreamToFileAtomicOnError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.gguf")
	if err := os.WriteFile(path, []byte("original"), 0o644); err != nil {
		t.Fatal(err)
	}
	tensors := []StreamTensor{{
		Name: "t", Dims: []uint64{4}, Type: quant.TypeF32, Size: 16,
		Write: func(w io.Writer) error {
			_, _ = w.Write(make([]byte, 8))
			return fmt.Errorf("boom")
		},
	}}
	if err := WriteStreamToFile(path, nil, tensors); err == nil {
		t.Fatal("want error")
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "original" {
		t.Fatalf("original clobbered: %q", got)
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("temp file left behind: %v", err)
	}
}

func TestWriteStreamToFileProgress(t *testing.T) {
	path := filepath.Join(t.TempDir(), "out.gguf")
	var seen []int
	fill := func(w io.Writer) error { _, err := w.Write(make([]byte, 16)); return err }
	tensors := []StreamTensor{
		{Name: "a", Dims: []uint64{4}, Type: quant.TypeF32, Size: 16, Write: fill},
		{Name: "b", Dims: []uint64{4}, Type: quant.TypeF32, Size: 16, Write: fill},
	}
	err := WriteStreamToFileProgress(path, nil, tensors, func(done, total int) {
		seen = append(seen, done)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(seen) != 2 || seen[0] != 1 || seen[1] != 2 {
		t.Fatalf("progress = %v, want [1 2]", seen)
	}
	g, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if len(g.Tensors) != 2 {
		t.Fatalf("tensors = %d, want 2", len(g.Tensors))
	}
}
