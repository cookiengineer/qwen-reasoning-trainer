package fsutil

import (
	"os"
	"path/filepath"
	"testing"
)

func TestAtomicWriterCommitReplaces(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out.bin")
	if err := os.WriteFile(path, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}

	aw, err := Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := aw.Write([]byte("new contents")); err != nil {
		t.Fatal(err)
	}
	if err := aw.Commit(); err != nil {
		t.Fatal(err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "new contents" {
		t.Fatalf("path = %q, want %q", got, "new contents")
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("temp file left behind: %v", err)
	}
}

func TestAtomicWriterAbortKeepsOriginal(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out.bin")
	if err := os.WriteFile(path, []byte("original"), 0o644); err != nil {
		t.Fatal(err)
	}

	aw, err := Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := aw.Write([]byte("half written")); err != nil {
		t.Fatal(err)
	}
	if err := aw.Abort(); err != nil {
		t.Fatal(err)
	}

	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "original" {
		t.Fatalf("path = %q, want %q", got, "original")
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf("temp file left behind: %v", err)
	}
}

func TestAtomicWriterCreatesParents(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "deep", "out.bin")
	aw, err := Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := aw.Write([]byte("x")); err != nil {
		t.Fatal(err)
	}
	if err := aw.Commit(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("committed file missing: %v", err)
	}
}
