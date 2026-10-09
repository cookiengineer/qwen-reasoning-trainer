package cli

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func writeTemp(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadPromptFileText(t *testing.T) {
	path := writeTemp(t, "prompts.txt", "first prompt\n\n# comment\nsecond prompt\n")
	got, err := readPromptFile(path, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"first prompt", "second prompt"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestReadPromptFileTextLimit(t *testing.T) {
	path := writeTemp(t, "prompts.txt", "a\nb\nc\n")
	got, err := readPromptFile(path, "", 2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []string{"a", "b"}) {
		t.Fatalf("got %v", got)
	}
}

func TestReadPromptFileJSONL(t *testing.T) {
	path := writeTemp(t, "data.jsonl",
		"{\"text\":\"hello\",\"id\":1}\n{\"text\":\"world\",\"id\":2}\n")
	got, err := readPromptFile(path, "", 0)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []string{"hello", "world"}) {
		t.Fatalf("got %v", got)
	}
}

func TestReadPromptFileJSONLColumn(t *testing.T) {
	path := writeTemp(t, "data.txt", "{\"prompt\":\"one\"}\n{\"prompt\":\"two\"}\n")
	got, err := readPromptFile(path, "prompt", 0)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, []string{"one", "two"}) {
		t.Fatalf("got %v", got)
	}
}

func TestReadPromptFileMissingColumn(t *testing.T) {
	path := writeTemp(t, "data.jsonl", "{\"other\":\"x\"}\n")
	if _, err := readPromptFile(path, "", 0); err == nil {
		t.Fatal("expected error for missing column")
	}
}

func TestReadPromptFileMalformedJSONL(t *testing.T) {
	path := writeTemp(t, "data.jsonl", "not json\n")
	if _, err := readPromptFile(path, "", 0); err == nil {
		t.Fatal("expected error for malformed JSONL")
	}
}
