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

// TestBundledHereticPrompts validates the prompt sets mirrored from Heretic's
// defaults, if the repository's prompts/ directory is present.
func TestBundledHereticPrompts(t *testing.T) {
	cases := []struct {
		file string
		want int
	}{
		{"harmful_behaviors_train.jsonl", 416},
		{"harmful_behaviors_test.jsonl", 104},
		{"harmless_alpaca_train.jsonl", 25058},
		{"harmless_alpaca_test.jsonl", 6265},
	}
	for _, c := range cases {
		path := filepath.Join("..", "..", "prompts", c.file)
		if _, err := os.Stat(path); err != nil {
			t.Skipf("bundled %s not present", c.file)
		}
		rows, err := readPromptFile(path, "", 0)
		if err != nil {
			t.Fatalf("%s: %v", c.file, err)
		}
		if len(rows) != c.want {
			t.Fatalf("%s: %d rows, want %d", c.file, len(rows), c.want)
		}
		for i, r := range rows {
			if r == "" {
				t.Fatalf("%s: row %d is empty", c.file, i)
			}
		}
	}
}

func TestLoadPrompts(t *testing.T) {
	if _, err := loadPrompts(filepath.Join(t.TempDir(), "missing.jsonl"), "", 0); err == nil {
		t.Fatal("expected error for missing file")
	}
	ok := writeTemp(t, "p.txt", "a\nb\n")
	rows, err := loadPrompts(ok, "", 0)
	if err != nil || len(rows) != 2 {
		t.Fatalf("loadPrompts = %v, %v", rows, err)
	}
	empty := writeTemp(t, "e.txt", "\n# comment\n")
	if _, err := loadPrompts(empty, "", 0); err == nil {
		t.Fatal("expected error for empty file")
	}
}
