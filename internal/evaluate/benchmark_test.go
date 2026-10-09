package evaluate

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

func TestLoadTasks(t *testing.T) {
	items, err := LoadTasks("testdata/tasks.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 3 {
		t.Fatalf("loaded %d items, want 3", len(items))
	}
	if items[0].ID != "cap" || items[0].Answer != 0 || len(items[0].Choices) != 2 {
		t.Fatalf("unexpected first item: %+v", items[0])
	}
}

func TestLoadTasksErrors(t *testing.T) {
	cases := map[string]string{
		"bad-json":    `{"context":`,
		"one-choice":  `{"context":"x","choices":["a"],"answer":0}`,
		"answer-high": `{"context":"x","choices":["a","b"],"answer":2}`,
		"answer-neg":  `{"context":"x","choices":["a","b"],"answer":-1}`,
	}
	for name, line := range cases {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "tasks.jsonl")
			if err := os.WriteFile(path, []byte(line+"\n"), 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadTasks(path); err == nil {
				t.Fatalf("expected error for %s", line)
			}
		})
	}
}

func TestLoadTasksBlankLines(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tasks.jsonl")
	body := "\n" + `{"context":"x","choices":["a","b"],"answer":1}` + "\n\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	items, err := LoadTasks(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 {
		t.Fatalf("loaded %d items, want 1", len(items))
	}
}

func TestMultipleChoiceRange(t *testing.T) {
	cfg := testConfig()
	m := qwen38.NewModel(qwen38.NewRandom(cfg, 21))
	vocab := testVocab()
	items, err := LoadTasks("testdata/tasks.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	for _, normalize := range []bool{false, true} {
		results, acc, err := MultipleChoice(m, vocab, items, normalize)
		if err != nil {
			t.Fatal(err)
		}
		if len(results) != len(items) {
			t.Fatalf("got %d results, want %d", len(results), len(items))
		}
		if math.IsNaN(acc) || acc < 0 || acc > 1 {
			t.Fatalf("accuracy out of range: %v", acc)
		}
		for _, r := range results {
			if len(r.Scores) != 2 && len(r.Scores) != 3 {
				t.Fatalf("scores length %d", len(r.Scores))
			}
			if r.Predicted < 0 || r.Predicted >= len(r.Scores) {
				t.Fatalf("predicted %d out of range", r.Predicted)
			}
		}
	}
}

func TestMultipleChoiceEmpty(t *testing.T) {
	cfg := testConfig()
	m := qwen38.NewModel(qwen38.NewRandom(cfg, 22))
	results, acc, err := MultipleChoice(m, testVocab(), nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 0 || acc != 0 {
		t.Fatalf("empty items gave results=%d acc=%v", len(results), acc)
	}
}
