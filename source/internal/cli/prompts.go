package cli

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"strings"

	"github.com/cookiengineer/qwen-reasoning-trainer/prompts"
)

// Bundled prompt sets embedded from source/prompts, mirroring Heretic's
// defaults (the mlabonne datasets published under heretic-org on Hugging Face).
const (
	BundledGoodPrompts     = "harmless_alpaca_train.jsonl"
	BundledBadPrompts      = "harmful_behaviors_train.jsonl"
	BundledEvalGoodPrompts = "harmless_alpaca_test.jsonl"
	BundledEvalBadPrompts  = "harmful_behaviors_test.jsonl"
)

// promptSource returns a human-readable label for a prompt set: the file path
// when one is given, otherwise the embedded bundled name.
func promptSource(path, bundled string) string {
	if path == "" {
		return "bundled:" + bundled
	}
	return path
}

// loadPrompts reads a prompt set. When path is empty, bundled selects one of
// the embedded sets (see BundledGoodPrompts and friends); otherwise path is
// read from disk. It returns a helpful error when the set is missing or empty.
func loadPrompts(path, bundled, column string, limit int) ([]string, error) {
	var (
		rows []string
		err  error
	)
	if path == "" {
		rows, err = readPromptFS(prompts.FS, bundled, column, limit)
	} else {
		rows, err = readPromptFile(path, column, limit)
	}
	if err != nil {
		return nil, err
	}
	if len(rows) == 0 {
		return nil, fmt.Errorf("%s: no prompts found", promptSource(path, bundled))
	}
	return rows, nil
}

// readPromptFile loads a prompt set from a filesystem path.
func readPromptFile(path, column string, limit int) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readPrompt(f, path, column, limit)
}

// readPromptFS loads a bundled prompt set from the embedded filesystem.
func readPromptFS(fsys fs.FS, name, column string, limit int) ([]string, error) {
	f, err := fsys.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return readPrompt(f, name, column, limit)
}

// readPrompt loads prompts from either a plain-text stream (one prompt per
// line, blank lines and lines starting with '#' skipped) or a JSONL/NDJSON
// stream (one JSON object per line), in which case column selects the string
// field to read (default "text"). JSONL is detected from the name's extension
// (.jsonl, .ndjson, .json) or forced by a non-empty column.
func readPrompt(r io.Reader, name, column string, limit int) ([]string, error) {
	lower := strings.ToLower(name)
	jsonl := column != "" ||
		strings.HasSuffix(lower, ".jsonl") ||
		strings.HasSuffix(lower, ".ndjson") ||
		strings.HasSuffix(lower, ".json")
	if column == "" {
		column = "text"
	}

	var out []string
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	lineNo := 0
	for sc.Scan() {
		lineNo++
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if jsonl {
			var obj map[string]json.RawMessage
			if err := json.Unmarshal([]byte(line), &obj); err != nil {
				return nil, fmt.Errorf("%s:%d: %w", name, lineNo, err)
			}
			raw, ok := obj[column]
			if !ok {
				return nil, fmt.Errorf("%s:%d: missing column %q", name, lineNo, column)
			}
			var s string
			if err := json.Unmarshal(raw, &s); err != nil {
				// Accept a non-string scalar by using its raw JSON text.
				s = strings.Trim(string(raw), `"`)
			}
			line = strings.TrimSpace(s)
			if line == "" {
				continue
			}
		}
		out = append(out, line)
		if limit > 0 && len(out) >= limit {
			break
		}
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}
