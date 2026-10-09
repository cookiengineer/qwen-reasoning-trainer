package cli

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// readPromptFile loads prompts from either a plain-text file (one prompt per
// line, blank lines and lines starting with '#' skipped) or a JSONL/NDJSON file
// (one JSON object per line), in which case column selects the string field to
// read (default "text"). JSONL is detected from the file extension (.jsonl,
// .ndjson, .json) or forced by a non-empty column.
func readPromptFile(path, column string, limit int) ([]string, error) {
	lower := strings.ToLower(path)
	jsonl := column != "" ||
		strings.HasSuffix(lower, ".jsonl") ||
		strings.HasSuffix(lower, ".ndjson") ||
		strings.HasSuffix(lower, ".json")
	if column == "" {
		column = "text"
	}

	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var out []string
	sc := bufio.NewScanner(f)
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
				return nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
			}
			raw, ok := obj[column]
			if !ok {
				return nil, fmt.Errorf("%s:%d: missing column %q", path, lineNo, column)
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
