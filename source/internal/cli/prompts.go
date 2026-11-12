package cli

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"sort"
	"strings"

	"github.com/cookiengineer/qwen-reasoning-trainer/prompts"
)

// Bundled prompt sets embedded from source/prompts, mirroring Heretic's
// defaults (the mlabonne datasets published under heretic-org on Hugging Face).
const (
	BundledHereticHarmlessTrain = "heretic_harmless_train.jsonl"
	BundledHereticHarmfulTrain  = "heretic_harmful_train.jsonl"
	BundledHereticHarmlessTest  = "heretic_harmless_test.jsonl"
	BundledHereticHarmfulTest   = "heretic_harmful_test.jsonl"

	BundledCybersecurityGoodTrain = "cybersecurity_defensive_train.jsonl"
	BundledCybersecurityBadTrain  = "cybersecurity_offensive_train.jsonl"
	BundledCybersecurityGoodTest  = "cybersecurity_defensive_test.jsonl"
	BundledCybersecurityBadTest   = "cybersecurity_offensive_test.jsonl"
)

// Catalog names accepted by --catalog.
const (
	CatalogHeretic       = "heretic"
	CatalogCybersecurity = "cybersecurity"
)

// DefaultCatalog is used when --catalog is not supplied.
const DefaultCatalog = CatalogHeretic

// Catalog is a named set of bundled abliteration prompt sets. Good prompts are
// desirable/benign behaviour used to compute the direction; bad prompts are
// undesirable/refusal-triggering prompts.
type Catalog struct {
	Name        string
	Description string
	GoodTrain   string
	BadTrain    string
	GoodTest    string
	BadTest     string
}

// BundledCatalogs maps a catalog name to its embedded prompt files.
var BundledCatalogs = map[string]Catalog{
	CatalogHeretic: {
		Name:        CatalogHeretic,
		Description: "Heretic's general-purpose harmless/harmful prompt sets",
		GoodTrain:   BundledHereticHarmlessTrain,
		BadTrain:    BundledHereticHarmfulTrain,
		GoodTest:    BundledHereticHarmlessTest,
		BadTest:     BundledHereticHarmfulTest,
	},
	CatalogCybersecurity: {
		Name:        CatalogCybersecurity,
		Description: "cybersecurity questions: defensive good prompts and offensive/dual-use bad prompts",
		GoodTrain:   BundledCybersecurityGoodTrain,
		BadTrain:    BundledCybersecurityBadTrain,
		GoodTest:    BundledCybersecurityGoodTest,
		BadTest:     BundledCybersecurityBadTest,
	},
}

// CatalogNames returns the sorted names of the bundled catalogs.
func CatalogNames() []string {
	names := make([]string, 0, len(BundledCatalogs))
	for name := range BundledCatalogs {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// LookupCatalog resolves a catalog name, treating the empty string as the
// default catalog.
func LookupCatalog(name string) (Catalog, error) {
	if name == "" {
		name = DefaultCatalog
	}
	c, ok := BundledCatalogs[name]
	if !ok {
		return Catalog{}, fmt.Errorf("unknown catalog %q (want %s)", name, strings.Join(CatalogNames(), ", "))
	}
	return c, nil
}

// promptSource returns a human-readable label for a prompt set: the file path
// when one is given, otherwise the embedded bundled name.
func promptSource(path, bundled string) string {
	if path == "" {
		return "bundled:" + bundled
	}
	return path
}

// loadPrompts reads a prompt set. When path is empty, bundled selects one of
// the embedded sets (see BundledHereticHarmlessTrain and friends); otherwise
// path is read from disk. It returns a helpful error when the set is missing or
// empty.
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
