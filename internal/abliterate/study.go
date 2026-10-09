package abliterate

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// TrialRecord is the persisted form of a trial. It is TrialResult with the
// decoded params, so a resumed search can rescore or re-apply the best trial
// without re-decoding.
type TrialRecord struct {
	Index  int         `json:"index"`
	Unit   []float64   `json:"unit"`
	Params TrialParams `json:"params"`
	Score  float64     `json:"score"`
}

// LoadStudy reads trial records from a JSON Lines study file. A missing file
// yields an empty slice and no error. Blank lines and lines starting with '#'
// are skipped.
func LoadStudy(path string) ([]TrialResult, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	var out []TrialResult
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	line := 0
	for sc.Scan() {
		line++
		text := strings.TrimSpace(sc.Text())
		if text == "" || strings.HasPrefix(text, "#") {
			continue
		}
		var rec TrialRecord
		if err := json.Unmarshal([]byte(text), &rec); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, line, err)
		}
		out = append(out, TrialResult{Index: rec.Index, Unit: rec.Unit, Params: rec.Params, Score: rec.Score})
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	return out, nil
}

// AppendStudy appends one trial to a JSON Lines study file, creating it if
// needed.
func AppendStudy(path string, r TrialResult) error {
	rec := TrialRecord{Index: r.Index, Unit: r.Unit, Params: r.Params, Score: r.Score}
	line, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := f.Write(append(line, '\n')); err != nil {
		return err
	}
	return nil
}
