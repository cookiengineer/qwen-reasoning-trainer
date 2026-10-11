package abliterate

import (
	"path/filepath"
	"testing"
)

func TestStudyRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "study.jsonl")
	if got, err := LoadStudy(path); err != nil || len(got) != 0 {
		t.Fatalf("missing study: got %v err %v", got, err)
	}
	records := []TrialResult{
		{Index: 0, Unit: []float64{0.1, 0.2}, Params: TrialParams{DirectionGlobal: true, DirectionIndex: 3}, Score: 1.5},
		{Index: 1, Unit: []float64{0.9, 0.8}, Params: TrialParams{DirectionGlobal: false, DirectionIndex: 0}, Score: 0.25},
	}
	for _, r := range records {
		if err := AppendStudy(path, r); err != nil {
			t.Fatal(err)
		}
	}
	got, err := LoadStudy(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(records) {
		t.Fatalf("got %d records, want %d", len(got), len(records))
	}
	for i := range records {
		if got[i].Index != records[i].Index || got[i].Score != records[i].Score {
			t.Fatalf("record %d mismatch: %+v vs %+v", i, got[i], records[i])
		}
		if len(got[i].Unit) != 2 || got[i].Unit[0] != records[i].Unit[0] {
			t.Fatalf("record %d unit mismatch: %v", i, got[i].Unit)
		}
	}
}
