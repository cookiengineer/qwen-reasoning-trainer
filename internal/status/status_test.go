package status

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSaveReadRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sub", "status.json")
	s := New("train", "models/x.gguf", 100, "step")
	s.Phase = "training"
	s.Done = 27
	s.Loss = 1.25
	s.BestLoss = 0.5
	s.LR = 1e-4
	s.CheckpointEvery = 25
	s.CheckpointDone = 1
	s.CheckpointTotal = 4
	s.CheckpointPath = "models/ck.ckpt"
	s.ETASeconds = 3600
	if err := writeFile(path, s); err != nil {
		t.Fatal(err)
	}
	got, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Kind != "train" || got.Done != 27 || got.Total != 100 {
		t.Fatalf("unexpected round-trip: %+v", got)
	}
	if got.CheckpointDone != 1 || got.CheckpointTotal != 4 {
		t.Fatalf("checkpoint fields lost: %+v", got)
	}
	if got.Loss != 1.25 || got.BestLoss != 0.5 {
		t.Fatalf("loss fields lost: %+v", got)
	}
}

func TestProgress(t *testing.T) {
	s := &Status{Total: 100, Done: 27}
	if p := s.Progress(); p < 0.269 || p > 0.271 {
		t.Fatalf("progress %v", p)
	}
	// Unknown total -> 0; over-total clamps to 1.
	if (&Status{Done: 5}).Progress() != 0 {
		t.Fatal("unknown total should be 0")
	}
	if (&Status{Total: 10, Done: 20}).Progress() != 1 {
		t.Fatal("over-total should clamp to 1")
	}
}

func TestRunning(t *testing.T) {
	host, _ := os.Hostname()
	s := &Status{Host: host, PID: 1}
	if local, alive := s.Running(); !local || !alive {
		t.Fatalf("pid 1 should be local+alive, got %v/%v", local, alive)
	}
	s.PID = 1 << 30 // implausible pid
	if local, alive := s.Running(); !local || alive {
		t.Fatalf("bogus pid should be local+dead, got %v/%v", local, alive)
	}
	s.Host = "some-other-host"
	if local, alive := s.Running(); local || !alive {
		t.Fatalf("remote host should be remote+unknown(alive), got %v/%v", local, alive)
	}
}

func TestRender(t *testing.T) {
	s := &Status{
		Kind: "train", Model: "m.gguf", Host: "h", PID: 42,
		StartedAt: time.Now().Add(-2 * time.Hour), UpdatedAt: time.Now(),
		Phase: "training", Unit: "step", Done: 270, Total: 1000,
		CheckpointEvery: 25, CheckpointDone: 10, CheckpointTotal: 40,
		CheckpointPath: "ck.ckpt", Loss: 1.2, BestLoss: 0.4, LR: 1e-4,
		ETASeconds: 7200, Message: "step 270",
	}
	var b bytes.Buffer
	Render(&b, s)
	out := b.String()
	for _, want := range []string{"qwen-trainer status", "27.0%", "270/1000", "10/40", "eta", "step 270"} {
		if !strings.Contains(out, want) {
			t.Errorf("render missing %q\n%s", want, out)
		}
	}
}

func TestWriterFinish(t *testing.T) {
	path := filepath.Join(t.TempDir(), "status.json")
	w := NewWriter(path, New("search", "m", 10, "trial"))
	if w.Err() != nil {
		t.Fatal(w.Err())
	}
	w.Status().Done = 3
	w.Save()
	w.Finish("done")
	got, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.Phase != "done" || got.FinishedAt == nil {
		t.Fatalf("finish not persisted: %+v", got)
	}
}
