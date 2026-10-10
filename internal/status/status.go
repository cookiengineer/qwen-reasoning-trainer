// Package status provides a small, self-describing progress file that
// long-running commands (train, abliterate, search) update periodically and
// `qwen-trainer status` reads back. It lets you ask "what's the progress?" of a
// training session running in another terminal, tmux, or host.
package status

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

// EnvPath is the environment variable that overrides the default status path.
const EnvPath = "QWEN38_STATUS_FILE"

// Status is the machine-readable progress of one running (or finished) job.
type Status struct {
	Kind            string     `json:"kind"` // train|abliterate|search
	Model           string     `json:"model,omitempty"`
	Host            string     `json:"host,omitempty"`
	PID             int        `json:"pid"`
	StartedAt       time.Time  `json:"started_at"`
	UpdatedAt       time.Time  `json:"updated_at"`
	FinishedAt      *time.Time `json:"finished_at,omitempty"`
	Phase           string     `json:"phase,omitempty"`
	Unit            string     `json:"unit"` // step|trial|matrix|tensor|prompt
	Done            int        `json:"done"`
	Total           int        `json:"total"`
	CheckpointEvery int        `json:"checkpoint_every,omitempty"`
	CheckpointDone  int        `json:"checkpoint_done,omitempty"`
	CheckpointTotal int        `json:"checkpoint_total,omitempty"`
	CheckpointPath  string     `json:"checkpoint_path,omitempty"`
	Loss            float64    `json:"loss,omitempty"`
	BestLoss        float64    `json:"best_loss,omitempty"`
	LR              float64    `json:"lr,omitempty"`
	Objective       float64    `json:"objective,omitempty"`
	ETASeconds      float64    `json:"eta_seconds"`
	Message         string     `json:"message,omitempty"`
}

// DefaultPath resolves the status file path: $QWEN38_STATUS_FILE if set, else
// $XDG_STATE_HOME/qwen-trainer/status.json, else ~/.local/state/qwen-trainer/.
func DefaultPath() string {
	if p := os.Getenv(EnvPath); p != "" {
		return p
	}
	base := os.Getenv("XDG_STATE_HOME")
	if base == "" {
		if home, err := os.UserHomeDir(); err == nil {
			base = filepath.Join(home, ".local", "state")
		}
	}
	if base == "" {
		return "qwen-trainer-status.json"
	}
	return filepath.Join(base, "qwen-trainer", "status.json")
}

// Path resolves the status path from an explicit value (global flag) if set,
// otherwise the env var / default.
func Path(explicit string) string {
	if explicit != "" {
		return explicit
	}
	return DefaultPath()
}

// New creates a status record for a job that will run Total units of work.
func New(kind, model string, total int, unit string) *Status {
	host, _ := os.Hostname()
	now := time.Now()
	return &Status{
		Kind:      kind,
		Model:     model,
		Host:      host,
		PID:       os.Getpid(),
		StartedAt: now,
		UpdatedAt: now,
		Unit:      unit,
		Total:     total,
	}
}

// Progress returns the fraction complete in [0,1] (0 when Total is unknown).
func (s *Status) Progress() float64 {
	if s.Total <= 0 {
		return 0
	}
	p := float64(s.Done) / float64(s.Total)
	if p < 0 {
		return 0
	}
	if p > 1 {
		return 1
	}
	return p
}

// Elapsed returns wall time since the job started (frozen at FinishedAt).
func (s *Status) Elapsed() time.Duration {
	end := time.Now()
	if s.FinishedAt != nil {
		end = *s.FinishedAt
	}
	return end.Sub(s.StartedAt)
}

// Writer periodically persists a Status and remembers the first write error.
type Writer struct {
	path string
	s    *Status
	err  error
}

// NewWriter creates a writer for path and immediately writes the first snapshot.
func NewWriter(path string, s *Status) *Writer {
	w := &Writer{path: path, s: s}
	w.Save()
	return w
}

// Status returns the live record to mutate between Save calls.
func (w *Writer) Status() *Status { return w.s }

// Save persists the current snapshot with a fresh UpdatedAt. The first failure
// is retained (and returned by Err) so callers can warn once.
func (w *Writer) Save() {
	w.s.UpdatedAt = time.Now()
	if err := writeFile(w.path, w.s); err != nil && w.err == nil {
		w.err = err
	}
}

// Err returns the first write error, if any.
func (w *Writer) Err() error { return w.err }

// Finish marks the job done (or failed) and writes a final snapshot.
func (w *Writer) Finish(phase string) {
	now := time.Now()
	w.s.Phase = phase
	w.s.Message = ""
	w.s.FinishedAt = &now
	w.Save()
}

func writeFile(path string, s *Status) error {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, append(data, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Read loads a status file.
func Read(path string) (*Status, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var s Status
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// Running reports whether the recorded PID is still alive on this host. It
// returns (local, alive); local is false when the record is from another host,
// in which case alive is unknown (reported true so the reader does not claim it
// stopped).
func (s *Status) Running() (local, alive bool) {
	host, _ := os.Hostname()
	if s.Host != "" && host != "" && s.Host != host {
		return false, true
	}
	if s.PID <= 0 {
		return true, false
	}
	proc, err := os.FindProcess(s.PID)
	if err != nil {
		return true, false
	}
	// Signal 0 probes existence without delivering a signal. A permission
	// error (EPERM, e.g. probing pid 1) still means the process exists.
	if err := proc.Signal(syscall.Signal(0)); err != nil {
		if errors.Is(err, os.ErrProcessDone) {
			return true, false
		}
		return true, true
	}
	return true, true
}

func humanDuration(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	d = d.Round(time.Second)
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	sec := int(d.Seconds()) % 60
	if h > 0 {
		return fmt.Sprintf("%dh%02dm%02ds", h, m, sec)
	}
	if m > 0 {
		return fmt.Sprintf("%dm%02ds", m, sec)
	}
	return fmt.Sprintf("%ds", sec)
}

// Render writes a human-readable progress summary.
func Render(w io.Writer, s *Status) {
	local, alive := s.Running()
	state := "not running"
	if !local {
		state = "running on " + s.Host
	} else if alive {
		state = "running"
	} else if s.FinishedAt != nil {
		state = "finished"
	}
	if s.Phase == "failed" {
		state = "failed"
	}

	fmt.Fprintf(w, "qwen-trainer status  (%s)\n", s.Kind)
	if s.Model != "" {
		fmt.Fprintf(w, "  model       %s\n", s.Model)
	}
	fmt.Fprintf(w, "  host        %s   pid %d   %s\n", s.Host, s.PID, state)
	if s.Phase != "" {
		fmt.Fprintf(w, "  phase       %s\n", s.Phase)
	}

	const width = 28
	filled := int(s.Progress() * width)
	if filled > width {
		filled = width
	}
	bar := strings.Repeat("#", filled) + strings.Repeat("-", width-filled)
	if s.Total > 0 {
		fmt.Fprintf(w, "  progress    [%s]  %5.1f%%\n", bar, s.Progress()*100)
		fmt.Fprintf(w, "  %-11s %d/%d\n", s.Unit, s.Done, s.Total)
	} else {
		fmt.Fprintf(w, "  %-11s %d\n", s.Unit, s.Done)
	}
	if s.CheckpointTotal > 0 {
		extra := ""
		if s.CheckpointEvery > 0 {
			extra = fmt.Sprintf(" (every %d %ss)", s.CheckpointEvery, s.Unit)
		}
		if s.CheckpointPath != "" {
			extra += " -> " + s.CheckpointPath
		}
		fmt.Fprintf(w, "  checkpoint  %d/%d%s\n", s.CheckpointDone, s.CheckpointTotal, extra)
	}
	if s.Kind == "search" && s.Objective != 0 {
		fmt.Fprintf(w, "  objective   %.4f (best so far)\n", s.Objective)
	} else if s.Loss != 0 || s.BestLoss != 0 {
		fmt.Fprintf(w, "  loss        %.4f (best %.4f)  lr %.3e\n", s.Loss, s.BestLoss, s.LR)
	}
	fmt.Fprintf(w, "  elapsed     %s\n", humanDuration(s.Elapsed()))
	if s.FinishedAt == nil {
		fmt.Fprintf(w, "  eta         %s\n", humanDuration(time.Duration(s.ETASeconds*float64(time.Second))))
	}
	fmt.Fprintf(w, "  updated     %s ago\n", humanDuration(time.Since(s.UpdatedAt)))
	if s.Message != "" {
		fmt.Fprintf(w, "  message     %s\n", s.Message)
	}
}
