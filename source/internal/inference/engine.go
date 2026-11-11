// Package inference provides the small generation engine shared by the
// qwen-server command: model loading, backend selection, chat rendering,
// sampling, and streaming parsing.
package inference

import (
	"fmt"
	"io"
	"sync"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/tokenizer"
)

// gpuHeadroomBytes reserves memory for activations, scratch, and state on top of
// the model when deciding whether a Vulkan device can host the model.
const gpuHeadroomBytes = 2 << 30

// LoadOptions configures model loading.
type LoadOptions struct {
	Path string
	// Device is "auto" (default), "cpu", or "vulkan".
	Device string
	// ContextLength overrides the GGUF context length when > 0.
	ContextLength int
	// Log receives backend selection diagnostics. Nil discards them.
	Log io.Writer
	// LogLevel is "quiet", "info", or "debug"; quiet silences backend logging.
	LogLevel string
}

// Engine owns a loaded model and serializes generation. The model's device
// weight cache and the Vulkan backend are not safe for concurrent use, so all
// requests share one generation slot.
type Engine struct {
	Model   *qwen38.Model
	Vocab   *tokenizer.Vocab
	Cfg     *modelcfg.Config
	Backend compute.Backend

	file       *gguf.File
	contextLen int
	imEnd      int32
	hasImEnd   bool

	sem   chan struct{}
	close sync.Once
}

// Load opens a GGUF, loads its weights, and selects a compute backend.
func Load(opt LoadOptions) (*Engine, error) {
	if opt.Path == "" {
		return nil, fmt.Errorf("inference: model path is required")
	}
	g, err := gguf.Open(opt.Path)
	if err != nil {
		return nil, err
	}
	mc, err := modelcfg.FromGGUF(g)
	if err != nil {
		g.Close()
		return nil, err
	}
	vocab, err := tokenizer.FromGGUF(g)
	if err != nil {
		g.Close()
		return nil, err
	}
	w, err := qwen38.LoadFromGGUF(g, mc)
	if err != nil {
		g.Close()
		return nil, err
	}
	be, err := selectBackend(opt, modelBytes(g)+gpuHeadroomBytes)
	if err != nil {
		g.Close()
		return nil, err
	}
	m := qwen38.NewModel(w)
	m.Backend = be

	ctx := mc.ContextLength
	if opt.ContextLength > 0 {
		ctx = opt.ContextLength
	}
	if ctx <= 0 {
		ctx = 32768
	}

	e := &Engine{
		Model:      m,
		Vocab:      vocab,
		Cfg:        mc,
		Backend:    be,
		file:       g,
		contextLen: ctx,
		sem:        make(chan struct{}, 1),
	}
	if id, ok := vocab.SpecialID("<|im_end|>"); ok {
		e.imEnd, e.hasImEnd = id, true
	}
	return e, nil
}

// ContextLength returns the maximum number of positions the server will use.
func (e *Engine) ContextLength() int { return e.contextLen }

// Close releases device weights, the backend, and the GGUF file.
func (e *Engine) Close() error {
	var err error
	e.close.Do(func() {
		if e.Model != nil {
			e.Model.FreeDevice()
		}
		if e.Backend != nil {
			e.Backend.Close()
		}
		if e.file != nil {
			err = e.file.Close()
		}
	})
	return err
}

func selectBackend(opt LoadOptions, need uint64) (compute.Backend, error) {
	device := opt.Device
	if device == "" {
		device = "auto"
	}
	switch device {
	case "cpu":
		return nil, nil
	case "auto", "vulkan", "gpu":
	default:
		return nil, fmt.Errorf("inference: unknown device %q (want auto, cpu, or vulkan)", device)
	}

	be, err := vulkan.New()
	if err != nil {
		if device == "auto" {
			logBackend(opt, "gpu: unavailable (%v); using CPU\n", err)
			return nil, nil
		}
		return nil, fmt.Errorf("inference: vulkan unavailable: %w", err)
	}
	caps := be.Capabilities()
	if caps.MemoryBytes > 0 && caps.MemoryBytes < need {
		be.Close()
		if device == "auto" {
			logBackend(opt, "gpu: %s has %s, need ~%s; using CPU\n",
				caps.Name, humanBytes(int64(caps.MemoryBytes)), humanBytes(int64(need)))
			return nil, nil
		}
		return nil, fmt.Errorf("inference: %s has %s, need ~%s",
			caps.Name, humanBytes(int64(caps.MemoryBytes)), humanBytes(int64(need)))
	}
	logBackend(opt, "gpu: using %s (%s)\n", caps.Name, humanBytes(int64(caps.MemoryBytes)))
	return be, nil
}

func logBackend(opt LoadOptions, format string, args ...any) {
	if opt.Log == nil || opt.LogLevel == "quiet" {
		return
	}
	fmt.Fprintf(opt.Log, format, args...)
}

func modelBytes(g *gguf.File) uint64 {
	var n uint64
	for _, t := range g.Tensors {
		if sz, err := t.ByteSize(); err == nil {
			n += uint64(sz)
		}
	}
	return n
}

func humanBytes(b int64) string {
	f := float64(b)
	const unit = 1024.0
	switch {
	case f >= unit*unit*unit:
		return fmt.Sprintf("%.2f GiB", f/(unit*unit*unit))
	case f >= unit*unit:
		return fmt.Sprintf("%.2f MiB", f/(unit*unit))
	case f >= unit:
		return fmt.Sprintf("%.2f KiB", f/(unit))
	default:
		return fmt.Sprintf("%d B", b)
	}
}
