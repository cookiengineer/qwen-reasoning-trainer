// Package inference provides the small generation engine shared by the
// qwen-server command: model loading, backend selection, chat rendering,
// sampling, and streaming parsing.
package inference

import (
	"fmt"
	"io"
	"os"
	"sync"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute/vulkan"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/modelcfg"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
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
	// MaxConcurrent bounds the number of in-flight generation requests.
	MaxConcurrent int
	// KVCacheType is "q4_k" (default) or "q8_0".
	KVCacheType string
	// Log receives backend selection diagnostics. Nil discards them.
	Log io.Writer
	// LogLevel is "quiet", "info", or "debug"; quiet silences backend logging.
	LogLevel string
}

// Engine owns a loaded model and admits generation requests. Individual forward
// calls are serialized by the model/backend execution guard, so requests may
// interleave between generated tokens without corrupting shared Vulkan state.
type Engine struct {
	Model   *qwen38.Model
	Vocab   *tokenizer.Vocab
	Cfg     *modelcfg.Config
	Backend compute.Backend

	file         *gguf.File
	contextLen   int
	modelBytes   uint64
	kvType       quant.Type
	prefillChunk int
	imEnd        int32
	hasImEnd     bool

	kvBudget uint64
	kvLimit  bool
	kvUsed   uint64
	kvMu     sync.Mutex

	sem   chan struct{}
	close sync.Once

	// life protects the engine lifespan: Chat holds a read lock for its whole
	// request, Close takes the write lock before freeing model/backend state.
	life   sync.RWMutex
	closed bool
}

// Load opens a GGUF, loads its weights, and selects a compute backend.
func Load(opt LoadOptions) (*Engine, error) {
	if opt.Path == "" {
		return nil, fmt.Errorf("inference: model path is required")
	}
	kvType, err := ParseKVCacheType(opt.KVCacheType)
	if err != nil {
		return nil, err
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
	if os.Getenv("QWEN38_BLOCK_ATTENTION") == "1" {
		if vb, ok := be.(*vulkan.Backend); ok {
			vb.SetFlashAttention(false)
		}
	}

	mb := modelBytes(g)
	ctx := mc.ContextLength
	if opt.ContextLength > 0 {
		ctx = opt.ContextLength
	}
	if ctx <= 0 {
		ctx = 32768
	}
	native := mc.ContextLength
	if native <= 0 {
		native = ctx
	}
	mode, factor, _ := ResolveRopeScaling(ctx, native, "auto", 0)
	m.Cfg.SetRopeScaling(mode, factor, native)
	if be != nil {
		if err := m.EnableDeviceForward(); err != nil {
			be.Close()
			g.Close()
			return nil, err
		}
	}
	concurrent := opt.MaxConcurrent
	if concurrent <= 0 {
		concurrent = 1
	}
	var caps compute.Capabilities
	if be != nil {
		caps = be.Capabilities()
	}

	e := &Engine{
		Model:        m,
		Vocab:        vocab,
		Cfg:          mc,
		Backend:      be,
		file:         g,
		contextLen:   ctx,
		modelBytes:   mb,
		kvType:       kvType,
		prefillChunk: AutoPrefillChunk(ctx, caps),
		sem:          make(chan struct{}, concurrent),
	}
	if be != nil && caps.MemoryBytes > 0 {
		e.kvLimit = true
		if caps.MemoryBytes > mb+gpuHeadroomBytes {
			e.kvBudget = caps.MemoryBytes - mb - gpuHeadroomBytes
		}
	}
	if id, ok := vocab.SpecialID("<|im_end|>"); ok {
		e.imEnd, e.hasImEnd = id, true
	}
	return e, nil
}

// ContextLength returns the maximum number of positions the server will use.
func (e *Engine) ContextLength() int { return e.contextLen }

// ModelBytes returns the GGUF tensor bytes held by the model.
func (e *Engine) ModelBytes() uint64 { return e.modelBytes }

// KVCacheType returns the selected KV cache storage type.
func (e *Engine) KVCacheType() quant.Type { return e.kvType }

// EstimatedKVBytes returns the K+V cache size at context tokens.
func (e *Engine) EstimatedKVBytes(context int) uint64 {
	return EstimateKVBytes(e.Cfg, context, e.kvType)
}

// PrefillChunk returns the automatic prompt chunk size for this engine.
func (e *Engine) PrefillChunk() int { return e.prefillChunk }

// KVBudgetBytes returns the device budget available for KV caches. Zero with
// KVLimited()==false means no limit was derived from the backend.
func (e *Engine) KVBudgetBytes() uint64 { return e.kvBudget }

// KVLimited reports whether a device KV budget is enforced.
func (e *Engine) KVLimited() bool { return e.kvLimit }

func (e *Engine) reserveKV(n uint64) bool {
	if !e.kvLimit {
		return true
	}
	e.kvMu.Lock()
	defer e.kvMu.Unlock()
	if e.kvUsed+n > e.kvBudget {
		return false
	}
	e.kvUsed += n
	return true
}

func (e *Engine) releaseKV(n uint64) {
	if !e.kvLimit {
		return
	}
	e.kvMu.Lock()
	if e.kvUsed >= n {
		e.kvUsed -= n
	} else {
		e.kvUsed = 0
	}
	e.kvMu.Unlock()
}

// Close releases device weights, the backend, and the GGUF file.
func (e *Engine) Close() error {
	var err error
	e.close.Do(func() {
		e.life.Lock()
		e.closed = true
		if e.Model != nil {
			e.Model.FreeDevice()
		}
		if e.Backend != nil {
			e.Backend.Close()
		}
		if e.file != nil {
			err = e.file.Close()
		}
		e.life.Unlock()
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
