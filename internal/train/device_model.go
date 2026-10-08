package train

import (
	"fmt"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

// adapterGrad is one LoRA adapter's gradients (device-resident).
type adapterGrad struct {
	lin    *devLin
	dA, dB compute.Buffer
}

// deviceLayer wraps a resident full-attention or GatedDeltaNet block.
type deviceLayer struct {
	attn *DeviceAttention
	gdn  *DeviceGDN
}

func (l *deviceLayer) forward(x compute.Buffer, positions []int32) (y compute.Buffer, ctx any, err error) {
	if l.attn != nil {
		c, err := l.attn.Forward(x, positions)
		if err != nil {
			return nil, nil, err
		}
		return c.y, c, nil
	}
	c, err := l.gdn.Forward(x, positions)
	if err != nil {
		return nil, nil, err
	}
	return c.y, c, nil
}

func (l *deviceLayer) backward(ctx any, dY compute.Buffer, positions []int32) (compute.Buffer, []adapterGrad, error) {
	if l.attn != nil {
		g, err := l.attn.Backward(ctx.(*attentionCtx), dY, positions)
		if err != nil {
			return nil, nil, err
		}
		return g.DX, []adapterGrad{
			{&l.attn.q, g.DQ[0], g.DQ[1]}, {&l.attn.k, g.DK[0], g.DK[1]},
			{&l.attn.v, g.DV[0], g.DV[1]}, {&l.attn.o, g.DO[0], g.DO[1]},
			{&l.attn.fg, g.DFG[0], g.DFG[1]}, {&l.attn.fu, g.DFU[0], g.DFU[1]},
			{&l.attn.fd, g.DFD[0], g.DFD[1]},
		}, nil
	}
	g, err := l.gdn.Backward(ctx.(*gdnCtx), dY)
	if err != nil {
		return nil, nil, err
	}
	return g.DX, []adapterGrad{
		{&l.gdn.qkv, g.DQKV[0], g.DQKV[1]}, {&l.gdn.gate, g.DGate[0], g.DGate[1]},
		{&l.gdn.out, g.DOUT[0], g.DOUT[1]},
		{&l.gdn.fg, g.DFG[0], g.DFG[1]}, {&l.gdn.fu, g.DFU[0], g.DFU[1]},
		{&l.gdn.fd, g.DFD[0], g.DFD[1]},
	}, nil
}

// DeviceModel is a resident Qwen3.8 model: the full 3:1 linear:full layer stack
// with a frozen quantized base, LoRA adapters, and a final norm + LM head. Its
// forward, loss, and backward run entirely on the backend.
//
// The token embedding is kept quantized on the device and gathered with
// GetRowsWeight (dequantizing only the used rows). The LM head is frozen.
type DeviceModel struct {
	be         compute.Backend
	cfg        *qwen38.Config
	layers     []*deviceLayer
	embedW     compute.Buffer
	finalNormB compute.Buffer
	lmHeadB    compute.Buffer
	// Checkpoint, when set, stores only the layer inputs during Forward and
	// recomputes each layer during Backward, bounding activation memory.
	Checkpoint bool
}

// NewDeviceModel uploads the frozen weights and builds LoRA adapters.
func NewDeviceModel(be compute.Backend, cfg *qwen38.Config, w *qwen38.Weights, loCfg LoRAConfig, seed uint64) (*DeviceModel, error) {
	m := &DeviceModel{be: be, cfg: cfg}
	var err error
	if m.embedW, err = be.UploadWeight(w.TokenEmbd.Typ, w.TokenEmbd.Raw, w.TokenEmbd.Dims); err != nil {
		return nil, err
	}
	if m.finalNormB, err = uploadVec(be, w.OutputNorm); err != nil {
		return nil, err
	}
	if m.lmHeadB, err = be.UploadWeight(w.Output.Typ, w.Output.Raw, w.Output.Dims); err != nil {
		return nil, err
	}
	for il := 0; il < cfg.TrunkLayers(); il++ {
		lw := &w.Layers[il]
		var l *deviceLayer
		if cfg.IsRecurrent(il) {
			g, err := NewDeviceGDN(be, cfg, lw, loCfg, seed+uint64(il)*10)
			if err != nil {
				return nil, err
			}
			l = &deviceLayer{gdn: g}
		} else {
			a, err := NewDeviceAttention(be, cfg, lw, loCfg, seed+uint64(il)*10)
			if err != nil {
				return nil, err
			}
			l = &deviceLayer{attn: a}
		}
		names := targetWeights(lw, cfg.IsRecurrent(il), loCfg.Targets, il)
		lins := l.devLins()
		if len(names) != len(lins) {
			return nil, fmt.Errorf("train: device path requires the default LoRA target set: %d names for %d adapters", len(names), len(lins))
		}
		for i := range lins {
			lins[i].name = names[i].name
		}
		m.layers = append(m.layers, l)
	}
	return m, nil
}

// devLins returns every adapter layer in canonical order matching
// targetWeights: layer order, then the layer type's projection order.
func (l *deviceLayer) devLins() []*devLin {
	if l.attn != nil {
		a := l.attn
		return []*devLin{&a.q, &a.k, &a.v, &a.o, &a.fg, &a.fu, &a.fd}
	}
	g := l.gdn
	return []*devLin{&g.qkv, &g.gate, &g.out, &g.fg, &g.fu, &g.fd}
}

// AdapterLins returns every device adapter in order.
func (m *DeviceModel) AdapterLins() []*devLin {
	var out []*devLin
	for _, l := range m.layers {
		out = append(out, l.devLins()...)
	}
	return out
}

// Params returns the host LoRA tensors [A,B] per adapter in AdapterLins order,
// matching train.NewQwenModel.Params. They are only synchronized on demand by
// SyncAdapters.
func (m *DeviceModel) Params() []*compute.Tensor {
	var out []*compute.Tensor
	for _, l := range m.AdapterLins() {
		out = append(out, l.lora.A, l.lora.B)
	}
	return out
}

// ModelCtx holds the per-layer contexts (or inputs, when checkpointing) for the
// backward pass.
type ModelCtx struct {
	layerCtx   []any
	xs         []compute.Buffer
	xFinal     compute.Buffer
	positions  []int32
	checkpoint bool
}

// Forward runs the model on the given token ids and returns the logits [V, T].
func (m *DeviceModel) Forward(tokens []int32) (compute.Buffer, *ModelCtx, error) {
	T := len(tokens)
	x, err := m.be.GetRowsWeight(m.embedW, tokens)
	if err != nil {
		return nil, nil, err
	}
	positions := make([]int32, T)
	for i := range positions {
		positions[i] = int32(i)
	}
	ctx := &ModelCtx{positions: positions, checkpoint: m.Checkpoint}
	for _, l := range m.layers {
		if m.Checkpoint {
			// Keep only the layer input; the recompute happens in Backward.
			// The scope reclaims this layer's forward temporaries immediately.
			ctx.xs = append(ctx.xs, x)
			m.be.BeginScope()
			y, _, err := l.forward(x, positions)
			if err != nil {
				m.be.EndScope()
				return nil, nil, err
			}
			m.be.EndScope(y)
			// Flush so the layer's temporaries are reclaimed before the next
			// layer allocates, bounding forward activation memory.
			if err := m.be.Sync(); err != nil {
				return nil, nil, err
			}
			x = y
			continue
		}
		y, lc, err := l.forward(x, positions)
		if err != nil {
			return nil, nil, err
		}
		ctx.layerCtx = append(ctx.layerCtx, lc)
		x = y
	}
	ctx.xFinal = x
	xn, err := m.be.RMSNorm(x, m.finalNormB, m.cfg.RmsEps)
	if err != nil {
		return nil, nil, err
	}
	logits, err := m.be.MatMulWeight(m.lmHeadB, xn)
	if err != nil {
		return nil, nil, err
	}
	return logits, ctx, nil
}

// Loss computes the mean cross-entropy loss for logits against targets
// (ignoreIndex entries skipped) and returns the loss value plus dLogits.
func (m *DeviceModel) Loss(logits compute.Buffer, targets []int32, ignoreIndex int) (float32, compute.Buffer, error) {
	lb, dl, err := m.be.CrossEntropy(logits, targets, ignoreIndex)
	if err != nil {
		return 0, nil, err
	}
	lt, err := m.be.Download(lb)
	if err != nil {
		return 0, nil, err
	}
	return lt.F32[0], dl, nil
}

// Backward propagates dLogits [V, T] through the model and returns the adapter
// gradients for every layer. With Checkpoint set, each layer is recomputed from
// its stored input.
func (m *DeviceModel) Backward(ctx *ModelCtx, dLogits compute.Buffer) ([]adapterGrad, error) {
	if ctx.checkpoint {
		return m.backwardCheckpoint(ctx, dLogits)
	}
	dxn, err := m.be.MatMulWeightTranspose(m.lmHeadB, dLogits)
	if err != nil {
		return nil, err
	}
	dX, err := m.be.RMSNormBack(ctx.xFinal, m.finalNormB, dxn, m.cfg.RmsEps)
	if err != nil {
		return nil, err
	}
	m.be.Free(dxn)
	m.be.Free(ctx.xFinal)
	var all []adapterGrad
	for i := len(m.layers) - 1; i >= 0; i-- {
		m.be.BeginScope()
		dx, ag, err := m.layers[i].backward(ctx.layerCtx[i], dX, ctx.positions)
		if err != nil {
			m.be.EndScope()
			return nil, err
		}
		keep := make([]compute.Buffer, 0, 1+2*len(ag))
		keep = append(keep, dx)
		for _, g := range ag {
			keep = append(keep, g.dA, g.dB)
		}
		m.be.EndScope(keep...)
		m.be.Sync()
		all = append(all, ag...)
		dX = dx
	}
	return all, nil
}

// backwardCheckpoint recomputes each layer from its stored input under an
// allocation scope, so only one layer's activations are live at a time. The
// adapter gradients and the running input gradient are retained.
func (m *DeviceModel) backwardCheckpoint(ctx *ModelCtx, dLogits compute.Buffer) ([]adapterGrad, error) {
	dxn, err := m.be.MatMulWeightTranspose(m.lmHeadB, dLogits)
	if err != nil {
		return nil, err
	}
	dX, err := m.be.RMSNormBack(ctx.xFinal, m.finalNormB, dxn, m.cfg.RmsEps)
	if err != nil {
		return nil, err
	}
	m.be.Free(dxn)
	m.be.Free(ctx.xFinal)

	var all []adapterGrad
	for i := len(m.layers) - 1; i >= 0; i-- {
		prev := dX
		m.be.BeginScope()
		_, lc, err := m.layers[i].forward(ctx.xs[i], ctx.positions)
		if err != nil {
			m.be.EndScope()
			return nil, err
		}
		dx, ag, err := m.layers[i].backward(lc, dX, ctx.positions)
		if err != nil {
			m.be.EndScope()
			return nil, err
		}
		keep := make([]compute.Buffer, 0, 1+2*len(ag))
		keep = append(keep, dx)
		for _, g := range ag {
			keep = append(keep, g.dA, g.dB)
		}
		m.be.EndScope(keep...)
		if prev != dx {
			m.be.Free(prev)
		}
		m.be.Free(ctx.xs[i])
		// Flush so the scope's frees land and the buffer pool can recycle.
		if err := m.be.Sync(); err != nil {
			return nil, err
		}
		all = append(all, ag...)
		dX = dx
	}
	return all, nil
}

// SyncAdapters downloads every device adapter into its host LoRA tensors so
// Params, Overrides and checkpointing observe the trained values.
func (m *DeviceModel) SyncAdapters() error {
	for _, l := range m.AdapterLins() {
		a, err := m.be.Download(l.a)
		if err != nil {
			return err
		}
		copy(l.lora.A.F32, a.F32)
		b, err := m.be.Download(l.b)
		if err != nil {
			return err
		}
		copy(l.lora.B.F32, b.F32)
	}
	return nil
}

// Overrides merges every adapter into its frozen base weight and returns the
// GGUF tensor overrides keyed by name, mirroring QwenModel.Overrides.
func (m *DeviceModel) Overrides() (map[string]qwen38.Override, error) {
	if err := m.SyncAdapters(); err != nil {
		return nil, err
	}
	out := make(map[string]qwen38.Override, len(m.layers)*7)
	for _, l := range m.AdapterLins() {
		t, raw, err := l.lora.MergeToQuant(l.w.Typ, l.w.Raw, l.w.Dims)
		if err != nil {
			return nil, err
		}
		out[l.name] = qwen38.Override{Type: t, Raw: raw}
	}
	return out, nil
}

// WriteMerged streams a copy of the base GGUF to outPath with all adapters
// merged and requantized.
func (m *DeviceModel) WriteMerged(base *gguf.File, outPath string) error {
	ov, err := m.Overrides()
	if err != nil {
		return err
	}
	return qwen38.RewriteGGUF(base, ov, outPath)
}

// release frees the activation buffers retained by a checkpointed forward.
func (c *ModelCtx) release(be compute.Backend) {
	for _, b := range c.xs {
		be.Free(b)
	}
	if c.xFinal != nil {
		be.Free(c.xFinal)
	}
}
