package qwen38

import (
	"fmt"
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// State carries the recurrent and KV-cache state between forward calls. A
// fresh (zeroed) state is used for the first call.
type State struct {
	// Conv holds the causal-conv history per layer: (DConv-1) rows of ConvDim.
	Conv [][]float32
	// SSM holds the GatedDeltaNet state per layer: HeadVDim*HeadVDim*DtRank.
	SSM [][]float32
	// KV holds the host key/value cache of each full-attention layer.
	KV []*kvCache
	// DeviceKV holds the quantized device cache when enabled. Recurrent layers
	// have a nil entry.
	DeviceKV []*DeviceKVCache
	// DeviceConv and DeviceSSM hold recurrent-layer state when device forward
	// is enabled. Full-attention layers have nil entries.
	DeviceConv []compute.Buffer
	DeviceSSM  []compute.Buffer

	// MTPKV is the host KV cache for the single NextN block.
	MTPKV *kvCache
	// DeviceMTPKV is the quantized device KV cache for the NextN block.
	DeviceMTPKV *DeviceKVCache
	// MTPPrevH / DeviceMTPPrevH hold the hidden row of the last processed
	// token for the NextN draft head.
	MTPPrevH       []float32
	DeviceMTPPrevH compute.Buffer

	backend       compute.Backend
	kvBackend     KVCacheBackend
	kvType        quant.Type
	deviceForward bool

	specActive       bool
	specKVLen        []int
	specDeviceMTPLen int
	specSSM          []compute.Buffer
	specConv         []compute.Buffer
	specHostKVLen    []int
	specHostMTPLen   int
	specHostSSM      [][]float32
	specHostConv     [][]float32
}

// kvCache stores the keys and values of a full-attention layer, laid out as
// [headDim, nHeadKV, length].
type kvCache struct {
	k, v   []float32
	length int
}

// KVCacheBackend is implemented by backends that can own a quantized KV cache
// and run attention directly over it.
type KVCacheBackend interface {
	compute.Backend
	AllocKV(headDim, nKVHeads, capacity int, typ quant.Type) (compute.Buffer, error)
	WriteKV(cache compute.Buffer, byteOffset int, data []byte) error
	AppendKV(k, v, kcache, vcache compute.Buffer, pos int) error
	AttentionCached(q, kcache, vcache compute.Buffer, nHead, nHeadKV, nKV int, scale float32) (compute.Buffer, error)
}

// DeviceKVCache is one full-attention layer's quantized K/V cache.
type DeviceKVCache struct {
	K, V compute.Buffer
	Len  int
	Cap  int
}

// StateOptions configures optional device-resident KV state.
type StateOptions struct {
	Backend       compute.Backend
	KVType        quant.Type
	Capacity      int
	DeviceForward bool
	// MTP reserves a device KV cache for the NextN draft block.
	MTP bool
}

// NewState allocates a zeroed state for cfg.
func NewState(cfg *Config) *State { return NewStateWithCapacity(cfg, 0) }

// NewStateWithOptions allocates a state and, when the backend supports it,
// reserves quantized device KV buffers for the full-attention layers.
func NewStateWithOptions(cfg *Config, opt StateOptions) (*State, error) {
	s := NewStateWithCapacity(cfg, 0)
	s.backend = opt.Backend
	if opt.DeviceForward && opt.Backend != nil {
		s.deviceForward = true
		s.DeviceConv = make([]compute.Buffer, cfg.TrunkLayers())
		s.DeviceSSM = make([]compute.Buffer, cfg.TrunkLayers())
		for il := 0; il < cfg.TrunkLayers(); il++ {
			if !cfg.IsRecurrent(il) {
				continue
			}
			conv, err := uploadDeviceZeros(opt.Backend, cfg.DConv-1, cfg.ConvDim())
			if err != nil {
				s.Free()
				return nil, err
			}
			ssm, err := uploadDeviceZeros(opt.Backend, cfg.HeadVDim(), cfg.HeadVDim(), cfg.DtRank)
			if err != nil {
				opt.Backend.Free(conv)
				s.Free()
				return nil, err
			}
			s.DeviceConv[il] = conv
			s.DeviceSSM[il] = ssm
		}
	}
	kb, ok := opt.Backend.(KVCacheBackend)
	if !ok || opt.Capacity <= 0 || !opt.KVType.IsQuantized() {
		return s, nil
	}
	s.kvBackend = kb
	s.kvType = opt.KVType
	s.DeviceKV = make([]*DeviceKVCache, cfg.TrunkLayers())
	for il := 0; il < cfg.TrunkLayers(); il++ {
		if cfg.IsRecurrent(il) {
			continue
		}
		k, err := kb.AllocKV(cfg.HeadDim, cfg.NHeadKV, opt.Capacity, opt.KVType)
		if err != nil {
			s.Free()
			return nil, err
		}
		v, err := kb.AllocKV(cfg.HeadDim, cfg.NHeadKV, opt.Capacity, opt.KVType)
		if err != nil {
			opt.Backend.Free(k)
			s.Free()
			return nil, err
		}
		s.DeviceKV[il] = &DeviceKVCache{K: k, V: v, Cap: opt.Capacity}
	}
	if opt.MTP {
		k, err := kb.AllocKV(cfg.HeadDim, cfg.NHeadKV, opt.Capacity, opt.KVType)
		if err != nil {
			s.Free()
			return nil, err
		}
		v, err := kb.AllocKV(cfg.HeadDim, cfg.NHeadKV, opt.Capacity, opt.KVType)
		if err != nil {
			kb.Free(k)
			s.Free()
			return nil, err
		}
		s.DeviceMTPKV = &DeviceKVCache{K: k, V: v, Cap: opt.Capacity}
		prev, err := opt.Backend.Upload(&compute.Tensor{Dims: []int{cfg.NEmbd}, F32: make([]float32, cfg.NEmbd)})
		if err != nil {
			s.Free()
			return nil, err
		}
		s.DeviceMTPPrevH = prev
	}
	return s, nil
}

func uploadDeviceZeros(be compute.Backend, dims ...int) (compute.Buffer, error) {
	n := 1
	for _, d := range dims {
		n *= d
	}
	return be.Upload(&compute.Tensor{Dims: append([]int(nil), dims...), F32: make([]float32, n)})
}

// Free releases device KV buffers. The host caches are garbage-collected.
func (s *State) Free() {
	if s == nil {
		return
	}
	if s.kvBackend != nil {
		for _, c := range s.DeviceKV {
			if c == nil {
				continue
			}
			if c.K != nil {
				s.kvBackend.Free(c.K)
				c.K = nil
			}
			if c.V != nil {
				s.kvBackend.Free(c.V)
				c.V = nil
			}
		}
		s.DeviceKV = nil
	}
	if s.kvBackend != nil && s.DeviceMTPKV != nil {
		if s.DeviceMTPKV.K != nil {
			s.kvBackend.Free(s.DeviceMTPKV.K)
		}
		if s.DeviceMTPKV.V != nil {
			s.kvBackend.Free(s.DeviceMTPKV.V)
		}
		s.DeviceMTPKV = nil
	}
	if s.backend != nil && s.DeviceMTPPrevH != nil {
		s.backend.Free(s.DeviceMTPPrevH)
		s.DeviceMTPPrevH = nil
	}
	if s.backend != nil {
		for _, b := range s.DeviceConv {
			if b != nil {
				s.backend.Free(b)
			}
		}
		for _, b := range s.DeviceSSM {
			if b != nil {
				s.backend.Free(b)
			}
		}
	}
	s.freeSpec()
	s.DeviceConv = nil
	s.DeviceSSM = nil
	s.backend = nil
	s.kvBackend = nil
}

// NewStateWithCapacity allocates a zeroed state, optionally reserving room for
// capacity KV positions per full-attention layer. A capacity of zero leaves the
// cache lazy; appendKVCache grows geometrically from there.
func NewStateWithCapacity(cfg *Config, capacity int) *State {
	s := &State{
		Conv: make([][]float32, cfg.TrunkLayers()),
		SSM:  make([][]float32, cfg.TrunkLayers()),
		KV:   make([]*kvCache, cfg.TrunkLayers()),
	}
	if capacity < 0 {
		capacity = 0
	}
	stride := cfg.HeadDim * cfg.NHeadKV
	for il := 0; il < cfg.TrunkLayers(); il++ {
		if cfg.IsRecurrent(il) {
			s.Conv[il] = make([]float32, (cfg.DConv-1)*cfg.ConvDim())
			s.SSM[il] = make([]float32, cfg.HeadVDim()*cfg.HeadVDim()*cfg.DtRank)
			continue
		}
		c := &kvCache{}
		if capacity > 0 && stride > 0 {
			elems := capacity * stride
			c.k = make([]float32, 0, elems)
			c.v = make([]float32, 0, elems)
		}
		s.KV[il] = c
	}
	return s
}

// ForwardOptions controls what the forward pass returns.
type ForwardOptions struct {
	// RecordHidden captures the residual stream after every block.
	RecordHidden bool
	// State is the recurrent state to use; nil starts from zero and does not
	// update caller state.
	State *State
	// Positions overrides the default 0..T-1 positions.
	Positions []int32
	// LastTokenOnly computes logits only for the last token, which is all that
	// is needed for generation and avoids a full-vocabulary projection.
	LastTokenOnly bool
	// CaptureHidden returns the post-final-norm hidden state for every token.
	// The MTP draft head consumes these rows.
	CaptureHidden bool
}

// ForwardResult is the output of a forward pass.
type ForwardResult struct {
	// Logits has Dims [n_vocab, n_tokens].
	Logits *compute.Tensor
	// Hidden, when requested, holds TrunkLayers+1 residual-stream tensors:
	// index 0 is the embedding output, index k+1 the output of block k.
	Hidden []*compute.Tensor
	// FinalHidden is the post-final-norm hidden state [NEmbd, T] when
	// ForwardOptions.CaptureHidden is set (host path).
	FinalHidden *compute.Tensor
	// DeviceHidden is the post-final-norm hidden state on the device when
	// CaptureHidden is set. The caller owns and must free it.
	DeviceHidden compute.Buffer
	// State is the recurrent state after the pass.
	State *State
}

// Forward runs the model on the given token ids. Complete forward passes are
// serialized: the model mutex protects the lazy device-weight cache, and the
// backend execution guard protects shared Vulkan command/descriptor state.
func (m *Model) Forward(tokens []int32, opts ForwardOptions) (*ForwardResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	var res *ForwardResult
	err := compute.Run(m.Backend, func() error {
		var r *ForwardResult
		var err error
		if m.device != nil && opts.State != nil && opts.State.deviceForward {
			r, err = m.forwardDevice(tokens, opts)
		} else {
			r, err = m.forward(tokens, opts)
		}
		res = r
		return err
	})
	return res, err
}

func (m *Model) forward(tokens []int32, opts ForwardOptions) (*ForwardResult, error) {
	cfg := m.Cfg
	T := len(tokens)
	if T == 0 {
		return nil, fmt.Errorf("qwen38: empty prompt")
	}
	x, err := m.W.TokenEmbd.GetRows(tokens)
	if err != nil {
		return nil, err
	}
	res := &ForwardResult{}
	if opts.RecordHidden {
		res.Hidden = append(res.Hidden, x.Clone())
	}

	st := opts.State
	if st == nil {
		st = NewState(cfg)
	}

	positions := opts.Positions
	if positions == nil {
		positions = make([]int32, T)
		for i := range positions {
			positions[i] = int32(i)
		}
	}
	if len(positions) != T {
		return nil, fmt.Errorf("qwen38: positions length %d != tokens %d", len(positions), T)
	}

	for il := 0; il < cfg.TrunkLayers(); il++ {
		lw := &m.W.Layers[il]

		h := rms(x, lw.AttnNorm, cfg.RmsEps)
		var attnOut *compute.Tensor
		if cfg.IsRecurrent(il) {
			attnOut, err = m.linearAttn(lw, h, st, il)
		} else {
			attnOut, err = m.fullAttn(lw, h, positions, st, il)
		}
		if err != nil {
			return nil, fmt.Errorf("qwen38: layer %d attention: %w", il, err)
		}
		x, err = add(x, attnOut)
		if err != nil {
			return nil, err
		}

		h2 := rms(x, lw.PostAttnNorm, cfg.RmsEps)
		ffnOut, err := m.ffn(lw, h2)
		if err != nil {
			return nil, fmt.Errorf("qwen38: layer %d ffn: %w", il, err)
		}
		x, err = add(x, ffnOut)
		if err != nil {
			return nil, err
		}
		if opts.RecordHidden {
			res.Hidden = append(res.Hidden, x.Clone())
		}
	}

	xn := rms(x, m.W.OutputNorm, cfg.RmsEps)
	if opts.CaptureHidden {
		res.FinalHidden = xn
	}
	if opts.LastTokenOnly && T > 1 {
		xn = &compute.Tensor{
			Dims: []int{cfg.NEmbd, 1},
			Type: xn.Type,
			F32:  xn.F32[(T-1)*cfg.NEmbd : T*cfg.NEmbd],
		}
	}
	logits, err := m.mm(m.W.Output, xn)
	if err != nil {
		return nil, err
	}
	res.Logits = logits
	res.State = st
	return res, nil
}

// ffn applies the SwiGLU feed-forward network to a normalized activation.
func (m *Model) ffn(lw *LayerWeights, h *compute.Tensor) (*compute.Tensor, error) {
	gate, err := m.mm(lw.FfnGate, h)
	if err != nil {
		return nil, err
	}
	up, err := m.mm(lw.FfnUp, h)
	if err != nil {
		return nil, err
	}
	act := siluInto(gate)
	act, err = compute.Mul(act, up)
	if err != nil {
		return nil, err
	}
	return m.mm(lw.FfnDown, act)
}

// fullAttn runs a full-attention block on the normalized activation h, using
// and updating the layer's KV cache.
func (m *Model) fullAttn(lw *LayerWeights, h *compute.Tensor, positions []int32, st *State, il int) (*compute.Tensor, error) {
	var cache *kvCache
	if il >= 0 && il < len(st.KV) {
		cache = st.KV[il]
	}
	return m.fullAttnCache(lw, h, positions, st, cache, il)
}

func (m *Model) fullAttnCache(lw *LayerWeights, h *compute.Tensor, positions []int32, st *State, cache *kvCache, il int) (*compute.Tensor, error) {
	cfg := m.Cfg
	hd := cfg.HeadDim
	T := h.Ne(1)

	qg, err := m.mm(lw.AttnQ, h)
	if err != nil {
		return nil, err
	}
	kcur, err := m.mm(lw.AttnK, h)
	if err != nil {
		return nil, err
	}
	vcur, err := m.mm(lw.AttnV, h)
	if err != nil {
		return nil, err
	}

	// Split the fused query+gate projection. qg has Dims
	// [2*n_head*headDim, T]; each head contributes headDim query values then
	// headDim gate values.
	nHead := cfg.NHead
	qgN := 2 * hd
	q := compute.NewF32(hd, nHead, T)
	gate := compute.NewF32(nHead*hd, T)
	for t := 0; t < T; t++ {
		for hh := 0; hh < nHead; hh++ {
			base := t*(nHead*qgN) + hh*qgN
			for d := 0; d < hd; d++ {
				q.F32[d+hh*hd+t*(nHead*hd)] = qg.F32[base+d]
				gate.F32[hh*hd+d+t*(nHead*hd)] = qg.F32[base+hd+d]
			}
		}
	}

	qn := rms(q, lw.AttnQNorm, cfg.RmsEps)
	kn, err := reshape3(kcur, hd, cfg.NHeadKV, T)
	if err != nil {
		return nil, err
	}
	kn = rms(kn, lw.AttnKNorm, cfg.RmsEps)
	vn, err := reshape3(vcur, hd, cfg.NHeadKV, T)
	if err != nil {
		return nil, err
	}

	rope := cfg.RopeScalingOptions()
	qn, err = compute.RoPENeoXScaled(qn, positions, cfg.RopeTheta, cfg.NRot, rope)
	if err != nil {
		return nil, err
	}
	kn, err = compute.RoPENeoXScaled(kn, positions, cfg.RopeTheta, cfg.NRot, rope)
	if err != nil {
		return nil, err
	}

	scale := float32(1.0)
	if hd > 0 {
		scale = float32(1.0 / math.Sqrt(float64(hd)))
	}
	if cfg.AttnScale > 0 {
		scale *= cfg.AttnScale
	}
	var attn *compute.Tensor
	if dc := deviceCache(st, il); dc != nil {
		if err := m.appendDeviceKV(dc, st.kvType, kn, vn); err != nil {
			return nil, err
		}
		attn, err = m.attentionCached(qn, dc, nHead, cfg.NHeadKV, scale)
		if err != nil {
			return nil, err
		}
	} else {
		// Append to the host KV cache and attend over the full history.
		kc, vc, total := appendKVCache(cache, kn, vn, hd, cfg.NHeadKV, T)
		causal := T == total
		attn, err = compute.Attention(qn, kc, vc, nHead, cfg.NHeadKV, scale, causal)
		if err != nil {
			return nil, err
		}
	}

	gateSig := sigmoidInto(gate)
	attn, err = compute.Mul(attn, gateSig)
	if err != nil {
		return nil, err
	}
	// Attention output is [headDim, nHead, T]; the layout is identical to
	// [nHead*headDim, T], which is what the output projection expects.
	attnFlat := &compute.Tensor{Dims: []int{nHead * hd, T}, Type: attn.Type, F32: attn.F32}
	return m.mm(lw.AttnOutput, attnFlat)
}

// linearAttn runs a GatedDeltaNet block on the normalized activation h.
func (m *Model) linearAttn(lw *LayerWeights, h *compute.Tensor, st *State, il int) (*compute.Tensor, error) {
	cfg := m.Cfg
	T := h.Ne(1)
	headVDim := cfg.HeadVDim()
	nV := cfg.DtRank
	nK := cfg.GroupCount
	headKDim := cfg.DState

	qkv, err := m.mm(lw.AttnQKV, h) // [convDim, T]
	if err != nil {
		return nil, err
	}
	z, err := m.mm(lw.AttnGate, h) // [valueDim, T]
	if err != nil {
		return nil, err
	}

	betaRaw, err := m.mm(lw.SSMBeta, h) // [dtRank, T]
	if err != nil {
		return nil, err
	}
	beta := sigmoidInto(betaRaw)

	alpha, err := m.mm(lw.SSMAlpha, h) // [dtRank, T]
	if err != nil {
		return nil, err
	}
	gate := compute.NewF32(1, nV, T)
	for t := 0; t < T; t++ {
		for hh := 0; hh < nV; hh++ {
			a := alpha.F32[hh+t*nV] + lw.SSMDt[hh]
			gate.F32[hh+t*nV] = softplus(a) * lw.SSMA[hh]
		}
	}
	betaT := compute.NewF32(1, nV, T)
	copy(betaT.F32, beta.F32)

	// Causal conv over the concatenated qkv channels.
	convDim := cfg.ConvDim()
	ncs := cfg.DConv - 1 + T
	convInput := compute.NewF32(ncs, convDim, 1)
	hist := st.Conv[il]
	for row := 0; row < cfg.DConv-1; row++ {
		for ch := 0; ch < convDim; ch++ {
			convInput.F32[row+ch*ncs] = hist[row*convDim+ch]
		}
	}
	for t := 0; t < T; t++ {
		for ch := 0; ch < convDim; ch++ {
			convInput.F32[(cfg.DConv-1+t)+ch*ncs] = qkv.F32[ch+t*convDim]
		}
	}
	convOut, err := compute.SSMConv(convInput, lw.SSMConv1d)
	if err != nil {
		return nil, err
	}
	convOut = siluInto(convOut) // [convDim, T]

	// Split q, k, v and reshape to head layouts.
	keyDim := cfg.KeyDim()
	q, err := gatherHeads(convOut, 0, headKDim, nK, T, convDim)
	if err != nil {
		return nil, err
	}
	k, err := gatherHeads(convOut, keyDim, headKDim, nK, T, convDim)
	if err != nil {
		return nil, err
	}
	v, err := gatherHeads(convOut, 2*keyDim, headVDim, nV, T, convDim)
	if err != nil {
		return nil, err
	}
	q = compute.L2Norm(q, 1e-6)
	k = compute.L2Norm(k, 1e-6)

	// Repeat q/k heads from nK to nV (v head j uses q/k head j%nK).
	q48 := repeatHeads(q, headKDim, nK, nV, T)
	k48 := repeatHeads(k, headKDim, nK, nV, T)

	state := &compute.Tensor{Dims: []int{headVDim, headVDim, nV}, F32: st.SSM[il]}
	out, newState, err := compute.GatedDeltaNet(q48, k48, v, gate, betaT, state)
	if err != nil {
		return nil, err
	}
	copy(st.SSM[il], newState.F32)

	// Update the conv history: the last DConv-1 rows of convInput.
	for row := 0; row < cfg.DConv-1; row++ {
		srcRow := ncs - (cfg.DConv - 1) + row
		for ch := 0; ch < convDim; ch++ {
			hist[row*convDim+ch] = convInput.F32[srcRow+ch*ncs]
		}
	}

	// Gated normalization: norm over head dim, times silu(z).
	norm := rms(out, lw.SSMNorm, cfg.RmsEps) // [headVDim, nV, T]
	zr := &compute.Tensor{Dims: []int{headVDim, nV, T}, F32: z.F32}
	gated, err := compute.Mul(norm, siluInto(zr))
	if err != nil {
		return nil, err
	}
	flat := &compute.Tensor{Dims: []int{cfg.ValueDim(), T}, F32: gated.F32}
	return m.mm(lw.SSMOut, flat)
}

// reshape3 views a [d*h, T] tensor as [d, h, T]. Both use GGML layout with the
// feature dimension leading, so no data movement is needed.
func reshape3(x *compute.Tensor, d, h, t int) (*compute.Tensor, error) {
	if x.NumElements() != d*h*t {
		return nil, fmt.Errorf("qwen38: reshape %v to [%d %d %d]", x.Dims, d, h, t)
	}
	return &compute.Tensor{Dims: []int{d, h, t}, Type: x.Type, F32: x.F32}, nil
}

// gatherHeads extracts head-structured channels from a conv output of Dims
// [convDim, T], producing [headDim, nHead, T]. Channel c = head*headDim + d.
func gatherHeads(convOut *compute.Tensor, offset, headDim, nHead, T, convDim int) (*compute.Tensor, error) {
	out := compute.NewF32(headDim, nHead, T)
	for t := 0; t < T; t++ {
		for hh := 0; hh < nHead; hh++ {
			base := offset + hh*headDim
			for d := 0; d < headDim; d++ {
				out.F32[d+hh*headDim+t*(nHead*headDim)] = convOut.F32[(base+d)+t*convDim]
			}
		}
	}
	return out, nil
}

// repeatHeads repeats q/k heads from nIn to nOut such that output head j takes
// input head j%nIn.
func repeatHeads(x *compute.Tensor, headDim, nIn, nOut, T int) *compute.Tensor {
	out := compute.NewF32(headDim, nOut, T)
	for t := 0; t < T; t++ {
		for hh := 0; hh < nOut; hh++ {
			src := hh % nIn
			for d := 0; d < headDim; d++ {
				out.F32[d+hh*headDim+t*(nOut*headDim)] = x.F32[d+src*headDim+t*(nIn*headDim)]
			}
		}
	}
	return out
}

func deviceCache(st *State, il int) *DeviceKVCache {
	if st == nil || il < 0 || il >= len(st.DeviceKV) {
		return nil
	}
	return st.DeviceKV[il]
}

func (m *Model) appendDeviceKV(dc *DeviceKVCache, typ quant.Type, k, v *compute.Tensor) error {
	if dc == nil {
		return fmt.Errorf("qwen38: nil device KV cache")
	}
	T := k.Ne(2)
	if T <= 0 {
		return nil
	}
	if dc.Len+T > dc.Cap {
		return fmt.Errorf("qwen38: KV cache capacity %d exceeded (%d + %d)", dc.Cap, dc.Len, T)
	}
	kb, ok := m.Backend.(KVCacheBackend)
	if !ok {
		return fmt.Errorf("qwen38: backend does not support device KV")
	}
	kraw, err := quant.QuantizeParallel(typ, k.F32)
	if err != nil {
		return err
	}
	vraw, err := quant.QuantizeParallel(typ, v.F32)
	if err != nil {
		return err
	}
	if len(kraw) == 0 || len(kraw)%T != 0 || len(vraw) != len(kraw) {
		return fmt.Errorf("qwen38: invalid quantized KV size")
	}
	nHeadKV := k.Ne(1)
	if nHeadKV <= 0 || len(kraw)%(T*nHeadKV) != 0 {
		return fmt.Errorf("qwen38: invalid quantized KV shape")
	}
	// Host-quantized bytes are token-major; the cache is head-major.
	perHeadToken := len(kraw) / (T * nHeadKV)
	for t := 0; t < T; t++ {
		for h := 0; h < nHeadKV; h++ {
			src := (t*nHeadKV + h) * perHeadToken
			dst := ((h*dc.Cap + dc.Len + t) * perHeadToken)
			if err := kb.WriteKV(dc.K, dst, kraw[src:src+perHeadToken]); err != nil {
				return err
			}
			if err := kb.WriteKV(dc.V, dst, vraw[src:src+perHeadToken]); err != nil {
				return err
			}
		}
	}
	dc.Len += T
	return nil
}

func (m *Model) attentionCached(q *compute.Tensor, dc *DeviceKVCache, nHead, nHeadKV int, scale float32) (*compute.Tensor, error) {
	kb, ok := m.Backend.(KVCacheBackend)
	if !ok {
		return nil, fmt.Errorf("qwen38: backend does not support cached attention")
	}
	qb, err := m.Backend.Upload(q)
	if err != nil {
		return nil, err
	}
	defer m.Backend.Free(qb)
	ob, err := kb.AttentionCached(qb, dc.K, dc.V, nHead, nHeadKV, dc.Len, scale)
	if err != nil {
		return nil, err
	}
	defer m.Backend.Free(ob)
	return m.Backend.Download(ob)
}

// reserve grows the backing arrays to hold need positions, preserving existing
// values. It reuses the existing allocation when capacity is sufficient and
// otherwise grows geometrically, which amortizes copying across a generation.
// SetLength rewinds the logical cache length. Cached bytes beyond n are
// ignored and may be overwritten by the next append.
func (c *kvCache) SetLength(n int) {
	if c == nil {
		return
	}
	if n < 0 {
		n = 0
	}
	c.length = n
}

func (c *kvCache) reserve(stride, need int) {
	needElems := need * stride
	if cap(c.k) >= needElems && cap(c.v) >= needElems {
		c.k = c.k[:needElems]
		c.v = c.v[:needElems]
		return
	}
	newCap := needElems
	if cap(c.k) > 0 && cap(c.k)*2 > newCap {
		newCap = cap(c.k) * 2
	}
	if cap(c.v) > 0 && cap(c.v)*2 > newCap {
		newCap = cap(c.v) * 2
	}
	nk := make([]float32, needElems, newCap)
	nv := make([]float32, needElems, newCap)
	copy(nk, c.k)
	copy(nv, c.v)
	c.k, c.v = nk, nv
}

// appendKVCache appends T keys/values to cache and returns the combined cache.
func appendKVCache(cache *kvCache, k, v *compute.Tensor, hd, nKV, T int) (*compute.Tensor, *compute.Tensor, int) {
	stride := hd * nKV
	total := cache.length + T
	cache.reserve(stride, total)
	old := cache.length * stride
	need := total * stride
	copy(cache.k[old:need], k.F32[:T*stride])
	copy(cache.v[old:need], v.F32[:T*stride])
	cache.length = total
	nk := &compute.Tensor{Dims: []int{hd, nKV, total}, Type: k.Type, F32: cache.k}
	nv := &compute.Tensor{Dims: []int{hd, nKV, total}, Type: v.Type, F32: cache.v}
	return nk, nv, total
}

// SpecSnapshot captures the recurrent state before a speculative verification
// batch. KV data is append-only, so only lengths are saved; GDN and conv state
// are copied because their buffers are overwritten in place.
func (s *State) SpecSnapshot() error {
	if s == nil || s.specActive {
		return nil
	}
	s.specHostKVLen = s.specHostKVLen[:0]
	for _, c := range s.KV {
		n := 0
		if c != nil {
			n = c.length
		}
		s.specHostKVLen = append(s.specHostKVLen, n)
	}
	s.specHostMTPLen = 0
	if s.MTPKV != nil {
		s.specHostMTPLen = s.MTPKV.length
	}
	if s.backend == nil {
		s.specHostSSM = s.specHostSSM[:0]
		for _, v := range s.SSM {
			s.specHostSSM = append(s.specHostSSM, append([]float32(nil), v...))
		}
		s.specHostConv = s.specHostConv[:0]
		for _, v := range s.Conv {
			s.specHostConv = append(s.specHostConv, append([]float32(nil), v...))
		}
		s.specActive = true
		return nil
	}
	s.specKVLen = s.specKVLen[:0]
	for _, c := range s.DeviceKV {
		n := 0
		if c != nil {
			n = c.Len
		}
		s.specKVLen = append(s.specKVLen, n)
	}
	if s.DeviceMTPKV != nil {
		s.specDeviceMTPLen = s.DeviceMTPKV.Len
	}
	for len(s.specSSM) < len(s.DeviceSSM) {
		s.specSSM = append(s.specSSM, nil)
		s.specConv = append(s.specConv, nil)
	}
	for il, src := range s.DeviceSSM {
		if src == nil {
			continue
		}
		dst := s.specSSM[il]
		if dst == nil {
			var err error
			dst, err = s.backend.Alloc(src.Dims(), quant.TypeF32)
			if err != nil {
				return err
			}
			s.specSSM[il] = dst
		}
		if err := s.backend.Copy(dst, src); err != nil {
			return err
		}
	}
	for il, src := range s.DeviceConv {
		if src == nil {
			continue
		}
		dst := s.specConv[il]
		if dst == nil {
			var err error
			dst, err = s.backend.Alloc(src.Dims(), quant.TypeF32)
			if err != nil {
				return err
			}
			s.specConv[il] = dst
		}
		if err := s.backend.Copy(dst, src); err != nil {
			return err
		}
	}
	s.specActive = true
	return nil
}

// SpecRestore rewinds the recurrent state to the last SpecSnapshot. KV buffers
// keep their data; only their lengths are rewound.
func (s *State) SpecRestore() error {
	if s == nil || !s.specActive {
		return nil
	}
	s.specActive = false
	if s.backend == nil {
		for i, c := range s.KV {
			if c != nil && i < len(s.specHostKVLen) {
				c.length = s.specHostKVLen[i]
			}
		}
		if s.MTPKV != nil {
			s.MTPKV.length = s.specHostMTPLen
		}
		for il := range s.SSM {
			if il < len(s.specHostSSM) && len(s.SSM[il]) == len(s.specHostSSM[il]) {
				copy(s.SSM[il], s.specHostSSM[il])
			}
		}
		for il := range s.Conv {
			if il < len(s.specHostConv) && len(s.Conv[il]) == len(s.specHostConv[il]) {
				copy(s.Conv[il], s.specHostConv[il])
			}
		}
		return nil
	}
	for i, c := range s.DeviceKV {
		if c != nil && i < len(s.specKVLen) {
			c.Len = s.specKVLen[i]
		}
	}
	if s.DeviceMTPKV != nil {
		s.DeviceMTPKV.Len = s.specDeviceMTPLen
	}
	for il, dst := range s.DeviceSSM {
		if dst != nil && il < len(s.specSSM) && s.specSSM[il] != nil {
			if err := s.backend.Copy(dst, s.specSSM[il]); err != nil {
				return err
			}
		}
	}
	for il, dst := range s.DeviceConv {
		if dst != nil && il < len(s.specConv) && s.specConv[il] != nil {
			if err := s.backend.Copy(dst, s.specConv[il]); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *State) freeSpec() {
	if s == nil {
		return
	}
	if s.backend != nil {
		for _, b := range s.specSSM {
			if b != nil {
				s.backend.Free(b)
			}
		}
		for _, b := range s.specConv {
			if b != nil {
				s.backend.Free(b)
			}
		}
	}
	s.specSSM = nil
	s.specConv = nil
	s.specHostSSM = nil
	s.specHostConv = nil
	s.specActive = false
}
