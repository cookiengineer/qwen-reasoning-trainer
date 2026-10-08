package qwen38

import (
	"fmt"
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// State carries the recurrent and KV-cache state between forward calls. A
// fresh (zeroed) state is used for the first call.
type State struct {
	// Conv holds the causal-conv history per layer: (DConv-1) rows of ConvDim.
	Conv [][]float32
	// SSM holds the GatedDeltaNet state per layer: HeadVDim*HeadVDim*DtRank.
	SSM [][]float32
	// KV holds the key/value cache of each full-attention layer.
	KV []*kvCache
}

// kvCache stores the keys and values of a full-attention layer, laid out as
// [headDim, nHeadKV, length].
type kvCache struct {
	k, v   []float32
	length int
}

// NewState allocates a zeroed state for cfg.
func NewState(cfg *Config) *State {
	s := &State{
		Conv: make([][]float32, cfg.TrunkLayers()),
		SSM:  make([][]float32, cfg.TrunkLayers()),
		KV:   make([]*kvCache, cfg.TrunkLayers()),
	}
	for il := 0; il < cfg.TrunkLayers(); il++ {
		if cfg.IsRecurrent(il) {
			s.Conv[il] = make([]float32, (cfg.DConv-1)*cfg.ConvDim())
			s.SSM[il] = make([]float32, cfg.HeadVDim()*cfg.HeadVDim()*cfg.DtRank)
		} else {
			s.KV[il] = &kvCache{}
		}
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
}

// ForwardResult is the output of a forward pass.
type ForwardResult struct {
	// Logits has Dims [n_vocab, n_tokens].
	Logits *compute.Tensor
	// Hidden, when requested, holds TrunkLayers+1 residual-stream tensors:
	// index 0 is the embedding output, index k+1 the output of block k.
	Hidden []*compute.Tensor
	// State is the recurrent state after the pass.
	State *State
}

// Forward runs the model on the given token ids.
func (m *Model) Forward(tokens []int32, opts ForwardOptions) (*ForwardResult, error) {
	cfg := m.Cfg
	T := len(tokens)
	if T == 0 {
		return nil, fmt.Errorf("qwen38: empty prompt")
	}
	x, err := compute.GetRows(m.W.TokenEmbd, tokens)
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

		h := rms(x, lw.AttnNorm.F32, cfg.RmsEps)
		var attnOut *compute.Tensor
		if cfg.IsRecurrent(il) {
			attnOut, err = m.linearAttn(lw, h, st, il)
		} else {
			attnOut, err = m.fullAttn(lw, h, positions, st.KV[il])
		}
		if err != nil {
			return nil, fmt.Errorf("qwen38: layer %d attention: %w", il, err)
		}
		x, err = add(x, attnOut)
		if err != nil {
			return nil, err
		}

		h2 := rms(x, lw.PostAttnNorm.F32, cfg.RmsEps)
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

	xn := rms(x, m.W.OutputNorm.F32, cfg.RmsEps)
	logits, err := mm(m.W.Output, xn)
	if err != nil {
		return nil, err
	}
	res.Logits = logits
	res.State = st
	return res, nil
}

// ffn applies the SwiGLU feed-forward network to a normalized activation.
func (m *Model) ffn(lw *LayerWeights, h *compute.Tensor) (*compute.Tensor, error) {
	gate, err := mm(lw.FfnGate, h)
	if err != nil {
		return nil, err
	}
	up, err := mm(lw.FfnUp, h)
	if err != nil {
		return nil, err
	}
	act := siluInto(gate)
	act, err = compute.Mul(act, up)
	if err != nil {
		return nil, err
	}
	return mm(lw.FfnDown, act)
}

// fullAttn runs a full-attention block on the normalized activation h, using
// and updating the layer's KV cache.
func (m *Model) fullAttn(lw *LayerWeights, h *compute.Tensor, positions []int32, cache *kvCache) (*compute.Tensor, error) {
	cfg := m.Cfg
	hd := cfg.HeadDim
	T := h.Ne(1)

	qg, err := mm(lw.AttnQ, h)
	if err != nil {
		return nil, err
	}
	kcur, err := mm(lw.AttnK, h)
	if err != nil {
		return nil, err
	}
	vcur, err := mm(lw.AttnV, h)
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

	qn := rms(q, lw.AttnQNorm.F32, cfg.RmsEps)
	kn, err := reshape3(kcur, hd, cfg.NHeadKV, T)
	if err != nil {
		return nil, err
	}
	kn = rms(kn, lw.AttnKNorm.F32, cfg.RmsEps)
	vn, err := reshape3(vcur, hd, cfg.NHeadKV, T)
	if err != nil {
		return nil, err
	}

	qn, err = compute.RoPENeoX(qn, positions, cfg.RopeTheta, cfg.NRot)
	if err != nil {
		return nil, err
	}
	kn, err = compute.RoPENeoX(kn, positions, cfg.RopeTheta, cfg.NRot)
	if err != nil {
		return nil, err
	}

	// Append to the KV cache and attend over the full history.
	kc, vc, total := appendKVCache(cache, kn, vn, hd, cfg.NHeadKV, T)
	causal := T == total

	scale := float32(1.0)
	if hd > 0 {
		scale = float32(1.0 / math.Sqrt(float64(hd)))
	}
	attn, err := compute.Attention(qn, kc, vc, nHead, cfg.NHeadKV, scale, causal)
	if err != nil {
		return nil, err
	}

	gateSig := sigmoidInto(gate)
	attn, err = compute.Mul(attn, gateSig)
	if err != nil {
		return nil, err
	}
	// Attention output is [headDim, nHead, T]; the layout is identical to
	// [nHead*headDim, T], which is what the output projection expects.
	attnFlat := &compute.Tensor{Dims: []int{nHead * hd, T}, Type: attn.Type, F32: attn.F32}
	return mm(lw.AttnOutput, attnFlat)
}

// linearAttn runs a GatedDeltaNet block on the normalized activation h.
func (m *Model) linearAttn(lw *LayerWeights, h *compute.Tensor, st *State, il int) (*compute.Tensor, error) {
	cfg := m.Cfg
	T := h.Ne(1)
	headVDim := cfg.HeadVDim()
	nV := cfg.DtRank
	nK := cfg.GroupCount
	headKDim := cfg.DState

	qkv, err := mm(lw.AttnQKV, h) // [convDim, T]
	if err != nil {
		return nil, err
	}
	z, err := mm(lw.AttnGate, h) // [valueDim, T]
	if err != nil {
		return nil, err
	}

	betaRaw, err := mm(lw.SSMBeta, h) // [dtRank, T]
	if err != nil {
		return nil, err
	}
	beta := sigmoidInto(betaRaw)

	alpha, err := mm(lw.SSMAlpha, h) // [dtRank, T]
	if err != nil {
		return nil, err
	}
	gate := compute.NewF32(1, nV, T)
	for t := 0; t < T; t++ {
		for hh := 0; hh < nV; hh++ {
			a := alpha.F32[hh+t*nV] + lw.SSMDt.F32[hh]
			gate.F32[hh+t*nV] = softplus(a) * lw.SSMA.F32[hh]
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
	norm := rms(out, lw.SSMNorm.F32, cfg.RmsEps) // [headVDim, nV, T]
	zr := &compute.Tensor{Dims: []int{headVDim, nV, T}, F32: z.F32}
	gated, err := compute.Mul(norm, siluInto(zr))
	if err != nil {
		return nil, err
	}
	flat := &compute.Tensor{Dims: []int{cfg.ValueDim(), T}, F32: gated.F32}
	return mm(lw.SSMOut, flat)
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

// appendKVCache appends T keys/values to cache and returns the combined cache.
func appendKVCache(cache *kvCache, k, v *compute.Tensor, hd, nKV, T int) (*compute.Tensor, *compute.Tensor, int) {
	stride := hd * nKV
	total := cache.length + T
	nk := compute.NewF32(hd, nKV, total)
	nv := compute.NewF32(hd, nKV, total)
	copy(nk.F32, cache.k)
	copy(nv.F32, cache.v)
	for t := 0; t < T; t++ {
		copy(nk.F32[(cache.length+t)*stride:], k.F32[t*stride:(t+1)*stride])
		copy(nv.F32[(cache.length+t)*stride:], v.F32[t*stride:(t+1)*stride])
	}
	cache.k = nk.F32
	cache.v = nv.F32
	cache.length = total
	return nk, nv, total
}
