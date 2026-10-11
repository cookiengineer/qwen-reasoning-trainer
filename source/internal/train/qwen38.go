package train

import (
	"fmt"
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/autograd"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

// QwenModel is a trainable Qwen3.8 graph. The quantized base weights are frozen
// (only the activation gradient flows through them) and F32 LoRA adapters are
// trained on the configured projections. It is the host reference; the device
// graph will mirror this structure.
type QwenModel struct {
	Cfg     *qwen38.Config
	W       *qwen38.Weights
	LoCfg   LoRAConfig
	entries []*loraEntry
	byW     map[*qwen38.Weight]*loraEntry
	params  []*compute.Tensor
}

type loraEntry struct {
	w     *qwen38.Weight
	l     *LoRA
	name  string
	aLeaf *autograd.Value
	bLeaf *autograd.Value
}

// namedWeight pairs a weight with its GGUF tensor name.
type namedWeight struct {
	name string
	w    *qwen38.Weight
}

// NewQwenModel attaches LoRA adapters to the configured projections.
func NewQwenModel(cfg *qwen38.Config, w *qwen38.Weights, loCfg LoRAConfig, seed uint64) *QwenModel {
	m := &QwenModel{Cfg: cfg, W: w, LoCfg: loCfg, byW: map[*qwen38.Weight]*loraEntry{}}
	for il := 0; il < cfg.TrunkLayers(); il++ {
		lw := &w.Layers[il]
		for _, tw := range targetWeights(lw, cfg.IsRecurrent(il), loCfg.Targets, il) {
			if tw.w == nil {
				continue
			}
			seed++
			e := &loraEntry{w: tw.w, name: tw.name, l: NewLoRA(seed, tw.w.In(), tw.w.Out(), loCfg.Rank, loCfg.Alpha)}
			m.entries = append(m.entries, e)
			m.byW[tw.w] = e
			m.params = append(m.params, e.l.A, e.l.B)
		}
	}
	return m
}

// Params returns the trainable adapter tensors (A/B pairs in entry order).
func (m *QwenModel) Params() []*compute.Tensor { return m.params }

// Adapters exposes the per-weight adapters (for merge/debugging).
func (m *QwenModel) Adapters() map[*qwen38.Weight]*LoRA {
	out := make(map[*qwen38.Weight]*LoRA, len(m.entries))
	for _, e := range m.entries {
		out[e.w] = e.l
	}
	return out
}

func targetWeights(lw *qwen38.LayerWeights, recurrent bool, targets []string, il int) []namedWeight {
	has := func(s string) bool {
		for _, t := range targets {
			if t == s {
				return true
			}
		}
		return false
	}
	blk := func(s string) string { return fmt.Sprintf("blk.%d.%s.weight", il, s) }
	var out []namedWeight
	add := func(name string, w *qwen38.Weight) {
		if w != nil {
			out = append(out, namedWeight{name: blk(name), w: w})
		}
	}
	if recurrent {
		if has("attn_qkv") {
			add("attn_qkv", lw.AttnQKV)
		}
		if has("attn_gate") {
			add("attn_gate", lw.AttnGate)
		}
		if has("ssm_out") {
			add("ssm_out", lw.SSMOut)
		}
	} else {
		if has("attn_q") {
			add("attn_q", lw.AttnQ)
		}
		if has("attn_k") {
			add("attn_k", lw.AttnK)
		}
		if has("attn_v") {
			add("attn_v", lw.AttnV)
		}
		if has("attn_output") {
			add("attn_output", lw.AttnOutput)
		}
	}
	if has("ffn_gate") {
		add("ffn_gate", lw.FfnGate)
	}
	if has("ffn_up") {
		add("ffn_up", lw.FfnUp)
	}
	if has("ffn_down") {
		add("ffn_down", lw.FfnDown)
	}
	return out
}

// mm multiplies a frozen weight (plus its optional adapter) by x.
func (m *QwenModel) mm(w *qwen38.Weight, x *autograd.Value) *autograd.Value {
	y := autograd.VMatMulWeight(w, x)
	if e := m.byW[w]; e != nil {
		u := autograd.VMatMul(e.aLeaf, x)
		upd := autograd.VScale(autograd.VMatMul(e.bLeaf, u), e.l.Scale)
		y = autograd.VAdd(y, upd)
	}
	return y
}

// build constructs the forward graph and returns logits, loss, and the adapter
// leaves (matching m.params order).
func (m *QwenModel) build(tokens []int32, targets []int, positions []int32) (logits, loss *autograd.Value, leaves []*autograd.Value) {
	cfg := m.Cfg
	leaves = make([]*autograd.Value, len(m.params))
	for i, p := range m.params {
		leaves[i] = autograd.Param(p)
	}
	for k, e := range m.entries {
		e.aLeaf = leaves[2*k]
		e.bLeaf = leaves[2*k+1]
	}

	x := autograd.VGetRowsWeight(m.W.TokenEmbd, tokens)
	for il := 0; il < cfg.TrunkLayers(); il++ {
		lw := &m.W.Layers[il]
		h := autograd.VRMSNorm(x, lw.AttnNorm, cfg.RmsEps)
		var attnOut *autograd.Value
		if cfg.IsRecurrent(il) {
			attnOut = m.linearAttn(lw, h)
		} else {
			attnOut = m.fullAttn(lw, h, positions)
		}
		x = autograd.VAdd(x, attnOut)
		h2 := autograd.VRMSNorm(x, lw.PostAttnNorm, cfg.RmsEps)
		x = autograd.VAdd(x, m.ffn(lw, h2))
	}
	xn := autograd.VRMSNorm(x, m.W.OutputNorm, cfg.RmsEps)
	logits = m.mm(m.W.Output, xn)
	loss = autograd.VCrossEntropy(logits, targets, -100)
	return logits, loss, leaves
}

func (m *QwenModel) fullAttn(lw *qwen38.LayerWeights, h *autograd.Value, positions []int32) *autograd.Value {
	cfg := m.Cfg
	hd := cfg.HeadDim
	T := h.Data.Ne(1)

	qg := m.mm(lw.AttnQ, h)
	qcur, gate := autograd.VSplitQG(qg, hd, cfg.NHead, T)
	kcur := m.mm(lw.AttnK, h)
	vcur := m.mm(lw.AttnV, h)

	q := autograd.VRMSNorm(qcur, lw.AttnQNorm, cfg.RmsEps)
	kn := autograd.VRMSNorm(autograd.VReshape(kcur, hd, cfg.NHeadKV, T), lw.AttnKNorm, cfg.RmsEps)
	vn := autograd.VReshape(vcur, hd, cfg.NHeadKV, T)

	q = autograd.VRoPENeoX(q, positions, cfg.RopeTheta, cfg.NRot)
	kn = autograd.VRoPENeoX(kn, positions, cfg.RopeTheta, cfg.NRot)

	scale := float32(1.0)
	if hd > 0 {
		scale = float32(1 / math.Sqrt(float64(hd)))
	}
	attn := autograd.VAttention(q, kn, vn, cfg.NHead, cfg.NHeadKV, scale, true)
	attn = autograd.VMul(attn, autograd.VSigmoid(gate))
	flat := autograd.VReshape(attn, cfg.NHead*hd, T)
	return m.mm(lw.AttnOutput, flat)
}

func (m *QwenModel) linearAttn(lw *qwen38.LayerWeights, h *autograd.Value) *autograd.Value {
	cfg := m.Cfg
	T := h.Data.Ne(1)
	headVDim := cfg.HeadVDim()
	nV := cfg.DtRank
	nK := cfg.GroupCount
	headKDim := cfg.DState

	qkv := m.mm(lw.AttnQKV, h)
	z := m.mm(lw.AttnGate, h)
	betaRaw := m.mm(lw.SSMBeta, h)
	beta := autograd.VSigmoid(betaRaw)
	alpha := m.mm(lw.SSMAlpha, h)

	dtT := compute.NewF32(nV, T)
	aT := compute.NewF32(nV, T)
	for t := 0; t < T; t++ {
		for hh := 0; hh < nV; hh++ {
			dtT.F32[hh+t*nV] = lw.SSMDt[hh]
			aT.F32[hh+t*nV] = lw.SSMA[hh]
		}
	}
	gate := autograd.VMul(autograd.VSoftplus(autograd.VAdd(alpha, autograd.Const(dtT))), autograd.Const(aT))
	gate = autograd.VReshape(gate, 1, nV, T)
	betaT := autograd.VReshape(beta, 1, nV, T)

	convDim := cfg.ConvDim()
	convIn := autograd.VConvInput(qkv, cfg.DConv, convDim, T)
	convOut := autograd.VSilu(autograd.VSSMConv(convIn, autograd.Const(lw.SSMConv1d)))
	q := autograd.VGatherHeads(convOut, 0, headKDim, nK, T, convDim)
	k := autograd.VGatherHeads(convOut, cfg.KeyDim(), headKDim, nK, T, convDim)
	v := autograd.VGatherHeads(convOut, 2*cfg.KeyDim(), headVDim, nV, T, convDim)
	q = autograd.VL2Norm(q, 1e-6)
	k = autograd.VL2Norm(k, 1e-6)
	q48 := autograd.VRepeatHeads(q, headKDim, nK, nV, T)
	k48 := autograd.VRepeatHeads(k, headKDim, nK, nV, T)

	state := autograd.Const(compute.NewF32(headVDim, headVDim, nV))
	out, _ := autograd.VGatedDeltaNet(q48, k48, v, gate, betaT, state)

	norm := autograd.VRMSNorm(out, lw.SSMNorm, cfg.RmsEps)
	zr := autograd.VReshape(z, headVDim, nV, T)
	gated := autograd.VMul(norm, autograd.VSilu(zr))
	flat := autograd.VReshape(gated, cfg.ValueDim(), T)
	return m.mm(lw.SSMOut, flat)
}

func (m *QwenModel) ffn(lw *qwen38.LayerWeights, h *autograd.Value) *autograd.Value {
	gate := m.mm(lw.FfnGate, h)
	up := m.mm(lw.FfnUp, h)
	act := autograd.VMul(autograd.VSilu(gate), up)
	return m.mm(lw.FfnDown, act)
}

// Logits returns the forward logits [vocab,T] for a token sequence.
func (m *QwenModel) Logits(tokens []int32) *compute.Tensor {
	T := len(tokens)
	positions := make([]int32, T)
	for i := range positions {
		positions[i] = int32(i)
	}
	targets := make([]int, T)
	for i := range targets {
		targets[i] = -100
	}
	logits, _, _ := m.build(tokens, targets, positions)
	return logits.Data
}

// ForwardBackward runs forward+backward for one example. Targets are the
// next-token ids; positions masked off in LossMask are ignored.
func (m *QwenModel) ForwardBackward(ex Example) (float32, []*compute.Tensor, error) {
	T := len(ex.Tokens)
	if T < 2 {
		return 0, nil, fmt.Errorf("train: example has %d tokens, need >= 2", T)
	}
	positions := make([]int32, T)
	for i := range positions {
		positions[i] = int32(i)
	}
	targets := make([]int, T)
	for i := range targets {
		targets[i] = -100
	}
	for t := 0; t < T-1; t++ {
		if len(ex.LossMask) == T && !ex.LossMask[t+1] {
			continue
		}
		targets[t] = int(ex.Tokens[t+1])
	}
	_, loss, leaves := m.build(ex.Tokens, targets, positions)
	loss.Backward()
	grads := make([]*compute.Tensor, len(leaves))
	found := false
	for i, l := range leaves {
		if l.Grad != nil {
			grads[i] = l.Grad
			found = true
		}
	}
	if !found {
		return 0, nil, fmt.Errorf("train: no trainable tokens in example")
	}
	return loss.Data.F32[0], grads, nil
}
