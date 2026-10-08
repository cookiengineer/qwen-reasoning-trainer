package train

import (
	"fmt"
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

// devLin is a device-resident frozen-weight linear layer plus its LoRA adapter.
type devLin struct {
	be    compute.Backend
	w     *qwen38.Weight
	base  compute.Buffer
	lora  *LoRA
	a, b  compute.Buffer
	scale float32
	// name is the GGUF tensor name of the frozen base weight, used to key the
	// merged overrides (e.g. "blk.3.attn_q.weight").
	name string
}

func newDevLin(be compute.Backend, w *qwen38.Weight, loCfg LoRAConfig, seed uint64) (devLin, error) {
	base, err := be.UploadWeight(w.Typ, w.Raw, w.Dims)
	if err != nil {
		return devLin{}, err
	}
	l := NewLoRA(seed, w.In(), w.Out(), loCfg.Rank, loCfg.Alpha)
	a, err := be.Upload(l.A)
	if err != nil {
		return devLin{}, err
	}
	b, err := be.Upload(l.B)
	if err != nil {
		return devLin{}, err
	}
	return devLin{be: be, w: w, base: base, lora: l, a: a, b: b, scale: l.Scale}, nil
}

func (d *devLin) forward(x compute.Buffer) (y, u compute.Buffer, err error) {
	return deviceLinear(d.be, d.base, d.a, d.b, x, d.scale)
}

// backward returns dX (the input gradient) plus the two adapter gradients.
func (d *devLin) backward(x, u, dY compute.Buffer) (dX, dA, dB compute.Buffer, err error) {
	return deviceLinearBackward(d.be, d.base, d.a, d.b, x, u, dY, d.scale)
}

// DeviceAttention is a resident full-attention transformer block (attention +
// SwiGLU FFN) with LoRA on the projection weights. Its forward and backward run
// entirely on the backend.
type DeviceAttention struct {
	be    compute.Backend
	cfg   *qwen38.Config
	eps   float32
	scale float32
	theta float64
	nrot  int

	attnNormB, postNormB, qNormB, kNormB compute.Buffer
	attnNorm, postNorm, qNorm, kNorm     []float32

	q, k, v, o, fg, fu, fd devLin
}

// NewDeviceAttention uploads a full-attention layer's frozen weights and builds
// its LoRA adapters.
func NewDeviceAttention(be compute.Backend, cfg *qwen38.Config, lw *qwen38.LayerWeights, loCfg LoRAConfig, seed uint64) (*DeviceAttention, error) {
	if lw.AttnQ == nil {
		return nil, fmt.Errorf("train: layer is not full attention")
	}
	d := &DeviceAttention{
		be: be, cfg: cfg, eps: cfg.RmsEps,
		scale: float32(1 / math.Sqrt(float64(cfg.HeadDim))),
		theta: cfg.RopeTheta, nrot: cfg.NRot,
		attnNorm: lw.AttnNorm, postNorm: lw.PostAttnNorm,
		qNorm: lw.AttnQNorm, kNorm: lw.AttnKNorm,
	}
	var err error
	if d.attnNormB, err = uploadVec(be, lw.AttnNorm); err != nil {
		return nil, err
	}
	if d.postNormB, err = uploadVec(be, lw.PostAttnNorm); err != nil {
		return nil, err
	}
	if d.qNormB, err = uploadVec(be, lw.AttnQNorm); err != nil {
		return nil, err
	}
	if d.kNormB, err = uploadVec(be, lw.AttnKNorm); err != nil {
		return nil, err
	}
	for _, e := range []struct {
		dst  *devLin
		w    *qwen38.Weight
		seed uint64
	}{
		{&d.q, lw.AttnQ, seed + 1}, {&d.k, lw.AttnK, seed + 2},
		{&d.v, lw.AttnV, seed + 3}, {&d.o, lw.AttnOutput, seed + 4},
		{&d.fg, lw.FfnGate, seed + 5}, {&d.fu, lw.FfnUp, seed + 6},
		{&d.fd, lw.FfnDown, seed + 7},
	} {
		if *e.dst, err = newDevLin(be, e.w, loCfg, e.seed); err != nil {
			return nil, err
		}
	}
	return d, nil
}

func uploadVec(be compute.Backend, v []float32) (compute.Buffer, error) {
	return be.Upload(&compute.Tensor{Dims: []int{len(v)}, F32: v})
}

// attentionCtx holds the forward activations needed by the backward pass.
type attentionCtx struct {
	y                                                      compute.Buffer
	x, h, qg, qPre, kPre, vn, qR, kR, attn0, gateSig, attn compute.Buffer
	attnFlat, o, x1, h2, fgPre, up, siluFG, act, ff        compute.Buffer
	uQ, uK, uV, uO, uFG, uFU, uFD                          compute.Buffer
}

// Forward runs the block on the device and returns the output in ctx.y.
func (d *DeviceAttention) Forward(x compute.Buffer, positions []int32) (*attentionCtx, error) {
	be := d.be
	cfg := d.cfg
	hd := cfg.HeadDim
	T := x.Dims()[1]
	ctx := &attentionCtx{x: x}
	var err error

	if ctx.h, err = be.RMSNorm(x, d.attnNormB, d.eps); err != nil {
		return nil, err
	}
	if ctx.qg, ctx.uQ, err = d.q.forward(ctx.h); err != nil {
		return nil, err
	}
	qSplit, gate, err := be.SplitQG(ctx.qg, hd, cfg.NHead, T)
	if err != nil {
		return nil, err
	}
	if ctx.gateSig, err = be.Unary(compute.UnarySigmoid, gate); err != nil {
		return nil, err
	}
	ctx.qPre = qSplit
	var kBuf, vBuf compute.Buffer
	if kBuf, ctx.uK, err = d.k.forward(ctx.h); err != nil {
		return nil, err
	}
	if vBuf, ctx.uV, err = d.v.forward(ctx.h); err != nil {
		return nil, err
	}
	if ctx.kPre, err = be.Reshape(kBuf, []int{hd, cfg.NHeadKV, T}); err != nil {
		return nil, err
	}
	if ctx.vn, err = be.Reshape(vBuf, []int{hd, cfg.NHeadKV, T}); err != nil {
		return nil, err
	}
	qN, err := be.RMSNorm(ctx.qPre, d.qNormB, d.eps)
	if err != nil {
		return nil, err
	}
	kN, err := be.RMSNorm(ctx.kPre, d.kNormB, d.eps)
	if err != nil {
		return nil, err
	}
	if ctx.qR, err = be.RoPE(qN, positions, d.theta, d.nrot); err != nil {
		return nil, err
	}
	if ctx.kR, err = be.RoPE(kN, positions, d.theta, d.nrot); err != nil {
		return nil, err
	}
	if ctx.attn0, err = be.Attention(ctx.qR, ctx.kR, ctx.vn, cfg.NHead, cfg.NHeadKV, d.scale, true); err != nil {
		return nil, err
	}
	if ctx.attn, err = be.Binary(compute.BinaryMul, ctx.attn0, ctx.gateSig); err != nil {
		return nil, err
	}
	if ctx.attnFlat, err = be.Reshape(ctx.attn, []int{cfg.NHead * hd, T}); err != nil {
		return nil, err
	}
	if ctx.o, ctx.uO, err = d.o.forward(ctx.attnFlat); err != nil {
		return nil, err
	}
	if ctx.x1, err = be.Binary(compute.BinaryAdd, x, ctx.o); err != nil {
		return nil, err
	}
	if ctx.h2, err = be.RMSNorm(ctx.x1, d.postNormB, d.eps); err != nil {
		return nil, err
	}
	if ctx.fgPre, ctx.uFG, err = d.fg.forward(ctx.h2); err != nil {
		return nil, err
	}
	if ctx.up, ctx.uFU, err = d.fu.forward(ctx.h2); err != nil {
		return nil, err
	}
	if ctx.siluFG, err = be.Unary(compute.UnarySilu, ctx.fgPre); err != nil {
		return nil, err
	}
	if ctx.act, err = be.Binary(compute.BinaryMul, ctx.siluFG, ctx.up); err != nil {
		return nil, err
	}
	if ctx.ff, ctx.uFD, err = d.fd.forward(ctx.act); err != nil {
		return nil, err
	}
	if ctx.y, err = be.Binary(compute.BinaryAdd, ctx.x1, ctx.ff); err != nil {
		return nil, err
	}
	return ctx, nil
}

// attentionGrads holds the adapter gradients and the input gradient.
type attentionGrads struct {
	DX             compute.Buffer
	DQ, DK, DV, DO [2]compute.Buffer
	DFG, DFU, DFD  [2]compute.Buffer
}

// Backward propagates dY through the block. dY has the shape of the block output.
func (d *DeviceAttention) Backward(ctx *attentionCtx, dY compute.Buffer, positions []int32) (*attentionGrads, error) {
	be := d.be
	cfg := d.cfg
	hd := cfg.HeadDim
	T := dY.Dims()[1]
	g := &attentionGrads{}
	var err error

	// FFN: y = x1 + FfnDown*act.
	var dAct, dA, dB compute.Buffer
	if dAct, dA, dB, err = d.fd.backward(ctx.act, ctx.uFD, dY); err != nil {
		return nil, err
	}
	g.DFD = [2]compute.Buffer{dA, dB}
	dSilu, err := be.Binary(compute.BinaryMul, dAct, ctx.up)
	if err != nil {
		return nil, err
	}
	dUp, err := be.Binary(compute.BinaryMul, dAct, ctx.siluFG)
	if err != nil {
		return nil, err
	}
	dFGpre, err := be.SiluBack(ctx.fgPre, dSilu)
	if err != nil {
		return nil, err
	}
	var dH2fg, dH2fu compute.Buffer
	if dH2fg, dA, dB, err = d.fg.backward(ctx.h2, ctx.uFG, dFGpre); err != nil {
		return nil, err
	}
	g.DFG = [2]compute.Buffer{dA, dB}
	if dH2fu, dA, dB, err = d.fu.backward(ctx.h2, ctx.uFU, dUp); err != nil {
		return nil, err
	}
	g.DFU = [2]compute.Buffer{dA, dB}
	dH2, err := be.Binary(compute.BinaryAdd, dH2fg, dH2fu)
	if err != nil {
		return nil, err
	}
	dX1rms, err := be.RMSNormBack(ctx.x1, d.postNormB, dH2, d.eps)
	if err != nil {
		return nil, err
	}
	dX1, err := be.Binary(compute.BinaryAdd, dY, dX1rms)
	if err != nil {
		return nil, err
	}

	// Attention output projection: x1 = x + o.
	var dAttnFlat compute.Buffer
	if dAttnFlat, dA, dB, err = d.o.backward(ctx.attnFlat, ctx.uO, dX1); err != nil {
		return nil, err
	}
	g.DO = [2]compute.Buffer{dA, dB}
	dAttn, err := be.Reshape(dAttnFlat, []int{hd, cfg.NHead, T})
	if err != nil {
		return nil, err
	}
	dAttn0, err := be.Binary(compute.BinaryMul, dAttn, ctx.gateSig)
	if err != nil {
		return nil, err
	}
	dGateSig, err := be.Binary(compute.BinaryMul, dAttn, ctx.attn0)
	if err != nil {
		return nil, err
	}
	dGatePre, err := be.Binary(compute.BinarySigmoidBack, ctx.gateSig, dGateSig)
	if err != nil {
		return nil, err
	}
	dQR, dKR, dV, err := be.AttentionBackward(ctx.qR, ctx.kR, ctx.vn, dAttn0, cfg.NHead, cfg.NHeadKV, d.scale, true)
	if err != nil {
		return nil, err
	}
	neg := make([]int32, len(positions))
	for i, p := range positions {
		neg[i] = -p
	}
	dQN, err := be.RoPE(dQR, neg, d.theta, d.nrot)
	if err != nil {
		return nil, err
	}
	dKN, err := be.RoPE(dKR, neg, d.theta, d.nrot)
	if err != nil {
		return nil, err
	}
	dQpre, err := be.RMSNormBack(ctx.qPre, d.qNormB, dQN, d.eps)
	if err != nil {
		return nil, err
	}
	dKpre, err := be.RMSNormBack(ctx.kPre, d.kNormB, dKN, d.eps)
	if err != nil {
		return nil, err
	}
	dQG, err := be.SplitQGBack(dQpre, dGatePre)
	if err != nil {
		return nil, err
	}
	dKflat, err := be.Reshape(dKpre, []int{cfg.NHeadKV * hd, T})
	if err != nil {
		return nil, err
	}
	dVflat, err := be.Reshape(dV, []int{cfg.NHeadKV * hd, T})
	if err != nil {
		return nil, err
	}
	var dHq, dHk, dHv compute.Buffer
	if dHq, dA, dB, err = d.q.backward(ctx.h, ctx.uQ, dQG); err != nil {
		return nil, err
	}
	g.DQ = [2]compute.Buffer{dA, dB}
	if dHk, dA, dB, err = d.k.backward(ctx.h, ctx.uK, dKflat); err != nil {
		return nil, err
	}
	g.DK = [2]compute.Buffer{dA, dB}
	if dHv, dA, dB, err = d.v.backward(ctx.h, ctx.uV, dVflat); err != nil {
		return nil, err
	}
	g.DV = [2]compute.Buffer{dA, dB}
	dH, err := be.Binary(compute.BinaryAdd, dHq, dHk)
	if err != nil {
		return nil, err
	}
	if dH, err = be.Binary(compute.BinaryAdd, dH, dHv); err != nil {
		return nil, err
	}
	dXattn, err := be.RMSNormBack(ctx.x, d.attnNormB, dH, d.eps)
	if err != nil {
		return nil, err
	}
	if g.DX, err = be.Binary(compute.BinaryAdd, dX1, dXattn); err != nil {
		return nil, err
	}
	return g, nil
}
