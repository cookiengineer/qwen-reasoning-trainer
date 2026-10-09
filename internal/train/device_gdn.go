package train

import (
	"fmt"
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/model/qwen38"
)

// deviceGDNChunk is the chunk size for the resident GatedDeltaNet backward.
const deviceGDNChunk = 64

// chunkedGDNBackend is implemented by the Vulkan backend. When present, the
// resident GDN backward uses the matrix-based chunked kernel (no atomics, no
// sequence-length cap) instead of the naive atomic one.
type chunkedGDNBackend interface {
	GatedDeltaNetChunkedBackward(q, k, v, g, beta, state, dOut, dNewState compute.Buffer, chunk int) (compute.Buffer, compute.Buffer, compute.Buffer, compute.Buffer, compute.Buffer, compute.Buffer, error)
}

// DeviceGDN is a resident GatedDeltaNet (linear-attention) transformer block
// (causal conv + gated delta rule + SwiGLU FFN) with LoRA on qkv/gate/ssm_out.
// Its forward and backward run entirely on the backend.
type DeviceGDN struct {
	be    compute.Backend
	cfg   *qwen38.Config
	eps   float32
	scale float32

	attnNormB, postNormB, ssmNormB compute.Buffer
	conv1dB, alphaB, betaB         compute.Buffer
	ssma, ssdt                     []float32

	qkv, gate, out devLin
	fg, fu, fd     devLin
}

// NewDeviceGDN uploads a linear-attention layer's frozen weights and adapters.
func NewDeviceGDN(be compute.Backend, cfg *qwen38.Config, lw *qwen38.LayerWeights, loCfg LoRAConfig, seed uint64) (*DeviceGDN, error) {
	if lw.AttnQKV == nil {
		return nil, fmt.Errorf("train: layer is not linear attention")
	}
	d := &DeviceGDN{
		be: be, cfg: cfg, eps: cfg.RmsEps,
		scale: float32(1 / math.Sqrt(float64(cfg.HeadVDim()))),
		ssma:  lw.SSMA, ssdt: lw.SSMDt,
	}
	var err error
	if d.attnNormB, err = uploadVec(be, lw.AttnNorm); err != nil {
		return nil, err
	}
	if d.postNormB, err = uploadVec(be, lw.PostAttnNorm); err != nil {
		return nil, err
	}
	if d.ssmNormB, err = uploadVec(be, lw.SSMNorm); err != nil {
		return nil, err
	}
	if d.conv1dB, err = be.Upload(lw.SSMConv1d); err != nil {
		return nil, err
	}
	if d.alphaB, err = be.UploadWeight(lw.SSMAlpha.Typ, lw.SSMAlpha.Raw, lw.SSMAlpha.Dims); err != nil {
		return nil, err
	}
	if d.betaB, err = be.UploadWeight(lw.SSMBeta.Typ, lw.SSMBeta.Raw, lw.SSMBeta.Dims); err != nil {
		return nil, err
	}
	if d.qkv, err = newDevLin(be, lw.AttnQKV, loCfg, seed+1); err != nil {
		return nil, err
	}
	if d.gate, err = newDevLin(be, lw.AttnGate, loCfg, seed+2); err != nil {
		return nil, err
	}
	if d.out, err = newDevLin(be, lw.SSMOut, loCfg, seed+3); err != nil {
		return nil, err
	}
	if d.fg, err = newDevLin(be, lw.FfnGate, loCfg, seed+4); err != nil {
		return nil, err
	}
	if d.fu, err = newDevLin(be, lw.FfnUp, loCfg, seed+5); err != nil {
		return nil, err
	}
	if d.fd, err = newDevLin(be, lw.FfnDown, loCfg, seed+6); err != nil {
		return nil, err
	}
	return d, nil
}

// constVec builds a [n, T] buffer with v[hh] repeated per token.
func constVec(be compute.Backend, v []float32, T int) (compute.Buffer, error) {
	n := len(v)
	data := make([]float32, n*T)
	for t := 0; t < T; t++ {
		copy(data[t*n:], v)
	}
	return be.Upload(&compute.Tensor{Dims: []int{n, T}, F32: data})
}

func uploadZeros(be compute.Backend, dims ...int) (compute.Buffer, error) {
	n := 1
	for _, d := range dims {
		n *= d
	}
	return be.Upload(&compute.Tensor{Dims: dims, F32: make([]float32, n)})
}

// gdnCtx holds the forward activations needed by the backward pass.
type gdnCtx struct {
	y, x, h, qkv, z, beta, sigPre, gate, betaT compute.Buffer
	convIn, convRaw, convOut                   compute.Buffer
	qPre, kPre, v, q, k, q48, k48, out         compute.Buffer
	norm, zr, siluZ, gated, flat               compute.Buffer
	x1, h2, fgPre, up, siluFG, act, ff         compute.Buffer
	uQKV, uZ, uOut, uFG, uFU, uFD              compute.Buffer
	dtB, aB, zeroState                         compute.Buffer
}

// Forward runs the block and returns the output in ctx.y.
func (d *DeviceGDN) Forward(x compute.Buffer, positions []int32) (*gdnCtx, error) {
	be := d.be
	cfg := d.cfg
	T := x.Dims()[1]
	headVDim := cfg.HeadVDim()
	nV := cfg.DtRank
	nK := cfg.GroupCount
	headKDim := cfg.DState
	keyDim := cfg.KeyDim()
	convDim := cfg.ConvDim()
	ctx := &gdnCtx{x: x}
	var err error

	if ctx.dtB, err = constVec(be, d.ssdt, T); err != nil {
		return nil, err
	}
	if ctx.aB, err = constVec(be, d.ssma, T); err != nil {
		return nil, err
	}
	if ctx.zeroState, err = uploadZeros(be, headVDim, headVDim, nV); err != nil {
		return nil, err
	}

	if ctx.h, err = be.RMSNorm(x, d.attnNormB, d.eps); err != nil {
		return nil, err
	}
	if ctx.qkv, ctx.uQKV, err = d.qkv.forward(ctx.h); err != nil {
		return nil, err
	}
	if ctx.z, ctx.uZ, err = d.gate.forward(ctx.h); err != nil {
		return nil, err
	}
	betaRaw, err := be.MatMulWeight(d.betaB, ctx.h)
	if err != nil {
		return nil, err
	}
	if ctx.beta, err = be.Unary(compute.UnarySigmoid, betaRaw); err != nil {
		return nil, err
	}
	alpha, err := be.MatMulWeight(d.alphaB, ctx.h)
	if err != nil {
		return nil, err
	}
	pre, err := be.Binary(compute.BinaryAdd, alpha, ctx.dtB)
	if err != nil {
		return nil, err
	}
	if ctx.sigPre, err = be.Unary(compute.UnarySigmoid, pre); err != nil {
		return nil, err
	}
	sp, err := be.Unary(compute.UnarySoftplus, pre)
	if err != nil {
		return nil, err
	}
	gateFlat, err := be.Binary(compute.BinaryMul, sp, ctx.aB)
	if err != nil {
		return nil, err
	}
	if ctx.gate, err = be.Reshape(gateFlat, []int{1, nV, T}); err != nil {
		return nil, err
	}
	if ctx.betaT, err = be.Reshape(ctx.beta, []int{1, nV, T}); err != nil {
		return nil, err
	}
	if ctx.convIn, err = be.ConvInput(ctx.qkv, cfg.DConv, convDim, T); err != nil {
		return nil, err
	}
	if ctx.convRaw, err = be.SSMConv(ctx.convIn, d.conv1dB); err != nil {
		return nil, err
	}
	if ctx.convOut, err = be.Unary(compute.UnarySilu, ctx.convRaw); err != nil {
		return nil, err
	}
	if ctx.qPre, err = be.GatherHeads(ctx.convOut, 0, headKDim, nK, T, convDim); err != nil {
		return nil, err
	}
	if ctx.kPre, err = be.GatherHeads(ctx.convOut, keyDim, headKDim, nK, T, convDim); err != nil {
		return nil, err
	}
	if ctx.v, err = be.GatherHeads(ctx.convOut, 2*keyDim, headVDim, nV, T, convDim); err != nil {
		return nil, err
	}
	if ctx.q, err = be.L2Norm(ctx.qPre, 1e-6); err != nil {
		return nil, err
	}
	if ctx.k, err = be.L2Norm(ctx.kPre, 1e-6); err != nil {
		return nil, err
	}
	if ctx.q48, err = be.RepeatHeads(ctx.q, headKDim, nK, nV, T); err != nil {
		return nil, err
	}
	if ctx.k48, err = be.RepeatHeads(ctx.k, headKDim, nK, nV, T); err != nil {
		return nil, err
	}
	if ctx.out, _, err = be.GatedDeltaNet(ctx.q48, ctx.k48, ctx.v, ctx.gate, ctx.betaT, ctx.zeroState); err != nil {
		return nil, err
	}
	if ctx.norm, err = be.RMSNorm(ctx.out, d.ssmNormB, d.eps); err != nil {
		return nil, err
	}
	if ctx.zr, err = be.Reshape(ctx.z, []int{headVDim, nV, T}); err != nil {
		return nil, err
	}
	if ctx.siluZ, err = be.Unary(compute.UnarySilu, ctx.zr); err != nil {
		return nil, err
	}
	if ctx.gated, err = be.Binary(compute.BinaryMul, ctx.norm, ctx.siluZ); err != nil {
		return nil, err
	}
	if ctx.flat, err = be.Reshape(ctx.gated, []int{cfg.ValueDim(), T}); err != nil {
		return nil, err
	}
	var linOut compute.Buffer
	if linOut, ctx.uOut, err = d.out.forward(ctx.flat); err != nil {
		return nil, err
	}
	if ctx.x1, err = be.Binary(compute.BinaryAdd, x, linOut); err != nil {
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

// gdnGrads holds the adapter gradients and block input gradient.
type gdnGrads struct {
	DX                compute.Buffer
	DQKV, DOUT, DGate [2]compute.Buffer
	DFG, DFU, DFD     [2]compute.Buffer
}

// Backward propagates dY through the GatedDeltaNet block.
func (d *DeviceGDN) Backward(ctx *gdnCtx, dY compute.Buffer) (*gdnGrads, error) {
	be := d.be
	cfg := d.cfg
	T := dY.Dims()[1]
	headVDim := cfg.HeadVDim()
	nV := cfg.DtRank
	nK := cfg.GroupCount
	headKDim := cfg.DState
	keyDim := cfg.KeyDim()
	convDim := cfg.ConvDim()
	g := &gdnGrads{}
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

	// SSMOut: x1 = x + SSMOut*flat.
	var dFlat compute.Buffer
	if dFlat, dA, dB, err = d.out.backward(ctx.flat, ctx.uOut, dX1); err != nil {
		return nil, err
	}
	g.DOUT = [2]compute.Buffer{dA, dB}
	dGated, err := be.Reshape(dFlat, []int{headVDim, nV, T})
	if err != nil {
		return nil, err
	}
	dNorm, err := be.Binary(compute.BinaryMul, dGated, ctx.siluZ)
	if err != nil {
		return nil, err
	}
	dSiluZ, err := be.Binary(compute.BinaryMul, dGated, ctx.norm)
	if err != nil {
		return nil, err
	}
	dZr, err := be.SiluBack(ctx.zr, dSiluZ)
	if err != nil {
		return nil, err
	}
	dzFlat, err := be.Reshape(dZr, []int{cfg.ValueDim(), T})
	if err != nil {
		return nil, err
	}
	var dHz compute.Buffer
	if dHz, dA, dB, err = d.gate.backward(ctx.h, ctx.uZ, dzFlat); err != nil {
		return nil, err
	}
	g.DGate = [2]compute.Buffer{dA, dB}

	// GatedDeltaNet backward.
	dOut, err := be.RMSNormBack(ctx.out, d.ssmNormB, dNorm, d.eps)
	if err != nil {
		return nil, err
	}
	zerosNew, err := uploadZeros(be, headVDim, headVDim, nV)
	if err != nil {
		return nil, err
	}
	var dQ48, dK48, dV, dGate, dBeta compute.Buffer
	if cb, ok := be.(chunkedGDNBackend); ok {
		dQ48, dK48, dV, dGate, dBeta, _, err = cb.GatedDeltaNetChunkedBackward(
			ctx.q48, ctx.k48, ctx.v, ctx.gate, ctx.betaT, ctx.zeroState, dOut, zerosNew, deviceGDNChunk)
	} else {
		dQ48, dK48, dV, dGate, dBeta, _, err = be.GatedDeltaNetBackward(
			ctx.q48, ctx.k48, ctx.v, ctx.gate, ctx.betaT, ctx.zeroState, dOut, zerosNew)
	}
	if err != nil {
		return nil, err
	}
	// gate = softplus(pre) * A; pre = alpha + dt.
	dGate2, err := be.Reshape(dGate, []int{cfg.DtRank, T})
	if err != nil {
		return nil, err
	}
	dPreA, err := be.Binary(compute.BinaryMul, dGate2, ctx.aB)
	if err != nil {
		return nil, err
	}
	dPre, err := be.Binary(compute.BinaryMul, dPreA, ctx.sigPre)
	if err != nil {
		return nil, err
	}
	dHalpha, err := be.MatMulWeightTranspose(d.alphaB, dPre)
	if err != nil {
		return nil, err
	}
	dBeta2, err := be.Reshape(dBeta, []int{cfg.DtRank, T})
	if err != nil {
		return nil, err
	}
	dHbeta, err := be.MatMulWeightTranspose(d.betaB, dBeta2)
	if err != nil {
		return nil, err
	}

	// Repeat / L2 / gather / conv / qkv.
	dQ, err := be.RepeatHeadsBack(dQ48, headKDim, nK, nV, T)
	if err != nil {
		return nil, err
	}
	dK, err := be.RepeatHeadsBack(dK48, headKDim, nK, nV, T)
	if err != nil {
		return nil, err
	}
	dQpre, err := be.L2NormBack(ctx.qPre, dQ, 1e-6)
	if err != nil {
		return nil, err
	}
	dKpre, err := be.L2NormBack(ctx.kPre, dK, 1e-6)
	if err != nil {
		return nil, err
	}
	dCOq, err := be.GatherHeadsBack(dQpre, 0, headKDim, nK, T, convDim)
	if err != nil {
		return nil, err
	}
	dCOk, err := be.GatherHeadsBack(dKpre, keyDim, headKDim, nK, T, convDim)
	if err != nil {
		return nil, err
	}
	dCOv, err := be.GatherHeadsBack(dV, 2*keyDim, headVDim, nV, T, convDim)
	if err != nil {
		return nil, err
	}
	dCO, err := be.Binary(compute.BinaryAdd, dCOq, dCOk)
	if err != nil {
		return nil, err
	}
	if dCO, err = be.Binary(compute.BinaryAdd, dCO, dCOv); err != nil {
		return nil, err
	}
	dRaw, err := be.SiluBack(ctx.convRaw, dCO)
	if err != nil {
		return nil, err
	}
	dConvIn, _, err := be.SSMConvBack(ctx.convIn, d.conv1dB, dRaw)
	if err != nil {
		return nil, err
	}
	dQKV, err := be.ConvInputBack(dConvIn, cfg.DConv, convDim, T)
	if err != nil {
		return nil, err
	}
	dHqkv, dA, dB, err := d.qkv.backward(ctx.h, ctx.uQKV, dQKV)
	if err != nil {
		return nil, err
	}
	g.DQKV = [2]compute.Buffer{dA, dB}
	dH, err := be.Binary(compute.BinaryAdd, dHqkv, dHz)
	if err != nil {
		return nil, err
	}
	if dH, err = be.Binary(compute.BinaryAdd, dH, dHalpha); err != nil {
		return nil, err
	}
	if dH, err = be.Binary(compute.BinaryAdd, dH, dHbeta); err != nil {
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
