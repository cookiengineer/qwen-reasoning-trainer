package qwen38

import (
	"fmt"
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

func uploadConstVec(be compute.Backend, v []float32, T int) (compute.Buffer, error) {
	if len(v) == 0 || T <= 0 {
		return nil, fmt.Errorf("qwen38: invalid constant vector")
	}
	data := make([]float32, len(v)*T)
	for t := 0; t < T; t++ {
		copy(data[t*len(v):], v)
	}
	return be.Upload(&compute.Tensor{Dims: []int{len(v), T}, F32: data})
}

func (m *Model) forwardDevice(tokens []int32, opts ForwardOptions) (*ForwardResult, error) {
	cfg := m.Cfg
	be := m.Backend
	d := m.device
	st := opts.State
	if st == nil || !st.deviceForward || st.backend != be {
		return nil, fmt.Errorf("qwen38: device forward requires a matching device state")
	}
	T := len(tokens)
	if T == 0 {
		return nil, fmt.Errorf("qwen38: empty prompt")
	}
	x, err := be.GetRowsWeight(d.embed, tokens)
	if err != nil {
		return nil, err
	}
	if opts.RecordHidden {
		// Hidden capture is only used by the host diagnostic path.
		be.Free(x)
		return nil, fmt.Errorf("qwen38: RecordHidden is not supported on the device path")
	}

	positions := opts.Positions
	if positions == nil {
		positions = make([]int32, T)
		for i := range positions {
			positions[i] = int32(i)
		}
	}
	if len(positions) != T {
		be.Free(x)
		return nil, fmt.Errorf("qwen38: positions length %d != tokens %d", len(positions), T)
	}

	for il := 0; il < cfg.TrunkLayers(); il++ {
		be.BeginScope()
		var y, newState compute.Buffer
		if cfg.IsRecurrent(il) {
			y, newState, err = m.deviceGDNBlock(il, d.layers[il].gdn, x, st)
		} else {
			y, err = m.deviceAttnBlock(il, d.layers[il].attn, x, positions, st)
		}
		if err != nil {
			be.EndScope()
			be.Free(x)
			return nil, fmt.Errorf("qwen38: layer %d device forward: %w", il, err)
		}
		oldX := x
		x = y
		if cfg.IsRecurrent(il) && newState != nil {
			oldState := st.DeviceSSM[il]
			st.DeviceSSM[il] = newState
			be.Free(oldState)
			be.EndScope(y, newState)
		} else {
			be.EndScope(y)
		}
		if oldX != nil {
			be.Free(oldX)
		}
		// Long prefill layers can keep the GPU busy long enough for the
		// driver's watchdog to soft-recover the context. Submit each layer
		// separately; decode (T == 1) stays fully batched.
		if T > 1 {
			if err := be.Sync(); err != nil {
				be.Free(x)
				return nil, fmt.Errorf("qwen38: layer %d sync: %w", il, err)
			}
		}
	}

	be.BeginScope()
	xn, err := be.RMSNorm(x, d.finalNorm, cfg.RmsEps)
	if err != nil {
		be.EndScope()
		be.Free(x)
		return nil, err
	}
	if opts.LastTokenOnly && T > 1 {
		rb, ok := be.(DeviceForwardBackend)
		if !ok {
			be.EndScope()
			be.Free(x)
			return nil, fmt.Errorf("qwen38: backend lacks LastToken")
		}
		xn, err = rb.LastToken(xn)
		if err != nil {
			be.EndScope()
			be.Free(x)
			return nil, err
		}
	}
	logitsBuf, err := be.MatMulWeight(d.lmHead, xn)
	if err != nil {
		be.EndScope()
		be.Free(x)
		return nil, err
	}
	logits, err := be.Download(logitsBuf)
	be.EndScope()
	be.Free(x)
	if err != nil {
		return nil, err
	}
	return &ForwardResult{Logits: logits, State: st}, nil
}

func (m *Model) deviceAttnBlock(il int, lw *deviceAttnLayer, x compute.Buffer, positions []int32, st *State) (compute.Buffer, error) {
	cfg := m.Cfg
	be := m.Backend
	rb := be.(DeviceForwardBackend)
	T := x.Dims()[1]
	hd := cfg.HeadDim
	nHead := cfg.NHead

	h, err := be.RMSNorm(x, lw.norm, cfg.RmsEps)
	if err != nil {
		return nil, err
	}
	qg, err := be.MatMulWeight(lw.q, h)
	if err != nil {
		return nil, err
	}
	qSplit, gate, err := be.SplitQG(qg, hd, nHead, T)
	if err != nil {
		return nil, err
	}
	gateSig, err := be.Unary(compute.UnarySigmoid, gate)
	if err != nil {
		return nil, err
	}
	kBuf, err := be.MatMulWeight(lw.k, h)
	if err != nil {
		return nil, err
	}
	vBuf, err := be.MatMulWeight(lw.v, h)
	if err != nil {
		return nil, err
	}
	kPre, err := be.Reshape(kBuf, []int{hd, cfg.NHeadKV, T})
	if err != nil {
		return nil, err
	}
	vn, err := be.Reshape(vBuf, []int{hd, cfg.NHeadKV, T})
	if err != nil {
		return nil, err
	}
	qN, err := be.RMSNorm(qSplit, lw.qNorm, cfg.RmsEps)
	if err != nil {
		return nil, err
	}
	kN, err := be.RMSNorm(kPre, lw.kNorm, cfg.RmsEps)
	if err != nil {
		return nil, err
	}
	rope := cfg.RopeScalingOptions()
	qR, err := rb.RoPEScaled(qN, positions, cfg.RopeTheta, cfg.NRot, rope)
	if err != nil {
		return nil, err
	}
	kR, err := rb.RoPEScaled(kN, positions, cfg.RopeTheta, cfg.NRot, rope)
	if err != nil {
		return nil, err
	}
	dc := deviceCache(st, il)
	if dc == nil || st.kvBackend == nil {
		return nil, fmt.Errorf("device KV cache missing")
	}
	if dc.Len+T > dc.Cap {
		return nil, fmt.Errorf("device KV cache capacity %d exceeded (%d + %d)", dc.Cap, dc.Len, T)
	}
	if err := st.kvBackend.AppendKV(kR, vn, dc.K, dc.V, dc.Len); err != nil {
		return nil, err
	}
	dc.Len += T
	scale := float32(1 / math.Sqrt(float64(hd)))
	if cfg.AttnScale > 0 {
		scale *= cfg.AttnScale
	}
	attn0, err := st.kvBackend.AttentionCached(qR, dc.K, dc.V, nHead, cfg.NHeadKV, dc.Len, scale)
	if err != nil {
		return nil, err
	}
	attn, err := be.Binary(compute.BinaryMul, attn0, gateSig)
	if err != nil {
		return nil, err
	}
	attnFlat, err := be.Reshape(attn, []int{nHead * hd, T})
	if err != nil {
		return nil, err
	}
	o, err := be.MatMulWeight(lw.o, attnFlat)
	if err != nil {
		return nil, err
	}
	x1, err := be.Binary(compute.BinaryAdd, x, o)
	if err != nil {
		return nil, err
	}
	h2, err := be.RMSNorm(x1, lw.postNorm, cfg.RmsEps)
	if err != nil {
		return nil, err
	}
	fg, err := be.MatMulWeight(lw.fg, h2)
	if err != nil {
		return nil, err
	}
	up, err := be.MatMulWeight(lw.fu, h2)
	if err != nil {
		return nil, err
	}
	silu, err := be.Unary(compute.UnarySilu, fg)
	if err != nil {
		return nil, err
	}
	act, err := be.Binary(compute.BinaryMul, silu, up)
	if err != nil {
		return nil, err
	}
	ff, err := be.MatMulWeight(lw.fd, act)
	if err != nil {
		return nil, err
	}
	return be.Binary(compute.BinaryAdd, x1, ff)
}

func (m *Model) deviceGDNBlock(il int, lw *deviceGDNLayer, x compute.Buffer, st *State) (compute.Buffer, compute.Buffer, error) {
	cfg := m.Cfg
	be := m.Backend
	rb := be.(DeviceForwardBackend)
	T := x.Dims()[1]
	headVDim := cfg.HeadVDim()
	nV := cfg.DtRank
	nK := cfg.GroupCount
	headKDim := cfg.DState
	keyDim := cfg.KeyDim()
	convDim := cfg.ConvDim()
	src := m.W.Layers[il]

	h, err := be.RMSNorm(x, lw.norm, cfg.RmsEps)
	if err != nil {
		return nil, nil, err
	}
	qkv, err := be.MatMulWeight(lw.qkv, h)
	if err != nil {
		return nil, nil, err
	}
	z, err := be.MatMulWeight(lw.gate, h)
	if err != nil {
		return nil, nil, err
	}
	betaRaw, err := be.MatMulWeight(lw.beta, h)
	if err != nil {
		return nil, nil, err
	}
	beta, err := be.Unary(compute.UnarySigmoid, betaRaw)
	if err != nil {
		return nil, nil, err
	}
	alpha, err := be.MatMulWeight(lw.alpha, h)
	if err != nil {
		return nil, nil, err
	}
	dtB, err := uploadConstVec(be, src.SSMDt, T)
	if err != nil {
		return nil, nil, err
	}
	aB, err := uploadConstVec(be, src.SSMA, T)
	if err != nil {
		return nil, nil, err
	}
	pre, err := be.Binary(compute.BinaryAdd, alpha, dtB)
	if err != nil {
		return nil, nil, err
	}
	sp, err := be.Unary(compute.UnarySoftplus, pre)
	if err != nil {
		return nil, nil, err
	}
	gateFlat, err := be.Binary(compute.BinaryMul, sp, aB)
	if err != nil {
		return nil, nil, err
	}
	gate, err := be.Reshape(gateFlat, []int{1, nV, T})
	if err != nil {
		return nil, nil, err
	}
	betaT, err := be.Reshape(beta, []int{1, nV, T})
	if err != nil {
		return nil, nil, err
	}
	convIn, err := rb.ConvInputState(qkv, st.DeviceConv[il], cfg.DConv, convDim, T)
	if err != nil {
		return nil, nil, err
	}
	convRaw, err := be.SSMConv(convIn, lw.conv1d)
	if err != nil {
		return nil, nil, err
	}
	convOut, err := be.Unary(compute.UnarySilu, convRaw)
	if err != nil {
		return nil, nil, err
	}
	qPre, err := be.GatherHeads(convOut, 0, headKDim, nK, T, convDim)
	if err != nil {
		return nil, nil, err
	}
	kPre, err := be.GatherHeads(convOut, keyDim, headKDim, nK, T, convDim)
	if err != nil {
		return nil, nil, err
	}
	v, err := be.GatherHeads(convOut, 2*keyDim, headVDim, nV, T, convDim)
	if err != nil {
		return nil, nil, err
	}
	q, err := be.L2Norm(qPre, 1e-6)
	if err != nil {
		return nil, nil, err
	}
	k, err := be.L2Norm(kPre, 1e-6)
	if err != nil {
		return nil, nil, err
	}
	q48, err := be.RepeatHeads(q, headKDim, nK, nV, T)
	if err != nil {
		return nil, nil, err
	}
	k48, err := be.RepeatHeads(k, headKDim, nK, nV, T)
	if err != nil {
		return nil, nil, err
	}
	out, newState, err := be.GatedDeltaNet(q48, k48, v, gate, betaT, st.DeviceSSM[il])
	if err != nil {
		return nil, nil, err
	}
	norm, err := be.RMSNorm(out, lw.ssmNorm, cfg.RmsEps)
	if err != nil {
		return nil, nil, err
	}
	zr, err := be.Reshape(z, []int{headVDim, nV, T})
	if err != nil {
		return nil, nil, err
	}
	siluZ, err := be.Unary(compute.UnarySilu, zr)
	if err != nil {
		return nil, nil, err
	}
	gated, err := be.Binary(compute.BinaryMul, norm, siluZ)
	if err != nil {
		return nil, nil, err
	}
	flat, err := be.Reshape(gated, []int{cfg.ValueDim(), T})
	if err != nil {
		return nil, nil, err
	}
	linOut, err := be.MatMulWeight(lw.out, flat)
	if err != nil {
		return nil, nil, err
	}
	x1, err := be.Binary(compute.BinaryAdd, x, linOut)
	if err != nil {
		return nil, nil, err
	}
	h2, err := be.RMSNorm(x1, lw.postNorm, cfg.RmsEps)
	if err != nil {
		return nil, nil, err
	}
	fg, err := be.MatMulWeight(lw.fg, h2)
	if err != nil {
		return nil, nil, err
	}
	up, err := be.MatMulWeight(lw.fu, h2)
	if err != nil {
		return nil, nil, err
	}
	silu, err := be.Unary(compute.UnarySilu, fg)
	if err != nil {
		return nil, nil, err
	}
	act, err := be.Binary(compute.BinaryMul, silu, up)
	if err != nil {
		return nil, nil, err
	}
	ff, err := be.MatMulWeight(lw.fd, act)
	if err != nil {
		return nil, nil, err
	}
	y, err := be.Binary(compute.BinaryAdd, x1, ff)
	if err != nil {
		return nil, nil, err
	}
	return y, newState, nil
}
