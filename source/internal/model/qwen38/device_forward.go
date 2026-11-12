package qwen38

import (
	"fmt"
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

func (m *Model) forwardDevice(tokens []int32, opts ForwardOptions) (*ForwardResult, error) {
	cfg := m.Cfg
	be := m.Backend
	d := m.device
	st := opts.State
	if st == nil || !st.deviceForward || st.backend != be {
		return nil, fmt.Errorf("qwen38: device forward requires a matching device state")
	}
	stBackend, hasLastToken := be.(DeviceForwardBackend)
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
	var lastHidden []*compute.Tensor
	captureLast := func(buf compute.Buffer) error {
		if !opts.RecordLastHidden {
			return nil
		}
		if !hasLastToken {
			return fmt.Errorf("qwen38: backend lacks LastToken")
		}
		lt, err := stBackend.LastToken(buf)
		if err != nil {
			return err
		}
		t, err := be.Download(lt)
		be.Free(lt)
		if err != nil {
			return err
		}
		lastHidden = append(lastHidden, t)
		return nil
	}
	if err := captureLast(x); err != nil {
		be.Free(x)
		return nil, err
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

	h, err := be.RMSNorm(x, m.layerNormWeight(0), cfg.RmsEps)
	if err != nil {
		be.Free(x)
		return nil, err
	}

	for il := 0; il < cfg.TrunkLayers(); il++ {
		nextNorm := m.layerNormWeight(il + 1)
		be.BeginScope()
		var y, hnext, newState compute.Buffer
		if cfg.IsRecurrent(il) {
			y, hnext, newState, err = m.deviceGDNBlock(il, d.layers[il].gdn, h, x, nextNorm, st)
		} else {
			y, hnext, err = m.deviceAttnBlock(il, d.layers[il].attn, h, x, nextNorm, positions, st)
		}
		if err != nil {
			be.EndScope()
			be.Free(x)
			be.Free(h)
			return nil, fmt.Errorf("qwen38: layer %d device forward: %w", il, err)
		}
		if newState != nil {
			oldState := st.DeviceSSM[il]
			st.DeviceSSM[il] = newState
			be.Free(oldState)
			be.EndScope(y, hnext, newState)
		} else {
			be.EndScope(y, hnext)
		}
		be.Free(x)
		be.Free(h)
		x, h = y, hnext
		// Long prefill layers can keep the GPU busy long enough for the
		// driver's watchdog to soft-recover the context. Submit each layer
		// separately; decode (T == 1) stays fully batched.
		if T > 1 {
			if err := be.Sync(); err != nil {
				be.Free(x)
				be.Free(h)
				return nil, fmt.Errorf("qwen38: layer %d sync: %w", il, err)
			}
		}
		if err := captureLast(x); err != nil {
			be.Free(x)
			be.Free(h)
			return nil, err
		}
	}

	be.BeginScope()
	xn := h
	if opts.LastTokenOnly && T > 1 {
		if !hasLastToken {
			be.EndScope()
			be.Free(x)
			be.Free(h)
			return nil, fmt.Errorf("qwen38: backend lacks LastToken")
		}
		xn, err = stBackend.LastToken(xn)
		if err != nil {
			be.EndScope()
			be.Free(x)
			be.Free(h)
			return nil, err
		}
	}
	logitsBuf, err := be.MatMulWeight(d.lmHead, xn)
	if err != nil {
		be.EndScope()
		be.Free(x)
		be.Free(h)
		return nil, err
	}
	logits, err := be.Download(logitsBuf)
	be.EndScope()
	be.Free(x)
	if err != nil {
		be.Free(h)
		return nil, err
	}
	res := &ForwardResult{Logits: logits, State: st, LastHidden: lastHidden}
	if opts.CaptureHidden {
		res.DeviceHidden = h
	} else {
		be.Free(h)
	}
	return res, nil
}

func (m *Model) layerNormWeight(il int) compute.Buffer {
	d := m.device
	if il >= m.Cfg.TrunkLayers() {
		return d.finalNorm
	}
	if m.Cfg.IsRecurrent(il) {
		return d.layers[il].gdn.norm
	}
	return d.layers[il].attn.norm
}

func (m *Model) deviceAttnBlock(il int, lw *deviceAttnLayer, h, x, nextNorm compute.Buffer, positions []int32, st *State) (compute.Buffer, compute.Buffer, error) {
	return m.deviceAttnBlockCache(il, lw, h, x, nextNorm, positions, st, deviceCache(st, il))
}

func (m *Model) deviceAttnBlockCache(il int, lw *deviceAttnLayer, h, x, nextNorm compute.Buffer, positions []int32, st *State, dc *DeviceKVCache) (compute.Buffer, compute.Buffer, error) {
	cfg := m.Cfg
	be := m.Backend
	rb := be.(DeviceForwardBackend)
	T := x.Dims()[1]
	hd := cfg.HeadDim
	nHead := cfg.NHead

	qg, err := be.MatMulWeight(lw.q, h)
	if err != nil {
		return nil, nil, err
	}
	qSplit, gateSig, err := rb.SplitQGSigmoid(qg, hd, nHead, T)
	if err != nil {
		return nil, nil, err
	}
	kBuf, err := be.MatMulWeight(lw.k, h)
	if err != nil {
		return nil, nil, err
	}
	vBuf, err := be.MatMulWeight(lw.v, h)
	if err != nil {
		return nil, nil, err
	}
	kPre, err := be.Reshape(kBuf, []int{hd, cfg.NHeadKV, T})
	if err != nil {
		return nil, nil, err
	}
	vn, err := be.Reshape(vBuf, []int{hd, cfg.NHeadKV, T})
	if err != nil {
		return nil, nil, err
	}
	qN, err := be.RMSNorm(qSplit, lw.qNorm, cfg.RmsEps)
	if err != nil {
		return nil, nil, err
	}
	kN, err := be.RMSNorm(kPre, lw.kNorm, cfg.RmsEps)
	if err != nil {
		return nil, nil, err
	}
	rope := cfg.RopeScalingOptions()
	qR, err := rb.RoPEScaled(qN, positions, cfg.RopeTheta, cfg.NRot, rope)
	if err != nil {
		return nil, nil, err
	}
	kR, err := rb.RoPEScaled(kN, positions, cfg.RopeTheta, cfg.NRot, rope)
	if err != nil {
		return nil, nil, err
	}
	if dc == nil || st.kvBackend == nil {
		return nil, nil, fmt.Errorf("device KV cache missing")
	}
	if dc.Len+T > dc.Cap {
		return nil, nil, fmt.Errorf("device KV cache capacity %d exceeded (%d + %d)", dc.Cap, dc.Len, T)
	}
	if err := st.kvBackend.AppendKV(kR, vn, dc.K, dc.V, dc.Len); err != nil {
		return nil, nil, err
	}
	dc.Len += T
	scale := float32(1 / math.Sqrt(float64(hd)))
	if cfg.AttnScale > 0 {
		scale *= cfg.AttnScale
	}
	attn0, err := st.kvBackend.AttentionCached(qR, dc.K, dc.V, nHead, cfg.NHeadKV, dc.Len, scale)
	if err != nil {
		return nil, nil, err
	}
	attn, err := be.Binary(compute.BinaryMul, attn0, gateSig)
	if err != nil {
		return nil, nil, err
	}
	attnFlat, err := be.Reshape(attn, []int{nHead * hd, T})
	if err != nil {
		return nil, nil, err
	}
	o, err := be.MatMulWeight(lw.o, attnFlat)
	if err != nil {
		return nil, nil, err
	}
	x1, h2, err := rb.RMSNormAdd(x, o, lw.postNorm, cfg.RmsEps)
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
	act, err := rb.GatedSilu(fg, up)
	if err != nil {
		return nil, nil, err
	}
	ff, err := be.MatMulWeight(lw.fd, act)
	if err != nil {
		return nil, nil, err
	}
	y, hnext, err := rb.RMSNormAdd(x1, ff, nextNorm, cfg.RmsEps)
	if err != nil {
		return nil, nil, err
	}
	return y, hnext, nil
}

func (m *Model) deviceGDNBlock(il int, lw *deviceGDNLayer, h, x, nextNorm compute.Buffer, st *State) (compute.Buffer, compute.Buffer, compute.Buffer, error) {
	cfg := m.Cfg
	be := m.Backend
	rb := be.(DeviceForwardBackend)
	T := x.Dims()[1]
	headVDim := cfg.HeadVDim()
	nV := cfg.DtRank
	nK := cfg.GroupCount
	headKDim := cfg.DState
	convDim := cfg.ConvDim()
	var err error

	var qkv, z, betaRaw, alpha compute.Buffer
	var qkvBase, betaBase compute.Buffer
	if T == 1 && lw.qkvGate != nil {
		qkvBase, err = be.MatMulWeight(lw.qkvGate, h)
		if err != nil {
			return nil, nil, nil, err
		}
		qkv, err = rb.SliceF32(qkvBase, 0, []int{convDim, 1})
		if err != nil {
			be.Free(qkvBase)
			return nil, nil, nil, err
		}
		z, err = rb.SliceF32(qkvBase, convDim*4, []int{cfg.ValueDim(), 1})
		if err != nil {
			be.Free(qkvBase)
			return nil, nil, nil, err
		}
	} else {
		qkv, err = be.MatMulWeight(lw.qkv, h)
		if err != nil {
			return nil, nil, nil, err
		}
		z, err = be.MatMulWeight(lw.gate, h)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	if T == 1 && lw.betaAlpha != nil {
		betaBase, err = be.MatMulWeight(lw.betaAlpha, h)
		if err != nil {
			be.Free(qkvBase)
			return nil, nil, nil, err
		}
		betaRaw, err = rb.SliceF32(betaBase, 0, []int{nV, 1})
		if err != nil {
			be.Free(qkvBase)
			be.Free(betaBase)
			return nil, nil, nil, err
		}
		alpha, err = rb.SliceF32(betaBase, nV*4, []int{nV, 1})
		if err != nil {
			be.Free(qkvBase)
			be.Free(betaBase)
			return nil, nil, nil, err
		}
	} else {
		betaRaw, err = be.MatMulWeight(lw.beta, h)
		if err != nil {
			be.Free(qkvBase)
			return nil, nil, nil, err
		}
		alpha, err = be.MatMulWeight(lw.alpha, h)
		if err != nil {
			be.Free(qkvBase)
			return nil, nil, nil, err
		}
	}
	gate, beta, err := rb.GDNGateBeta(alpha, betaRaw, lw.ssmDt, lw.ssmA, nV, T)
	if err != nil {
		be.Free(qkvBase)
		be.Free(betaBase)
		return nil, nil, nil, err
	}
	if betaBase != nil {
		be.Free(betaBase)
	}
	betaT, err := be.Reshape(beta, []int{1, nV, T})
	if err != nil {
		return nil, nil, nil, err
	}
	convIn, err := rb.ConvInputState(qkv, st.DeviceConv[il], cfg.DConv, convDim, T)
	if err != nil {
		be.Free(qkvBase)
		return nil, nil, nil, err
	}
	convOut, err := rb.SSMConvSilu(convIn, lw.conv1d)
	if err != nil {
		return nil, nil, nil, err
	}
	q, k, v, err := rb.GatherHeads3Norm(convOut, headKDim, nK, headVDim, nV, T, convDim, 1e-6)
	if err != nil {
		return nil, nil, nil, err
	}
	out, newState, err := rb.GatedDeltaNetQK(q, k, v, gate, betaT, st.DeviceSSM[il], nK)
	if err != nil {
		return nil, nil, nil, err
	}
	zr, err := be.Reshape(z, []int{headVDim, nV, T})
	if err != nil {
		return nil, nil, nil, err
	}
	gated, err := rb.NormSiluMul(out, lw.ssmNorm, zr, cfg.RmsEps)
	if qkvBase != nil {
		be.Free(qkvBase)
	}
	if err != nil {
		return nil, nil, nil, err
	}
	flat, err := be.Reshape(gated, []int{cfg.ValueDim(), T})
	if err != nil {
		return nil, nil, nil, err
	}
	linOut, err := be.MatMulWeight(lw.out, flat)
	if err != nil {
		return nil, nil, nil, err
	}
	x1, h2, err := rb.RMSNormAdd(x, linOut, lw.postNorm, cfg.RmsEps)
	if err != nil {
		return nil, nil, nil, err
	}
	fg, err := be.MatMulWeight(lw.fg, h2)
	if err != nil {
		return nil, nil, nil, err
	}
	up, err := be.MatMulWeight(lw.fu, h2)
	if err != nil {
		return nil, nil, nil, err
	}
	act, err := rb.GatedSilu(fg, up)
	if err != nil {
		return nil, nil, nil, err
	}
	ff, err := be.MatMulWeight(lw.fd, act)
	if err != nil {
		return nil, nil, nil, err
	}
	y, hnext, err := rb.RMSNormAdd(x1, ff, nextNorm, cfg.RmsEps)
	if err != nil {
		return nil, nil, nil, err
	}
	return y, hnext, newState, nil
}
