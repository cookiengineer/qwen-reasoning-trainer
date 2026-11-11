package qwen38

import (
	"fmt"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// DeviceForwardBackend is the optional backend surface used by the fully
// device-resident inference path.
type DeviceForwardBackend interface {
	KVCacheBackend
	RoPEScaled(a compute.Buffer, positions []int32, theta float64, nDims int, sc compute.RoPEScaling) (compute.Buffer, error)
	ConvInputState(qkv, hist compute.Buffer, dConv, convDim, T int) (compute.Buffer, error)
	LastToken(x compute.Buffer) (compute.Buffer, error)
	SplitQGSigmoid(qg compute.Buffer, hd, nHead, T int) (q, gate compute.Buffer, err error)
	GatedSilu(gate, up compute.Buffer) (compute.Buffer, error)
	GDNGate(alpha, dt, a compute.Buffer, nV, T int) (compute.Buffer, error)
	GDNGateBeta(alpha, betaRaw, dt, a compute.Buffer, nV, T int) (gate, beta compute.Buffer, err error)
	RMSNormAdd(a, add, w compute.Buffer, eps float32) (sum, norm compute.Buffer, err error)
	GatedDeltaNetQK(q, k, v, g, beta, state compute.Buffer, nK int) (out, newState compute.Buffer, err error)
	SSMConvSilu(sx, c compute.Buffer) (compute.Buffer, error)
	GatherHeads3(convOut compute.Buffer, headK, nK, headV, nV, T, convDim int) (q, k, v compute.Buffer, err error)
	GatherHeads3Norm(convOut compute.Buffer, headK, nK, headV, nV, T, convDim int, eps float32) (q, k, v compute.Buffer, err error)
	L2Norm2(a, b compute.Buffer, eps float32) (compute.Buffer, compute.Buffer, error)
	MTPConcat(emb, hiddenRows, prevH, wE, wH compute.Buffer, n, t int, eps float32) (compute.Buffer, error)
	SliceF32(buf compute.Buffer, byteOffset int, dims []int) (compute.Buffer, error)
	NormSiluMul(x, w, z compute.Buffer, eps float32) (compute.Buffer, error)
}

type deviceAttnLayer struct {
	norm, postNorm, qNorm, kNorm compute.Buffer
	q, k, v, o, fg, fu, fd       compute.Buffer
}

type deviceGDNLayer struct {
	norm, postNorm, ssmNorm, conv1d, alpha, beta compute.Buffer
	ssmDt, ssmA                                  compute.Buffer
	qkv, gate, out, fg, fu, fd                   compute.Buffer
	qkvGate, betaAlpha                           compute.Buffer
}

type deviceLayer struct {
	attn *deviceAttnLayer
	gdn  *deviceGDNLayer
}

type deviceMTP struct {
	attn                     *deviceAttnLayer
	ehProj                   compute.Buffer
	eNorm, hNorm, sharedNorm compute.Buffer
}

type deviceModel struct {
	embed, finalNorm, lmHead compute.Buffer
	layers                   []deviceLayer
	mtp                      *deviceMTP
}

func (d *deviceModel) free(be compute.Backend) {
	if d == nil {
		return
	}
	free := func(b compute.Buffer) {
		if b != nil {
			be.Free(b)
		}
	}
	free(d.embed)
	free(d.finalNorm)
	if d.lmHead != d.embed {
		free(d.lmHead)
	}
	for i := range d.layers {
		l := &d.layers[i]
		if l.attn != nil {
			a := l.attn
			free(a.norm)
			free(a.postNorm)
			free(a.qNorm)
			free(a.kNorm)
			free(a.q)
			free(a.k)
			free(a.v)
			free(a.o)
			free(a.fg)
			free(a.fu)
			free(a.fd)
		}
		if l.gdn != nil {
			g := l.gdn
			free(g.norm)
			free(g.postNorm)
			free(g.ssmNorm)
			free(g.conv1d)
			free(g.alpha)
			free(g.beta)
			free(g.ssmDt)
			free(g.ssmA)
			free(g.qkv)
			free(g.gate)
			free(g.out)
			free(g.fg)
			free(g.fu)
			free(g.fd)
			free(g.qkvGate)
			free(g.betaAlpha)
		}
	}
	if d.mtp != nil {
		m := d.mtp
		if m.attn != nil {
			a := m.attn
			free(a.norm)
			free(a.postNorm)
			free(a.qNorm)
			free(a.kNorm)
			free(a.q)
			free(a.k)
			free(a.v)
			free(a.o)
			free(a.fg)
			free(a.fu)
			free(a.fd)
		}
		free(m.ehProj)
		free(m.eNorm)
		free(m.hNorm)
		free(m.sharedNorm)
	}
	d.mtp = nil
	d.layers = nil
}

// EnableDeviceForward uploads the full inference graph once and switches the
// model to the device-resident forward path. It is a no-op for models without
// a backend.
func (m *Model) EnableDeviceForward() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.Backend == nil || m.device != nil {
		return nil
	}
	if _, ok := m.Backend.(DeviceForwardBackend); !ok {
		return fmt.Errorf("qwen38: backend does not support device forward")
	}
	be := m.Backend
	upW := func(w *Weight) (compute.Buffer, error) {
		return be.UploadWeight(w.Typ, w.Raw, w.Dims)
	}
	upT := func(t *compute.Tensor) (compute.Buffer, error) {
		return be.Upload(t)
	}
	upV := func(v []float32) (compute.Buffer, error) {
		return be.Upload(&compute.Tensor{Dims: []int{len(v)}, F32: v})
	}

	d := &deviceModel{}
	var err error
	if d.embed, err = upW(m.W.TokenEmbd); err != nil {
		d.free(be)
		return err
	}
	if d.finalNorm, err = upV(m.W.OutputNorm); err != nil {
		d.free(be)
		return err
	}
	if m.W.Output == m.W.TokenEmbd {
		d.lmHead = d.embed
	} else if d.lmHead, err = upW(m.W.Output); err != nil {
		d.free(be)
		return err
	}
	d.layers = make([]deviceLayer, len(m.W.Layers))
	for il := range m.W.Layers {
		lw := &m.W.Layers[il]
		if m.Cfg.IsRecurrent(il) {
			g := &deviceGDNLayer{}
			if g.norm, err = upV(lw.AttnNorm); err != nil {
				d.free(be)
				return err
			}
			if g.postNorm, err = upV(lw.PostAttnNorm); err != nil {
				d.free(be)
				return err
			}
			if g.ssmNorm, err = upV(lw.SSMNorm); err != nil {
				d.free(be)
				return err
			}
			if g.conv1d, err = upT(lw.SSMConv1d); err != nil {
				d.free(be)
				return err
			}
			if g.alpha, err = upW(lw.SSMAlpha); err != nil {
				d.free(be)
				return err
			}
			if g.beta, err = upW(lw.SSMBeta); err != nil {
				d.free(be)
				return err
			}
			if g.ssmDt, err = upV(lw.SSMDt); err != nil {
				d.free(be)
				return err
			}
			if g.ssmA, err = upV(lw.SSMA); err != nil {
				d.free(be)
				return err
			}
			for _, e := range []struct {
				dst *compute.Buffer
				w   *Weight
			}{
				{&g.qkv, lw.AttnQKV}, {&g.gate, lw.AttnGate}, {&g.out, lw.SSMOut},
				{&g.fg, lw.FfnGate}, {&g.fu, lw.FfnUp}, {&g.fd, lw.FfnDown},
			} {
				if *e.dst, err = upW(e.w); err != nil {
					d.free(be)
					return err
				}
			}
			if g.qkvGate, err = uploadWeightGroup(be, lw.AttnQKV, lw.AttnGate); err != nil {
				d.free(be)
				return err
			}
			if g.betaAlpha, err = uploadWeightGroup(be, lw.SSMBeta, lw.SSMAlpha); err != nil {
				d.free(be)
				return err
			}
			d.layers[il] = deviceLayer{gdn: g}
			continue
		}
		a := &deviceAttnLayer{}
		if a.norm, err = upV(lw.AttnNorm); err != nil {
			d.free(be)
			return err
		}
		if a.postNorm, err = upV(lw.PostAttnNorm); err != nil {
			d.free(be)
			return err
		}
		if a.qNorm, err = upV(lw.AttnQNorm); err != nil {
			d.free(be)
			return err
		}
		if a.kNorm, err = upV(lw.AttnKNorm); err != nil {
			d.free(be)
			return err
		}
		for _, e := range []struct {
			dst *compute.Buffer
			w   *Weight
		}{
			{&a.q, lw.AttnQ}, {&a.k, lw.AttnK}, {&a.v, lw.AttnV}, {&a.o, lw.AttnOutput},
			{&a.fg, lw.FfnGate}, {&a.fu, lw.FfnUp}, {&a.fd, lw.FfnDown},
		} {
			if *e.dst, err = upW(e.w); err != nil {
				d.free(be)
				return err
			}
		}
		d.layers[il] = deviceLayer{attn: a}
	}
	if m.W.MTP != nil {
		mtp := &deviceMTP{}
		mtp.attn = &deviceAttnLayer{}
		if mtp.attn.norm, err = upV(m.W.MTP.AttnNorm); err != nil {
			d.mtp = mtp
			d.free(be)
			return err
		}
		if mtp.attn.postNorm, err = upV(m.W.MTP.PostAttnNorm); err != nil {
			d.mtp = mtp
			d.free(be)
			return err
		}
		if mtp.attn.qNorm, err = upV(m.W.MTP.AttnQNorm); err != nil {
			d.mtp = mtp
			d.free(be)
			return err
		}
		if mtp.attn.kNorm, err = upV(m.W.MTP.AttnKNorm); err != nil {
			d.mtp = mtp
			d.free(be)
			return err
		}
		for _, e := range []struct {
			dst *compute.Buffer
			w   *Weight
		}{
			{&mtp.attn.q, m.W.MTP.AttnQ}, {&mtp.attn.k, m.W.MTP.AttnK},
			{&mtp.attn.v, m.W.MTP.AttnV}, {&mtp.attn.o, m.W.MTP.AttnOutput},
			{&mtp.attn.fg, m.W.MTP.FfnGate}, {&mtp.attn.fu, m.W.MTP.FfnUp},
			{&mtp.attn.fd, m.W.MTP.FfnDown}, {&mtp.ehProj, m.W.MTP.EHProj},
		} {
			if *e.dst, err = upW(e.w); err != nil {
				d.mtp = mtp
				d.free(be)
				return err
			}
		}
		if mtp.eNorm, err = upV(m.W.MTP.ENorm); err != nil {
			d.mtp = mtp
			d.free(be)
			return err
		}
		if mtp.hNorm, err = upV(m.W.MTP.HNorm); err != nil {
			d.mtp = mtp
			d.free(be)
			return err
		}
		if mtp.sharedNorm, err = upV(m.W.MTP.SharedNorm); err != nil {
			d.mtp = mtp
			d.free(be)
			return err
		}
		d.mtp = mtp
	}
	m.device = d
	return nil
}

// uploadWeightGroup concatenates same-type, same-K weights along their output
// dimension. It returns nil when the weights cannot share a quantized layout.
func uploadWeightGroup(be compute.Backend, ws ...*Weight) (compute.Buffer, error) {
	if len(ws) < 2 {
		return nil, nil
	}
	typ := ws[0].Typ
	k := ws[0].In()
	total := 0
	for _, w := range ws {
		if w == nil || w.Typ != typ || w.In() != k {
			return nil, nil
		}
		total += w.Out()
	}
	raw := make([]byte, 0)
	for _, w := range ws {
		raw = append(raw, w.Raw...)
	}
	return be.UploadWeight(typ, raw, []int{k, total})
}

// DeviceForwardEnabled reports whether the device-resident path is active.
func (m *Model) DeviceForwardEnabled() bool { return m.device != nil }
