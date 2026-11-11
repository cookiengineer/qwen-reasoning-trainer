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
}

type deviceAttnLayer struct {
	norm, postNorm, qNorm, kNorm compute.Buffer
	q, k, v, o, fg, fu, fd       compute.Buffer
}

type deviceGDNLayer struct {
	norm, postNorm, ssmNorm, conv1d, alpha, beta compute.Buffer
	qkv, gate, out, fg, fu, fd                   compute.Buffer
}

type deviceLayer struct {
	attn *deviceAttnLayer
	gdn  *deviceGDNLayer
}

type deviceModel struct {
	embed, finalNorm, lmHead compute.Buffer
	layers                   []deviceLayer
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
			free(g.qkv)
			free(g.gate)
			free(g.out)
			free(g.fg)
			free(g.fu)
			free(g.fd)
		}
	}
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
	m.device = d
	return nil
}

// DeviceForwardEnabled reports whether the device-resident path is active.
func (m *Model) DeviceForwardEnabled() bool { return m.device != nil }
