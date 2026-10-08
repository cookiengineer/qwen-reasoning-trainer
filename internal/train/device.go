package train

import (
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// DeviceMLP is a two-layer LoRA-augmented MLP whose forward and backward run
// entirely on a compute.Backend without host round-trips between ops. It is the
// first resident-graph building block: the frozen base weights stay quantized on
// the device and only the F32 LoRA adapters are trained.
//
//	y = W2 * silu(W1 * x + s*B1*(A1*x)) + s*B2*(A2*h)
type DeviceMLP struct {
	be compute.Backend

	base1, a1, b1 compute.Buffer
	base2, a2, b2 compute.Buffer

	scale float32
	in    int
	hid   int
	out   int
	rank  int
}

// DeviceMLPConfig describes the block geometry.
type DeviceMLPConfig struct {
	In, Hidden, Out, Rank int
	Base1In, Base1Out     int
	Scale                 float32
}

// NewDeviceMLP uploads the base weights and adapters. base1/base2 are raw
// (possibly quantized) weight bytes; a1,a2,b1,b2 are F32 adapter tensors.
func NewDeviceMLP(be compute.Backend, base1 compute.Buffer, a1, b1 *compute.Tensor, base2 compute.Buffer, a2, b2 *compute.Tensor, in, hid, out, rank int, scale float32) (*DeviceMLP, error) {
	ba1, err := be.Upload(a1)
	if err != nil {
		return nil, err
	}
	bb1, err := be.Upload(b1)
	if err != nil {
		return nil, err
	}
	ba2, err := be.Upload(a2)
	if err != nil {
		return nil, err
	}
	bb2, err := be.Upload(b2)
	if err != nil {
		return nil, err
	}
	return &DeviceMLP{
		be: be, base1: base1, a1: ba1, b1: bb1,
		base2: base2, a2: ba2, b2: bb2,
		scale: scale, in: in, hid: hid, out: out, rank: rank,
	}, nil
}

// mlpCtx holds the forward activations needed by the backward pass.
type mlpCtx struct {
	x, u1, l1, h, u2 compute.Buffer
}

// MLPGrads holds the device-resident adapter gradients.
type MLPGrads struct {
	DA1, DB1, DA2, DB2 compute.Buffer
}

// linearForward computes base*x + scale*B*(A*x).
func (m *DeviceMLP) linearForward(base, a, b, x compute.Buffer) (y, u compute.Buffer, err error) {
	baseOut, err := m.be.MatMulWeight(base, x)
	if err != nil {
		return nil, nil, err
	}
	u, err = m.be.MatMul(a, x)
	if err != nil {
		return nil, nil, err
	}
	v, err := m.be.MatMul(b, u)
	if err != nil {
		return nil, nil, err
	}
	vs, err := m.be.Scale(v, m.scale)
	if err != nil {
		return nil, nil, err
	}
	y, err = m.be.Binary(compute.BinaryAdd, baseOut, vs)
	if err != nil {
		return nil, nil, err
	}
	return y, u, nil
}

// Forward runs the whole block on the device.
func (m *DeviceMLP) Forward(x compute.Buffer) (*mlpCtx, error) {
	l1, u1, err := m.linearForward(m.base1, m.a1, m.b1, x)
	if err != nil {
		return nil, err
	}
	h, err := m.be.Unary(compute.UnarySilu, l1)
	if err != nil {
		return nil, err
	}
	_, u2, err := m.linearForward(m.base2, m.a2, m.b2, h)
	if err != nil {
		return nil, err
	}
	return &mlpCtx{x: x, u1: u1, l1: l1, h: h, u2: u2}, nil
}

// linearBackward returns dX and the adapter gradients for a linear layer.
func (m *DeviceMLP) linearBackward(base, a, b, x, u, dY compute.Buffer) (dX, dA, dB compute.Buffer, err error) {
	dBase, err := m.be.MatMulWeightTranspose(base, dY)
	if err != nil {
		return nil, nil, nil, err
	}
	dVs, err := m.be.Scale(dY, m.scale)
	if err != nil {
		return nil, nil, nil, err
	}
	dB, err = m.be.MatMulWeightGrad(u, dVs)
	if err != nil {
		return nil, nil, nil, err
	}
	dU, err := m.be.MatMulWeightTranspose(b, dVs)
	if err != nil {
		return nil, nil, nil, err
	}
	dA, err = m.be.MatMulWeightGrad(x, dU)
	if err != nil {
		return nil, nil, nil, err
	}
	dXlora, err := m.be.MatMulWeightTranspose(a, dU)
	if err != nil {
		return nil, nil, nil, err
	}
	dX, err = m.be.Binary(compute.BinaryAdd, dBase, dXlora)
	if err != nil {
		return nil, nil, nil, err
	}
	return dX, dA, dB, nil
}

// Backward propagates dY through the block, returning the adapter gradients and
// the input gradient. All work is recorded on the device.
func (m *DeviceMLP) Backward(ctx *mlpCtx, dY compute.Buffer) (*MLPGrads, compute.Buffer, error) {
	dH, dA2, dB2, err := m.linearBackward(m.base2, m.a2, m.b2, ctx.h, ctx.u2, dY)
	if err != nil {
		return nil, nil, err
	}
	dL1, err := m.be.SiluBack(ctx.l1, dH)
	if err != nil {
		return nil, nil, err
	}
	dX, dA1, dB1, err := m.linearBackward(m.base1, m.a1, m.b1, ctx.x, ctx.u1, dL1)
	if err != nil {
		return nil, nil, err
	}
	return &MLPGrads{DA1: dA1, DB1: dB1, DA2: dA2, DB2: dB2}, dX, nil
}

// DownloadGrads copies the adapter gradients to the host.
func (m *DeviceMLP) DownloadGrads(g *MLPGrads) (da1, db1, da2, db2 *compute.Tensor, err error) {
	if da1, err = m.be.Download(g.DA1); err != nil {
		return
	}
	if db1, err = m.be.Download(g.DB1); err != nil {
		return
	}
	if da2, err = m.be.Download(g.DA2); err != nil {
		return
	}
	db2, err = m.be.Download(g.DB2)
	return
}
