// Package cpu implements the reference compute.Backend in pure Go. It is the
// numerical oracle for the Vulkan backend and runs in GPU-less CI.
package cpu

import (
	"fmt"
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// Backend is a host-memory compute backend.
type Backend struct{}

// New returns a new CPU backend.
func New() *Backend { return &Backend{} }

type buffer struct {
	t *compute.Tensor
}

func (b *buffer) Dims() []int      { return b.t.Dims }
func (b *buffer) NumElements() int { return b.t.NumElements() }
func (b *buffer) Type() quant.Type { return b.t.Type }

// Capabilities implements compute.Backend.
func (b *Backend) Capabilities() compute.Capabilities {
	return compute.Capabilities{Name: "cpu", HostFallback: true}
}

// Alloc implements compute.Backend.
func (b *Backend) Alloc(dims []int, t quant.Type) (compute.Buffer, error) {
	return &buffer{t: compute.NewF32(dims...)}, nil
}

// Upload implements compute.Backend.
func (b *Backend) Upload(t *compute.Tensor) (compute.Buffer, error) {
	return &buffer{t: t.Clone()}, nil
}

// Download implements compute.Backend.
func (b *Backend) Download(buf compute.Buffer) (*compute.Tensor, error) {
	cb, ok := buf.(*buffer)
	if !ok {
		return nil, fmt.Errorf("cpu: foreign buffer")
	}
	return cb.t.Clone(), nil
}

// Free implements compute.Backend.
func (b *Backend) Free(buf compute.Buffer) {}

// Sync implements compute.Backend.
func (b *Backend) Sync() error { return nil }

// Close implements compute.Backend.
func (b *Backend) Close() error { return nil }

func wrap(t *compute.Tensor) compute.Buffer { return &buffer{t: t} }

func tensor(buf compute.Buffer) (*compute.Tensor, error) {
	cb, ok := buf.(*buffer)
	if !ok {
		return nil, fmt.Errorf("cpu: foreign buffer")
	}
	return cb.t, nil
}

// Unary implements compute.Backend.
func (b *Backend) Unary(op compute.UnaryOp, a compute.Buffer) (compute.Buffer, error) {
	ta, err := tensor(a)
	if err != nil {
		return nil, err
	}
	out := compute.NewF32(ta.Dims...)
	for i, v := range ta.F32 {
		out.F32[i] = applyUnary(op, v)
	}
	return wrap(out), nil
}

func applyUnary(op compute.UnaryOp, v float32) float32 {
	switch op {
	case compute.UnarySilu:
		return compute.Silu(v)
	case compute.UnarySigmoid:
		return compute.Sigmoid(v)
	case compute.UnarySoftplus:
		return float32(math.Log1p(math.Exp(float64(v))))
	case compute.UnaryGelu:
		return float32(0.5 * float64(v) * (1 + math.Erf(float64(v)/math.Sqrt2)))
	case compute.UnaryNeg:
		return -v
	case compute.UnaryExp:
		return float32(math.Exp(float64(v)))
	case compute.UnarySqr:
		return v * v
	case compute.UnarySqrt:
		return float32(math.Sqrt(float64(v)))
	case compute.UnaryTanh:
		return float32(math.Tanh(float64(v)))
	default:
		return v
	}
}

// Binary implements compute.Backend.
func (b *Backend) Binary(op compute.BinaryOp, a, c compute.Buffer) (compute.Buffer, error) {
	ta, err := tensor(a)
	if err != nil {
		return nil, err
	}
	tc, err := tensor(c)
	if err != nil {
		return nil, err
	}
	if len(ta.F32) != len(tc.F32) {
		return nil, compute.ErrShape
	}
	out := compute.NewF32(ta.Dims...)
	for i := range ta.F32 {
		switch op {
		case compute.BinaryAdd:
			out.F32[i] = ta.F32[i] + tc.F32[i]
		case compute.BinarySub:
			out.F32[i] = ta.F32[i] - tc.F32[i]
		case compute.BinaryMul:
			out.F32[i] = ta.F32[i] * tc.F32[i]
		case compute.BinaryDiv:
			out.F32[i] = ta.F32[i] / tc.F32[i]
		}
	}
	return wrap(out), nil
}

// Scale implements compute.Backend.
func (b *Backend) Scale(a compute.Buffer, s float32) (compute.Buffer, error) {
	ta, err := tensor(a)
	if err != nil {
		return nil, err
	}
	return wrap(compute.Scale(ta, s)), nil
}

// MatMul implements compute.Backend.
func (b *Backend) MatMul(a, c compute.Buffer) (compute.Buffer, error) {
	ta, err := tensor(a)
	if err != nil {
		return nil, err
	}
	tc, err := tensor(c)
	if err != nil {
		return nil, err
	}
	out, err := compute.MatMul(ta, tc)
	if err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// RMSNorm implements compute.Backend.
func (b *Backend) RMSNorm(a, w compute.Buffer, eps float32) (compute.Buffer, error) {
	ta, err := tensor(a)
	if err != nil {
		return nil, err
	}
	var weight []float32
	if w != nil {
		tw, err := tensor(w)
		if err != nil {
			return nil, err
		}
		weight = tw.F32
	}
	return wrap(compute.RMSNorm(ta, weight, eps)), nil
}

// Softmax implements compute.Backend.
func (b *Backend) Softmax(a compute.Buffer) (compute.Buffer, error) {
	ta, err := tensor(a)
	if err != nil {
		return nil, err
	}
	out := ta.Clone()
	compute.Softmax(out)
	return wrap(out), nil
}

// GetRows implements compute.Backend.
func (b *Backend) GetRows(table compute.Buffer, indices []int32) (compute.Buffer, error) {
	tt, err := tensor(table)
	if err != nil {
		return nil, err
	}
	out, err := compute.GetRows(tt, indices)
	if err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// RoPE implements compute.Backend.
func (b *Backend) RoPE(a compute.Buffer, positions []int32, theta float64, nDims int) (compute.Buffer, error) {
	ta, err := tensor(a)
	if err != nil {
		return nil, err
	}
	out, err := compute.RoPENeoX(ta, positions, theta, nDims)
	if err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// Attention implements compute.Backend.
func (b *Backend) Attention(q, k, v compute.Buffer, nHead, nHeadKV int, scale float32, causal bool) (compute.Buffer, error) {
	tq, err := tensor(q)
	if err != nil {
		return nil, err
	}
	tk, err := tensor(k)
	if err != nil {
		return nil, err
	}
	tv, err := tensor(v)
	if err != nil {
		return nil, err
	}
	out, err := compute.Attention(tq, tk, tv, nHead, nHeadKV, scale, causal)
	if err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// SSMConv implements compute.Backend.
func (b *Backend) SSMConv(sx, c compute.Buffer) (compute.Buffer, error) {
	ts, err := tensor(sx)
	if err != nil {
		return nil, err
	}
	tc, err := tensor(c)
	if err != nil {
		return nil, err
	}
	out, err := compute.SSMConv(ts, tc)
	if err != nil {
		return nil, err
	}
	return wrap(out), nil
}

// GatedDeltaNet implements compute.Backend.
func (b *Backend) GatedDeltaNet(q, k, v, g, beta, state compute.Buffer) (compute.Buffer, compute.Buffer, error) {
	tq, err := tensor(q)
	if err != nil {
		return nil, nil, err
	}
	tk, err := tensor(k)
	if err != nil {
		return nil, nil, err
	}
	tv, err := tensor(v)
	if err != nil {
		return nil, nil, err
	}
	tg, err := tensor(g)
	if err != nil {
		return nil, nil, err
	}
	tb, err := tensor(beta)
	if err != nil {
		return nil, nil, err
	}
	ts, err := tensor(state)
	if err != nil {
		return nil, nil, err
	}
	out, ns, err := compute.GatedDeltaNet(tq, tk, tv, tg, tb, ts)
	if err != nil {
		return nil, nil, err
	}
	return wrap(out), wrap(ns), nil
}

// Copy implements compute.Backend.
func (b *Backend) Copy(dst, src compute.Buffer) error {
	td, err := tensor(dst)
	if err != nil {
		return err
	}
	ts, err := tensor(src)
	if err != nil {
		return err
	}
	if len(td.F32) != len(ts.F32) {
		return compute.ErrShape
	}
	copy(td.F32, ts.F32)
	return nil
}

// weightBuffer holds raw quantized weight bytes.
type weightBuffer struct {
	typ  quant.Type
	dims []int
	raw  []byte
}

func (w *weightBuffer) Dims() []int      { return w.dims }
func (w *weightBuffer) Type() quant.Type { return w.typ }
func (w *weightBuffer) NumElements() int {
	n := 1
	for _, d := range w.dims {
		n *= d
	}
	return n
}

// UploadWeight implements compute.Backend.
func (b *Backend) UploadWeight(t quant.Type, raw []byte, dims []int) (compute.Buffer, error) {
	return &weightBuffer{typ: t, dims: append([]int(nil), dims...), raw: append([]byte(nil), raw...)}, nil
}

// DequantWeight implements compute.Backend.
func (b *Backend) DequantWeight(w compute.Buffer) (compute.Buffer, error) {
	wb, ok := w.(*weightBuffer)
	if !ok {
		return nil, fmt.Errorf("cpu: not a weight buffer")
	}
	f32, err := quant.Dequant(wb.typ, wb.raw, int64(wb.NumElements()))
	if err != nil {
		return nil, err
	}
	return wrap(&compute.Tensor{Dims: wb.dims, Type: quant.TypeF32, F32: f32}), nil
}

// MatMulWeight implements compute.Backend.
func (b *Backend) MatMulWeight(w, x compute.Buffer) (compute.Buffer, error) {
	dq, err := b.DequantWeight(w)
	if err != nil {
		return nil, err
	}
	return b.MatMul(dq, x)
}
