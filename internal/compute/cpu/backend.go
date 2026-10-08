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

// RMSNormBack implements compute.Backend.
func (b *Backend) RMSNormBack(a, w, dOut compute.Buffer, eps float32) (compute.Buffer, error) {
	tx, err := tensor(a)
	if err != nil {
		return nil, err
	}
	td, err := tensor(dOut)
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
	row := tx.Ne(0)
	rows := len(tx.F32) / row
	out := compute.NewF32(tx.Dims...)
	for r := 0; r < rows; r++ {
		base := r * row
		var ss float64
		for i := 0; i < row; i++ {
			v := float64(tx.F32[base+i])
			ss += v * v
		}
		inv := 1 / math.Sqrt(ss/float64(row)+float64(eps))
		var dot float64
		for i := 0; i < row; i++ {
			wv := 1.0
			if weight != nil {
				wv = float64(weight[i])
			}
			dot += float64(td.F32[base+i]) * wv * float64(tx.F32[base+i])
		}
		c := inv * inv * inv / float64(row)
		for i := 0; i < row; i++ {
			wv := 1.0
			if weight != nil {
				wv = float64(weight[i])
			}
			dyhat := float64(td.F32[base+i]) * wv
			out.F32[base+i] = float32(dyhat*inv - c*float64(tx.F32[base+i])*dot)
		}
	}
	return wrap(out), nil
}

// SiluBack implements compute.Backend.
func (b *Backend) SiluBack(x, dOut compute.Buffer) (compute.Buffer, error) {
	tx, err := tensor(x)
	if err != nil {
		return nil, err
	}
	td, err := tensor(dOut)
	if err != nil {
		return nil, err
	}
	if len(tx.F32) != len(td.F32) {
		return nil, compute.ErrShape
	}
	out := compute.NewF32(tx.Dims...)
	for i, v := range tx.F32 {
		s := float64(compute.Sigmoid(v))
		out.F32[i] = float32(float64(td.F32[i]) * s * (1 + float64(v)*(1-s)))
	}
	return wrap(out), nil
}

// L2Norm implements compute.Backend.
func (b *Backend) L2Norm(a compute.Buffer, eps float32) (compute.Buffer, error) {
	ta, err := tensor(a)
	if err != nil {
		return nil, err
	}
	return wrap(compute.L2Norm(ta, eps)), nil
}

// L2NormBack implements compute.Backend.
func (b *Backend) L2NormBack(a, dOut compute.Buffer, eps float32) (compute.Buffer, error) {
	tx, err := tensor(a)
	if err != nil {
		return nil, err
	}
	td, err := tensor(dOut)
	if err != nil {
		return nil, err
	}
	row := tx.Ne(0)
	rows := len(tx.F32) / row
	out := compute.NewF32(tx.Dims...)
	for r := 0; r < rows; r++ {
		base := r * row
		var ss float64
		for i := 0; i < row; i++ {
			v := float64(tx.F32[base+i])
			ss += v * v
		}
		inv := 1 / math.Sqrt(ss+float64(eps))
		var c float64
		for i := 0; i < row; i++ {
			c += float64(td.F32[base+i]) * float64(tx.F32[base+i]) * inv
		}
		for i := 0; i < row; i++ {
			outv := float64(tx.F32[base+i]) * inv
			out.F32[base+i] = float32(inv * (float64(td.F32[base+i]) - c*outv))
		}
	}
	return wrap(out), nil
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

// SoftmaxBack implements compute.Backend.
func (b *Backend) SoftmaxBack(out, dOut compute.Buffer) (compute.Buffer, error) {
	to, err := tensor(out)
	if err != nil {
		return nil, err
	}
	td, err := tensor(dOut)
	if err != nil {
		return nil, err
	}
	row := to.Ne(0)
	rows := len(to.F32) / row
	dx := compute.NewF32(to.Dims...)
	for r := 0; r < rows; r++ {
		base := r * row
		var dot float64
		for i := 0; i < row; i++ {
			dot += float64(td.F32[base+i]) * float64(to.F32[base+i])
		}
		for i := 0; i < row; i++ {
			o := float64(to.F32[base+i])
			dx.F32[base+i] = float32(o * (float64(td.F32[base+i]) - dot))
		}
	}
	return wrap(dx), nil
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

// SSMConvBack implements compute.Backend.
func (b *Backend) SSMConvBack(sx, c, dOut compute.Buffer) (compute.Buffer, compute.Buffer, error) {
	ts, err := tensor(sx)
	if err != nil {
		return nil, nil, err
	}
	tc, err := tensor(c)
	if err != nil {
		return nil, nil, err
	}
	td, err := tensor(dOut)
	if err != nil {
		return nil, nil, err
	}
	dConv := tc.Ne(0)
	dInner := tc.Ne(1)
	ncs := ts.Ne(0)
	nT := ncs - dConv + 1
	nSeq := ts.Ne(2)
	if ts.Ne(1) != dInner || nT < 0 {
		return nil, nil, compute.ErrShape
	}
	dSx := compute.NewF32(ts.Dims...)
	dC := compute.NewF32(tc.Dims...)
	for s := 0; s < nSeq; s++ {
		for t := 0; t < nT; t++ {
			for ch := 0; ch < dInner; ch++ {
				g := float64(td.F32[(s*nT+t)*dInner+ch])
				if g == 0 {
					continue
				}
				sOff := ch*ncs + t
				cOff := ch * dConv
				for i := 0; i < dConv; i++ {
					dSx.F32[sOff+i] += float32(g * float64(tc.F32[cOff+i]))
					dC.F32[cOff+i] += float32(g * float64(ts.F32[sOff+i]))
				}
			}
		}
	}
	return wrap(dSx), wrap(dC), nil
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

// MatMulWeightTranspose implements compute.Backend. It is the reference for the
// device dX = W^T dY kernel: w [K,N], dY [N,M] -> [K,M].
func (b *Backend) MatMulWeightTranspose(w, dY compute.Buffer) (compute.Buffer, error) {
	wb, ok := w.(*weightBuffer)
	if !ok {
		return nil, fmt.Errorf("cpu: not a weight buffer")
	}
	td, err := tensor(dY)
	if err != nil {
		return nil, err
	}
	k := wb.dims[0]
	n := 1
	if len(wb.dims) > 1 {
		n = wb.dims[1]
	}
	m := td.Ne(1)
	if td.Ne(0) != n {
		return nil, compute.ErrShape
	}
	f32, err := quant.Dequant(wb.typ, wb.raw, int64(wb.NumElements()))
	if err != nil {
		return nil, err
	}
	out := compute.NewF32(k, m)
	for ni := 0; ni < n; ni++ {
		base := ni * k
		for mi := 0; mi < m; mi++ {
			g := td.F32[ni+mi*n]
			if g == 0 {
				continue
			}
			for ki := 0; ki < k; ki++ {
				out.F32[ki+mi*k] += f32[base+ki] * g
			}
		}
	}
	return wrap(out), nil
}

// MatMulWeightGrad implements compute.Backend. It is the reference for the
// device dW = x . dOut^T kernel: x [In,M], dOut [Out,M] -> dW [In,Out].
func (b *Backend) MatMulWeightGrad(x, dOut compute.Buffer) (compute.Buffer, error) {
	tx, err := tensor(x)
	if err != nil {
		return nil, err
	}
	td, err := tensor(dOut)
	if err != nil {
		return nil, err
	}
	in := tx.Ne(0)
	out := td.Ne(0)
	m := tx.Ne(1)
	if td.Ne(1) != m {
		return nil, compute.ErrShape
	}
	dw := compute.NewF32(in, out)
	for mm := 0; mm < m; mm++ {
		xbase := mm * in
		dbase := mm * out
		for n := 0; n < out; n++ {
			g := td.F32[dbase+n]
			if g == 0 {
				continue
			}
			wbase := n * in
			for i := 0; i < in; i++ {
				dw.F32[wbase+i] += tx.F32[xbase+i] * g
			}
		}
	}
	return wrap(dw), nil
}

// AttentionBackward implements compute.Backend. It mirrors
// autograd.AttentionBackward: GQA with optional causality.
func (b *Backend) AttentionBackward(q, k, v, dOut compute.Buffer, nHead, nHeadKV int, scale float32, causal bool) (compute.Buffer, compute.Buffer, compute.Buffer, error) {
	tq, err := tensor(q)
	if err != nil {
		return nil, nil, nil, err
	}
	tk, err := tensor(k)
	if err != nil {
		return nil, nil, nil, err
	}
	tv, err := tensor(v)
	if err != nil {
		return nil, nil, nil, err
	}
	td, err := tensor(dOut)
	if err != nil {
		return nil, nil, nil, err
	}
	hd := tq.Ne(0)
	nQ := tq.Ne(2)
	nKV := tk.Ne(2)
	group := nHead / nHeadKV
	dQ := compute.NewF32(hd, nHead, nQ)
	dK := compute.NewF32(hd, nHeadKV, nKV)
	dV := compute.NewF32(hd, nHeadKV, nKV)
	logits := make([]float64, nKV)
	dp := make([]float64, nKV)
	for h := 0; h < nHead; h++ {
		kv := h / group
		for i := 0; i < nQ; i++ {
			qb := (i*nHead + h) * hd
			maxJ := nKV
			if causal && nQ == nKV {
				maxJ = i + 1
			}
			best := math.Inf(-1)
			for j := 0; j < maxJ; j++ {
				kb := (j*nHeadKV + kv) * hd
				var dot float64
				for d := 0; d < hd; d++ {
					dot += float64(tq.F32[qb+d]) * float64(tk.F32[kb+d])
				}
				dot *= float64(scale)
				logits[j] = dot
				if dot > best {
					best = dot
				}
			}
			var sum float64
			for j := 0; j < maxJ; j++ {
				e := math.Exp(logits[j] - best)
				logits[j] = e
				sum += e
			}
			for j := 0; j < maxJ; j++ {
				logits[j] /= sum
			}
			var pd float64
			for j := 0; j < maxJ; j++ {
				vb := (j*nHeadKV + kv) * hd
				var acc float64
				for d := 0; d < hd; d++ {
					acc += float64(td.F32[qb+d]) * float64(tv.F32[vb+d])
				}
				dp[j] = acc
				pd += logits[j] * acc
			}
			for j := 0; j < maxJ; j++ {
				vb := (j*nHeadKV + kv) * hd
				dDot := logits[j] * (dp[j] - pd) * float64(scale)
				for d := 0; d < hd; d++ {
					dQ.F32[qb+d] += float32(dDot * float64(tk.F32[vb+d]))
					dK.F32[vb+d] += float32(dDot * float64(tq.F32[qb+d]))
					dV.F32[vb+d] += float32(logits[j] * float64(td.F32[qb+d]))
				}
			}
		}
	}
	return wrap(dQ), wrap(dK), wrap(dV), nil
}
