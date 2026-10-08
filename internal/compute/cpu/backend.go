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

// BeginScope implements compute.Backend. The CPU backend relies on Go garbage
// collection, so scopes are no-ops.
func (b *Backend) BeginScope() {}

// EndScope implements compute.Backend.
func (b *Backend) EndScope(keep ...compute.Buffer) {}

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
		case compute.BinarySigmoidBack:
			out.F32[i] = ta.F32[i] * (1 - ta.F32[i]) * tc.F32[i]
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

// GatedDeltaNetBackward implements compute.Backend. Naive (one state snapshot
// per token) reference for the device atomic kernel.
func (b *Backend) GatedDeltaNetBackward(q, k, v, g, beta, state, dOut, dNewState compute.Buffer) (compute.Buffer, compute.Buffer, compute.Buffer, compute.Buffer, compute.Buffer, compute.Buffer, error) {
	tq, err := tensor(q)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	tk, err := tensor(k)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	tv, err := tensor(v)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	tg, err := tensor(g)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	tb, err := tensor(beta)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	ts, err := tensor(state)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	td, err := tensor(dOut)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	tn, err := tensor(dNewState)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	sv := tv.Ne(0)
	h := tv.Ne(1)
	nTok := tv.Ne(2)
	kda := tg.Ne(0) == sv
	gstride := tg.Ne(0)
	scale := float64(1 / math.Sqrt(float64(sv)))

	dQ := compute.NewF32(tq.Dims...)
	dK := compute.NewF32(tk.Dims...)
	dV := compute.NewF32(tv.Dims...)
	dG := compute.NewF32(tg.Dims...)
	dB := compute.NewF32(tb.Dims...)
	dS := compute.NewF32(ts.Dims...)

	work := make([]float64, sv*sv)
	snap := make([][]float64, nTok+1)
	for t := range snap {
		snap[t] = make([]float64, sv*sv)
	}
	decay := make([]float64, sv)
	delta := make([]float64, sv)
	ds := make([]float64, sv*sv)
	dA := make([]float64, sv*sv)
	ddelta := make([]float64, sv)

	for hh := 0; hh < h; hh++ {
		for i := 0; i < sv*sv; i++ {
			work[i] = float64(ts.F32[hh*sv*sv+i])
		}
		copy(snap[0], work)
		for t := 0; t < nTok; t++ {
			off := (t*h + hh) * sv
			gOff := (t*h + hh) * gstride
			if kda {
				for i := 0; i < sv; i++ {
					decay[i] = math.Exp(float64(tg.F32[gOff+i]))
				}
				for j := 0; j < sv; j++ {
					r := work[j*sv : j*sv+sv]
					for i := 0; i < sv; i++ {
						r[i] *= decay[i]
					}
				}
			} else {
				dec := math.Exp(float64(tg.F32[gOff]))
				for i := range work {
					work[i] *= dec
				}
			}
			bv := float64(tb.F32[t*h+hh])
			for j := 0; j < sv; j++ {
				var sum float64
				r := work[j*sv : j*sv+sv]
				for i := 0; i < sv; i++ {
					sum += r[i] * float64(tk.F32[off+i])
				}
				delta[j] = (float64(tv.F32[off+j]) - sum) * bv
			}
			for j := 0; j < sv; j++ {
				dj := delta[j]
				r := work[j*sv : j*sv+sv]
				for i := 0; i < sv; i++ {
					r[i] += float64(tk.F32[off+i]) * dj
				}
			}
			copy(snap[t+1], work)
		}
		for i := 0; i < sv*sv; i++ {
			ds[i] = float64(tn.F32[hh*sv*sv+i])
		}
		for t := nTok - 1; t >= 0; t-- {
			off := (t*h + hh) * sv
			gOff := (t*h + hh) * gstride
			bOff := t*h + hh
			if kda {
				for i := 0; i < sv; i++ {
					decay[i] = math.Exp(float64(tg.F32[gOff+i]))
				}
			} else {
				dec := math.Exp(float64(tg.F32[gOff]))
				for i := 0; i < sv; i++ {
					decay[i] = dec
				}
			}
			bv := float64(tb.F32[bOff])
			for j := 0; j < sv; j++ {
				do := float64(td.F32[off+j])
				if do == 0 {
					continue
				}
				for i := 0; i < sv; i++ {
					ds[j*sv+i] += scale * do * float64(tq.F32[off+i])
				}
			}
			for i := 0; i < sv; i++ {
				var sum float64
				for j := 0; j < sv; j++ {
					sum += float64(td.F32[off+j]) * snap[t+1][j*sv+i]
				}
				dQ.F32[off+i] += float32(scale * sum)
			}
			for j := 0; j < sv; j++ {
				var sum float64
				for i := 0; i < sv; i++ {
					sum += snap[t][j*sv+i] * decay[i] * float64(tk.F32[off+i])
				}
				delta[j] = (float64(tv.F32[off+j]) - sum) * bv
			}
			for j := 0; j < sv; j++ {
				for i := 0; i < sv; i++ {
					dA[j*sv+i] = ds[j*sv+i]
					dK.F32[off+i] += float32(ds[j*sv+i] * delta[j])
				}
			}
			for j := 0; j < sv; j++ {
				var sum float64
				for i := 0; i < sv; i++ {
					sum += ds[j*sv+i] * float64(tk.F32[off+i])
				}
				ddelta[j] = sum
			}
			var dba float64
			for j := 0; j < sv; j++ {
				var r float64
				for i := 0; i < sv; i++ {
					r += snap[t][j*sv+i] * decay[i] * float64(tk.F32[off+i])
				}
				r = float64(tv.F32[off+j]) - r
				dV.F32[off+j] += float32(bv * ddelta[j])
				dba += ddelta[j] * r
				for i := 0; i < sv; i++ {
					dA[j*sv+i] -= bv * ddelta[j] * float64(tk.F32[off+i])
					dK.F32[off+i] -= float32(bv * ddelta[j] * snap[t][j*sv+i] * decay[i])
				}
			}
			dB.F32[bOff] += float32(dba)
			for j := 0; j < sv; j++ {
				for i := 0; i < sv; i++ {
					ds[j*sv+i] = dA[j*sv+i] * decay[i]
				}
			}
			if kda {
				for i := 0; i < sv; i++ {
					var sum float64
					for j := 0; j < sv; j++ {
						sum += dA[j*sv+i] * snap[t][j*sv+i]
					}
					dG.F32[gOff+i] += float32(sum * decay[i])
				}
			} else {
				var sum float64
				for c := 0; c < sv*sv; c++ {
					sum += dA[c] * snap[t][c]
				}
				dG.F32[gOff] += float32(sum * decay[0])
			}
		}
		for i := 0; i < sv*sv; i++ {
			dS.F32[hh*sv*sv+i] = float32(ds[i])
		}
	}
	return wrap(dQ), wrap(dK), wrap(dV), wrap(dG), wrap(dB), wrap(dS), nil
}

// SplitQG implements compute.Backend.
func (b *Backend) SplitQG(qg compute.Buffer, hd, nHead, T int) (compute.Buffer, compute.Buffer, error) {
	tq, err := tensor(qg)
	if err != nil {
		return nil, nil, err
	}
	row := nHead * hd
	q := compute.NewF32(hd, nHead, T)
	gate := compute.NewF32(row, T)
	for t := 0; t < T; t++ {
		for hh := 0; hh < nHead; hh++ {
			src := t*(2*row) + hh*2*hd
			for d := 0; d < hd; d++ {
				dst := d + hh*hd + t*row
				q.F32[dst] = tq.F32[src+d]
				gate.F32[dst] = tq.F32[src+hd+d]
			}
		}
	}
	return wrap(q), wrap(gate), nil
}

// SplitQGBack implements compute.Backend.
func (b *Backend) SplitQGBack(dq, dgate compute.Buffer) (compute.Buffer, error) {
	tq, err := tensor(dq)
	if err != nil {
		return nil, err
	}
	tg, err := tensor(dgate)
	if err != nil {
		return nil, err
	}
	hd := tq.Ne(0)
	nHead := tq.Ne(1)
	T := tq.Ne(2)
	row := nHead * hd
	dqg := compute.NewF32(2*row, T)
	for t := 0; t < T; t++ {
		for hh := 0; hh < nHead; hh++ {
			dst := t*(2*row) + hh*2*hd
			for d := 0; d < hd; d++ {
				src := d + hh*hd + t*row
				dqg.F32[dst+d] = tq.F32[src]
				dqg.F32[dst+hd+d] = tg.F32[src]
			}
		}
	}
	return wrap(dqg), nil
}

// Reshape implements compute.Backend.
func (b *Backend) Reshape(buf compute.Buffer, dims []int) (compute.Buffer, error) {
	t, err := tensor(buf)
	if err != nil {
		return nil, err
	}
	n := 1
	for _, d := range dims {
		n *= d
	}
	if n != len(t.F32) {
		return nil, compute.ErrShape
	}
	return wrap(&compute.Tensor{Dims: append([]int(nil), dims...), Type: t.Type, F32: t.F32}), nil
}

// ConvInput implements compute.Backend.
func (b *Backend) ConvInput(qkv compute.Buffer, dConv, convDim, T int) (compute.Buffer, error) {
	t, err := tensor(qkv)
	if err != nil {
		return nil, err
	}
	ncs := dConv - 1 + T
	out := compute.NewF32(ncs, convDim, 1)
	for tt := 0; tt < T; tt++ {
		for ch := 0; ch < convDim; ch++ {
			out.F32[(dConv-1+tt)+ch*ncs] = t.F32[ch+tt*convDim]
		}
	}
	return wrap(out), nil
}

// ConvInputBack implements compute.Backend.
func (b *Backend) ConvInputBack(dConvIn compute.Buffer, dConv, convDim, T int) (compute.Buffer, error) {
	t, err := tensor(dConvIn)
	if err != nil {
		return nil, err
	}
	ncs := dConv - 1 + T
	out := compute.NewF32(convDim, T)
	for tt := 0; tt < T; tt++ {
		for ch := 0; ch < convDim; ch++ {
			out.F32[ch+tt*convDim] = t.F32[(dConv-1+tt)+ch*ncs]
		}
	}
	return wrap(out), nil
}

// GatherHeads implements compute.Backend.
func (b *Backend) GatherHeads(convOut compute.Buffer, offset, headDim, nHead, T, convDim int) (compute.Buffer, error) {
	t, err := tensor(convOut)
	if err != nil {
		return nil, err
	}
	out := compute.NewF32(headDim, nHead, T)
	for tt := 0; tt < T; tt++ {
		for hh := 0; hh < nHead; hh++ {
			base := offset + hh*headDim
			for d := 0; d < headDim; d++ {
				out.F32[d+hh*headDim+tt*nHead*headDim] = t.F32[(base+d)+tt*convDim]
			}
		}
	}
	return wrap(out), nil
}

// GatherHeadsBack implements compute.Backend.
func (b *Backend) GatherHeadsBack(dOut compute.Buffer, offset, headDim, nHead, T, convDim int) (compute.Buffer, error) {
	t, err := tensor(dOut)
	if err != nil {
		return nil, err
	}
	out := compute.NewF32(convDim, T)
	for tt := 0; tt < T; tt++ {
		for hh := 0; hh < nHead; hh++ {
			base := offset + hh*headDim
			for d := 0; d < headDim; d++ {
				out.F32[(base+d)+tt*convDim] = t.F32[d+hh*headDim+tt*nHead*headDim]
			}
		}
	}
	return wrap(out), nil
}

// RepeatHeads implements compute.Backend.
func (b *Backend) RepeatHeads(x compute.Buffer, headDim, nIn, nOut, T int) (compute.Buffer, error) {
	t, err := tensor(x)
	if err != nil {
		return nil, err
	}
	out := compute.NewF32(headDim, nOut, T)
	for tt := 0; tt < T; tt++ {
		for hh := 0; hh < nOut; hh++ {
			src := hh % nIn
			for d := 0; d < headDim; d++ {
				out.F32[d+hh*headDim+tt*nOut*headDim] = t.F32[d+src*headDim+tt*nIn*headDim]
			}
		}
	}
	return wrap(out), nil
}

// RepeatHeadsBack implements compute.Backend.
func (b *Backend) RepeatHeadsBack(dOut compute.Buffer, headDim, nIn, nOut, T int) (compute.Buffer, error) {
	t, err := tensor(dOut)
	if err != nil {
		return nil, err
	}
	out := compute.NewF32(headDim, nIn, T)
	for tt := 0; tt < T; tt++ {
		for hh := 0; hh < nIn; hh++ {
			for d := 0; d < headDim; d++ {
				var s float32
				for o := hh; o < nOut; o += nIn {
					s += t.F32[d+o*headDim+tt*nOut*headDim]
				}
				out.F32[d+hh*headDim+tt*nIn*headDim] = s
			}
		}
	}
	return wrap(out), nil
}

// GetRowsWeight implements compute.Backend.
func (b *Backend) GetRowsWeight(w compute.Buffer, indices []int32) (compute.Buffer, error) {
	wb, ok := w.(*weightBuffer)
	if !ok {
		return b.GetRows(w, indices)
	}
	rowLen := wb.dims[0]
	nRows := 1
	if len(wb.dims) > 1 {
		nRows = wb.dims[1]
	}
	rowBytes, err := wb.typ.Size(int64(rowLen))
	if err != nil {
		return nil, err
	}
	out := compute.NewF32(rowLen, len(indices))
	for i, idx := range indices {
		if idx < 0 || int(idx) >= nRows {
			return nil, compute.ErrShape
		}
		if err := quant.DequantTo(wb.typ, wb.raw[int(idx)*int(rowBytes):(int(idx)+1)*int(rowBytes)], out.F32[i*rowLen:(i+1)*rowLen]); err != nil {
			return nil, err
		}
	}
	return wrap(out), nil
}

// AdamWStep implements compute.Backend: the reference for the device optimizer.
func (b *Backend) AdamWStep(param, grad, m, v compute.Buffer, p compute.AdamWParams) error {
	tp, err := tensor(param)
	if err != nil {
		return err
	}
	tg, err := tensor(grad)
	if err != nil {
		return err
	}
	tm, err := tensor(m)
	if err != nil {
		return err
	}
	tv, err := tensor(v)
	if err != nil {
		return err
	}
	if len(tp.F32) != len(tg.F32) || len(tm.F32) != len(tg.F32) || len(tv.F32) != len(tg.F32) {
		return compute.ErrShape
	}
	b1, b2 := float64(p.Beta1), float64(p.Beta2)
	alpha, eps, wd := float64(p.Alpha), float64(p.Eps), float64(p.WeightDecay)
	b1h, b2h := float64(p.Beta1Hat), float64(p.Beta2Hat)
	for i := range tg.F32 {
		gi := float64(tg.F32[i])
		gmi := float64(tm.F32[i])*b1 + gi*(1-b1)
		gvi := float64(tv.F32[i])*b2 + gi*gi*(1-b2)
		tm.F32[i] = float32(gmi)
		tv.F32[i] = float32(gvi)
		mh := gmi * b1h
		vh := math.Sqrt(gvi*b2h) + eps
		tp.F32[i] = float32(float64(tp.F32[i])*(1-alpha*wd) - alpha*mh/vh)
	}
	return nil
}

// SumSquares implements compute.Backend.
func (b *Backend) SumSquares(dst, src compute.Buffer) error {
	td, err := tensor(dst)
	if err != nil {
		return err
	}
	ts, err := tensor(src)
	if err != nil {
		return err
	}
	if len(td.F32) < 1 {
		return compute.ErrShape
	}
	var ss float64
	for _, x := range ts.F32 {
		ss += float64(x) * float64(x)
	}
	td.F32[0] += float32(ss)
	return nil
}

// CrossEntropy implements compute.Backend.
func (b *Backend) CrossEntropy(logits compute.Buffer, targets []int32, ignoreIndex int) (compute.Buffer, compute.Buffer, error) {
	t, err := tensor(logits)
	if err != nil {
		return nil, nil, err
	}
	V := t.Ne(0)
	T := t.Ne(1)
	if len(targets) != T {
		return nil, nil, compute.ErrShape
	}
	loss := compute.NewF32(1)
	dLogits := compute.NewF32(V, T)
	count := 0
	for _, tg := range targets {
		if int(tg) != ignoreIndex {
			count++
		}
	}
	if count == 0 {
		return wrap(loss), wrap(dLogits), nil
	}
	inv := 1 / float64(count)
	var total float64
	for tt := 0; tt < T; tt++ {
		if int(targets[tt]) == ignoreIndex {
			continue
		}
		base := tt * V
		max := math.Inf(-1)
		for v := 0; v < V; v++ {
			if float64(t.F32[base+v]) > max {
				max = float64(t.F32[base+v])
			}
		}
		var sum float64
		for v := 0; v < V; v++ {
			sum += math.Exp(float64(t.F32[base+v]) - max)
		}
		pt := math.Exp(float64(t.F32[base+int(targets[tt])])-max) / sum
		total += -math.Log(pt)
		for v := 0; v < V; v++ {
			p := math.Exp(float64(t.F32[base+v])-max) / sum
			if v == int(targets[tt]) {
				p -= 1
			}
			dLogits.F32[base+v] = float32(p * inv)
		}
	}
	loss.F32[0] = float32(total * inv)
	return wrap(loss), wrap(dLogits), nil
}
