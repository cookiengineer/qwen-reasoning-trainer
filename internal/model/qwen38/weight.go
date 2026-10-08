package qwen38

import (
	"fmt"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/gguf"
	"github.com/cookiengineer/qwen-reasoning-trainer/internal/quant"
)

// Weight holds a model parameter in its on-disk storage type. Large matrices
// stay quantized (e.g. Q4_K) and are dequantized on the fly, which keeps the
// resident footprint near the GGUF file size instead of 4 bytes per parameter.
type Weight struct {
	Typ  quant.Type
	Dims []int  // GGML order, Dims[0] is the reduction (in) dimension
	Raw  []byte // on-disk bytes, length == Typ.Size(product(Dims))
}

// NewWeightF32 builds a float32 weight from host data (used by tests).
func NewWeightF32(dims []int, data []float32) *Weight {
	raw, err := quant.Quantize(quant.TypeF32, data)
	if err != nil {
		panic(err)
	}
	return &Weight{Typ: quant.TypeF32, Dims: append([]int(nil), dims...), Raw: raw}
}

// In returns the input (reduction) dimension Dims[0].
func (w *Weight) In() int { return dimOr1(w.Dims, 0) }

// Out returns the output dimension Dims[1].
func (w *Weight) Out() int { return dimOr1(w.Dims, 1) }

// NumElements returns the product of the dimensions.
func (w *Weight) NumElements() int {
	n := 1
	for _, d := range w.Dims {
		n *= d
	}
	return n
}

// rowBytes returns the storage size of one output row (In elements).
func (w *Weight) rowBytes() (int, error) {
	sz, err := w.Typ.Size(int64(w.In()))
	if err != nil {
		return 0, err
	}
	return int(sz), nil
}

// Dequant returns the whole weight as a float32 tensor.
func (w *Weight) Dequant() (*compute.Tensor, error) {
	f32, err := quant.Dequant(w.Typ, w.Raw, int64(w.NumElements()))
	if err != nil {
		return nil, err
	}
	return &compute.Tensor{Dims: append([]int(nil), w.Dims...), Type: quant.TypeF32, F32: f32}, nil
}

// Tensor is a convenience wrapper that ignores the dequantization error. It is
// intended for tests and diagnostics; callers that must handle errors should
// use Dequant.
func (w *Weight) Tensor() *compute.Tensor {
	t, err := w.Dequant()
	if err != nil {
		panic(err)
	}
	return t
}

// Vector dequantizes a small weight (a norm or bias) to []float32.
func (w *Weight) Vector() ([]float32, error) {
	f32, err := quant.Dequant(w.Typ, w.Raw, int64(w.NumElements()))
	if err != nil {
		return nil, err
	}
	return f32, nil
}

// GetRows dequantizes only the selected rows, matching ggml_get_rows on a
// weight of shape [rowLen, nRows]. The result has Dims [rowLen, len(indices)].
func (w *Weight) GetRows(indices []int32) (*compute.Tensor, error) {
	rowLen := w.In()
	nRows := w.Out()
	rb, err := w.rowBytes()
	if err != nil {
		return nil, err
	}
	out := compute.NewF32(rowLen, len(indices))
	for i, idx := range indices {
		if idx < 0 || int(idx) >= nRows {
			return nil, fmt.Errorf("qwen38: row %d out of range [0,%d)", idx, nRows)
		}
		row := w.Raw[int(idx)*rb : (int(idx)+1)*rb]
		if err := quant.DequantTo(w.Typ, row, out.F32[i*rowLen:(i+1)*rowLen]); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// MatMul computes weight * x with the GGML layout: w [K,N], x [K,M] -> [N,M].
// The weight is dequantized in row blocks so peak host memory stays bounded
// regardless of the matrix size.
func (w *Weight) MatMul(x *compute.Tensor) (*compute.Tensor, error) {
	k := w.In()
	n := w.Out()
	m := x.Ne(1)
	if x.Ne(0) != k {
		return nil, compute.ErrShape
	}
	rb, err := w.rowBytes()
	if err != nil {
		return nil, err
	}
	out := compute.NewF32(n, m)

	const block = 1024
	sub := make([]float32, 0)
	for n0 := 0; n0 < n; n0 += block {
		rows := block
		if n0+rows > n {
			rows = n - n0
		}
		need := rows * k
		if cap(sub) < need {
			sub = make([]float32, need)
		}
		sub = sub[:need]
		for r := 0; r < rows; r++ {
			row := w.Raw[(n0+r)*rb : (n0+r+1)*rb]
			if err := quant.DequantTo(w.Typ, row, sub[r*k:(r+1)*k]); err != nil {
				return nil, err
			}
		}
		wsub := &compute.Tensor{Dims: []int{k, rows}, Type: quant.TypeF32, F32: sub}
		blockOut, err := compute.MatMul(wsub, x)
		if err != nil {
			return nil, err
		}
		for t := 0; t < m; t++ {
			copy(out.F32[(n0)+t*n:(n0+rows)+t*n], blockOut.F32[t*rows:(t+1)*rows])
		}
	}
	return out, nil
}

// MatMulTranspose computes W^T * dY: w [K,N], dY [N,M] -> [K,M]. It is the
// activation gradient of MatMul when the weight is frozen and used by the
// training autograd. The weight is dequantized in row blocks so peak host
// memory stays bounded.
func (w *Weight) MatMulTranspose(dY *compute.Tensor) (*compute.Tensor, error) {
	k := w.In()
	n := w.Out()
	m := dY.Ne(1)
	if dY.Ne(0) != n {
		return nil, compute.ErrShape
	}
	rb, err := w.rowBytes()
	if err != nil {
		return nil, err
	}
	out := compute.NewF32(k, m)
	const block = 1024
	sub := make([]float32, 0)
	for n0 := 0; n0 < n; n0 += block {
		rows := block
		if n0+rows > n {
			rows = n - n0
		}
		need := rows * k
		if cap(sub) < need {
			sub = make([]float32, need)
		}
		sub = sub[:need]
		for r := 0; r < rows; r++ {
			row := w.Raw[(n0+r)*rb : (n0+r+1)*rb]
			if err := quant.DequantTo(w.Typ, row, sub[r*k:(r+1)*k]); err != nil {
				return nil, err
			}
		}
		for r := 0; r < rows; r++ {
			wrow := sub[r*k : r*k+k]
			for mm := 0; mm < m; mm++ {
				g := dY.F32[(n0+r)+mm*n]
				if g == 0 {
					continue
				}
				base := mm * k
				for kk := 0; kk < k; kk++ {
					out.F32[base+kk] += wrow[kk] * g
				}
			}
		}
	}
	return out, nil
}

// loadWeightFromGGUF reads a weight without dequantizing it.
func loadWeightFromGGUF(g *gguf.File, name string) (*Weight, error) {
	ti, ok := g.Tensor(name)
	if !ok {
		return nil, fmt.Errorf("qwen38: missing tensor %q", name)
	}
	raw, err := g.ReadTensor(ti)
	if err != nil {
		return nil, err
	}
	dims := make([]int, len(ti.Dims))
	for i, d := range ti.Dims {
		dims[i] = int(d)
	}
	return &Weight{Typ: ti.Type, Dims: dims, Raw: raw}, nil
}

func dimOr1(dims []int, i int) int {
	if i < 0 || i >= len(dims) {
		return 1
	}
	return dims[i]
}
