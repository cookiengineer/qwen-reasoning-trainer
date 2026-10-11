package compute

import (
	"errors"
	"math"
)

// ErrShape reports incompatible tensor shapes.
var ErrShape = errors.New("compute: shape mismatch")

// MatMul computes the GGML-style matrix product of a [K,N] and b [K,M],
// producing [N,M]. This matches ggml_mul_mat: the reduction dimension is Dims[0]
// of both inputs and the result uses GGML layout, i.e. element (i,j) is stored
// at i + j*N. Weight matrices are passed as a and activations as b.
func MatMul(a, b *Tensor) (*Tensor, error) {
	k, n, m := a.Ne(0), a.Ne(1), b.Ne(1)
	if b.Ne(0) != k {
		return nil, ErrShape
	}
	out := NewF32(n, m)
	for ni := 0; ni < n; ni++ {
		aRow := a.F32[ni*k : ni*k+k]
		for mi := 0; mi < m; mi++ {
			bRow := b.F32[mi*k : mi*k+k]
			var acc float64
			for ki := 0; ki < k; ki++ {
				acc += float64(aRow[ki]) * float64(bRow[ki])
			}
			out.F32[ni+mi*n] = float32(acc)
		}
	}
	return out, nil
}

// Add returns a+b elementwise.
func Add(a, b *Tensor) (*Tensor, error) {
	if len(a.F32) != len(b.F32) {
		return nil, ErrShape
	}
	out := NewF32(a.Dims...)
	for i := range a.F32 {
		out.F32[i] = a.F32[i] + b.F32[i]
	}
	return out, nil
}

// Mul returns a*b elementwise.
func Mul(a, b *Tensor) (*Tensor, error) {
	if len(a.F32) != len(b.F32) {
		return nil, ErrShape
	}
	out := NewF32(a.Dims...)
	for i := range a.F32 {
		out.F32[i] = a.F32[i] * b.F32[i]
	}
	return out, nil
}

// Scale returns a*s elementwise.
func Scale(a *Tensor, s float32) *Tensor {
	out := NewF32(a.Dims...)
	for i := range a.F32 {
		out.F32[i] = a.F32[i] * s
	}
	return out
}

// RMSNorm applies root-mean-square normalization over each contiguous row of
// length Dims[0], then multiplies by weight (which may be nil).
func RMSNorm(x *Tensor, weight []float32, eps float32) *Tensor {
	row := x.Ne(0)
	rows := len(x.F32) / row
	out := NewF32(x.Dims...)
	for r := 0; r < rows; r++ {
		base := r * row
		var ss float64
		for i := 0; i < row; i++ {
			v := float64(x.F32[base+i])
			ss += v * v
		}
		mean := ss / float64(row)
		inv := float32(1 / math.Sqrt(mean+float64(eps)))
		for i := 0; i < row; i++ {
			v := x.F32[base+i] * inv
			if weight != nil {
				v *= weight[i]
			}
			out.F32[base+i] = v
		}
	}
	return out
}

// Softmax applies softmax over each contiguous row of length Dims[0] in place.
func Softmax(x *Tensor) {
	row := x.Ne(0)
	rows := len(x.F32) / row
	for r := 0; r < rows; r++ {
		base := r * row
		max := float32(math.Inf(-1))
		for i := 0; i < row; i++ {
			if x.F32[base+i] > max {
				max = x.F32[base+i]
			}
		}
		var sum float64
		for i := 0; i < row; i++ {
			e := float32(math.Exp(float64(x.F32[base+i] - max)))
			x.F32[base+i] = e
			sum += float64(e)
		}
		inv := float32(1 / sum)
		for i := 0; i < row; i++ {
			x.F32[base+i] *= inv
		}
	}
}

// CrossEntropy returns the mean cross-entropy loss of logits [vocab,n] against
// targets of length n, ignoring entries equal to ignoreIndex.
func CrossEntropy(logits *Tensor, targets []int, ignoreIndex int) (float32, error) {
	vocab := logits.Ne(0)
	n := logits.Ne(1)
	if len(targets) != n {
		return 0, ErrShape
	}
	var loss float64
	count := 0
	for t := 0; t < n; t++ {
		if targets[t] == ignoreIndex {
			continue
		}
		if targets[t] < 0 || targets[t] >= vocab {
			return 0, ErrShape
		}
		base := t * vocab
		max := float32(math.Inf(-1))
		for i := 0; i < vocab; i++ {
			if logits.F32[base+i] > max {
				max = logits.F32[base+i]
			}
		}
		var sum float64
		for i := 0; i < vocab; i++ {
			sum += math.Exp(float64(logits.F32[base+i] - max))
		}
		lse := float64(max) + math.Log(sum)
		loss += lse - float64(logits.F32[base+targets[t]])
		count++
	}
	if count == 0 {
		return 0, nil
	}
	return float32(loss / float64(count)), nil
}
