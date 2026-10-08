package autograd

import (
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// MatMulBackward computes gradients of compute.MatMul(a, b) with the GGML
// layout: a [K,N], b [K,M] -> [N,M]. It returns dA and dB given dOut.
func MatMulBackward(a, b, dOut *compute.Tensor) (dA, dB *compute.Tensor) {
	k, n, m := a.Ne(0), a.Ne(1), b.Ne(1)
	dA = compute.NewF32(a.Dims...)
	dB = compute.NewF32(b.Dims...)
	for ni := 0; ni < n; ni++ {
		for mi := 0; mi < m; mi++ {
			g := float64(dOut.F32[ni+mi*n])
			if g == 0 {
				continue
			}
			for ki := 0; ki < k; ki++ {
				dA.F32[ni*k+ki] += float32(g * float64(b.F32[mi*k+ki]))
				dB.F32[mi*k+ki] += float32(g * float64(a.F32[ni*k+ki]))
			}
		}
	}
	return dA, dB
}

// RMSNormBackward computes gradients of compute.RMSNorm with respect to x and
// the per-row weight. weight may be nil. dOut has the same shape as the output.
func RMSNormBackward(x, dOut *compute.Tensor, weight []float32, eps float32) (dX *compute.Tensor, dWeight []float32) {
	row := x.Ne(0)
	rows := len(x.F32) / row
	dX = compute.NewF32(x.Dims...)
	if weight != nil {
		dWeight = make([]float32, row)
	}
	for r := 0; r < rows; r++ {
		base := r * row
		var ss float64
		for i := 0; i < row; i++ {
			v := float64(x.F32[base+i])
			ss += v * v
		}
		mean := ss / float64(row)
		inv := 1 / math.Sqrt(mean+float64(eps))

		var dot float64 // sum_i dyhat_i * x_i
		for i := 0; i < row; i++ {
			w := 1.0
			if weight != nil {
				w = float64(weight[i])
			}
			dyhat := float64(dOut.F32[base+i]) * w
			if weight != nil {
				dWeight[i] += dOut.F32[base+i] * float32(float64(x.F32[base+i])*inv)
			}
			dot += dyhat * float64(x.F32[base+i])
		}
		for i := 0; i < row; i++ {
			w := 1.0
			if weight != nil {
				w = float64(weight[i])
			}
			dyhat := float64(dOut.F32[base+i]) * w
			dX.F32[base+i] = float32(dyhat*inv - (inv*inv*inv/float64(row))*float64(x.F32[base+i])*dot)
		}
	}
	return dX, dWeight
}

// SiluBackward computes the gradient of compute.Silu elementwise.
func SiluBackward(x, dOut *compute.Tensor) *compute.Tensor {
	dx := compute.NewF32(x.Dims...)
	for i, v := range x.F32 {
		s := float64(compute.Sigmoid(v))
		// silu'(x) = sigmoid(x) * (1 + x*(1 - sigmoid(x))).
		dx.F32[i] = float32(float64(dOut.F32[i]) * s * (1 + float64(v)*(1-s)))
	}
	return dx
}

// SoftplusBackward computes the gradient of softplus elementwise. softplus is
// log1p(exp(x)); its derivative is sigmoid(x).
func SoftplusBackward(x, dOut *compute.Tensor) *compute.Tensor {
	dx := compute.NewF32(x.Dims...)
	for i, v := range x.F32 {
		dx.F32[i] = dOut.F32[i] * compute.Sigmoid(v)
	}
	return dx
}

// SigmoidBackward computes the gradient of the sigmoid elementwise, given the
// forward output out = sigmoid(x).
func SigmoidBackward(out, dOut *compute.Tensor) *compute.Tensor {
	dx := compute.NewF32(out.Dims...)
	for i, o := range out.F32 {
		dx.F32[i] = dOut.F32[i] * o * (1 - o)
	}
	return dx
}

// SoftmaxBackward computes the gradient of the softmax given its forward output
// and dOut. out and dOut have identical shape.
func SoftmaxBackward(out, dOut *compute.Tensor) *compute.Tensor {
	row := out.Ne(0)
	rows := len(out.F32) / row
	dx := compute.NewF32(out.Dims...)
	for r := 0; r < rows; r++ {
		base := r * row
		var dot float64
		for i := 0; i < row; i++ {
			dot += float64(dOut.F32[base+i]) * float64(out.F32[base+i])
		}
		for i := 0; i < row; i++ {
			o := float64(out.F32[base+i])
			dx.F32[base+i] = float32(o * (float64(dOut.F32[base+i]) - dot))
		}
	}
	return dx
}

// RoPENeoXBackward computes the gradient of compute.RoPENeoX. RoPE is a fixed
// rotation, so the backward applies the inverse (negative-angle) rotation to
// dOut. x is unused except for its shape; positions, theta and nDims must match
// the forward call.
func RoPENeoXBackward(x, dOut *compute.Tensor, positions []int32, theta float64, nDims int) *compute.Tensor {
	headDim := x.Ne(0)
	nHead := x.Ne(1)
	nTok := x.Ne(2)
	dx := dOut.Clone()
	half := nDims / 2
	for t := 0; t < nTok; t++ {
		pos := float64(positions[t])
		for h := 0; h < nHead; h++ {
			base := (t*nHead + h) * headDim
			for i := 0; i < half; i++ {
				freq := math.Pow(theta, -2*float64(i)/float64(nDims))
				ang := pos * freq
				c, s := math.Cos(ang), math.Sin(ang)
				g0 := float64(dOut.F32[base+i])
				g1 := float64(dOut.F32[base+i+half])
				dx.F32[base+i] = float32(g0*c + g1*s)
				dx.F32[base+i+half] = float32(-g0*s + g1*c)
			}
		}
	}
	return dx
}

// GetRowsBackward scatters the rows of dOut back into a table of nRows rows,
// each rowLen wide, accumulating where an index is repeated.
func GetRowsBackward(rowLen, nRows int, indices []int32, dOut *compute.Tensor) *compute.Tensor {
	dTable := compute.NewF32(rowLen, nRows)
	for idx, row := range indices {
		if row < 0 || int(row) >= nRows {
			continue
		}
		for i := 0; i < rowLen; i++ {
			dTable.F32[int(row)*rowLen+i] += dOut.F32[idx*rowLen+i]
		}
	}
	return dTable
}

// CrossEntropyBackward computes the gradient of the mean cross-entropy loss
// from compute.CrossEntropy with respect to the logits. Ignored targets
// contribute nothing, matching the forward's mean over non-ignored entries.
func CrossEntropyBackward(logits *compute.Tensor, targets []int, ignoreIndex int) *compute.Tensor {
	vocab := logits.Ne(0)
	n := logits.Ne(1)
	dLogits := compute.NewF32(logits.Dims...)
	count := 0
	for t := 0; t < n; t++ {
		if targets[t] == ignoreIndex {
			continue
		}
		count++
	}
	if count == 0 {
		return dLogits
	}
	inv := 1 / float64(count)
	for t := 0; t < n; t++ {
		if targets[t] == ignoreIndex {
			continue
		}
		base := t * vocab
		row := logits.F32[base : base+vocab]
		drow := dLogits.F32[base : base+vocab]
		max := float64(math.Inf(-1))
		for _, v := range row {
			if float64(v) > max {
				max = float64(v)
			}
		}
		var sum float64
		for _, v := range row {
			sum += math.Exp(float64(v) - max)
		}
		tgt := targets[t]
		for i := 0; i < vocab; i++ {
			p := math.Exp(float64(row[i])-max) / sum
			if i == tgt {
				p -= 1
			}
			drow[i] = float32(p * inv)
		}
	}
	return dLogits
}
