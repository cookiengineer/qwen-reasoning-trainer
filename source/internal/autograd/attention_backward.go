package autograd

import (
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// AttentionBackward computes gradients of compute.Attention with grouped-query
// support and optional causality. q has Dims [headDim,nHead,nQ]; k, v have
// [headDim,nHeadKV,nKV]. out and dOut have the same shape as q's head layout.
// The forward must be the same call (same nHead, nHeadKV, scale, causal).
func AttentionBackward(q, k, v, dOut *compute.Tensor, nHead, nHeadKV int, scale float32, causal bool) (dQ, dK, dV *compute.Tensor) {
	headDim := q.Ne(0)
	nQ := q.Ne(2)
	nKV := k.Ne(2)
	group := nHead / nHeadKV

	dQ = compute.NewF32(q.Dims...)
	dK = compute.NewF32(k.Dims...)
	dV = compute.NewF32(v.Dims...)

	logits := make([]float64, nKV)
	dp := make([]float64, nKV)

	for h := 0; h < nHead; h++ {
		kv := h / group
		for i := 0; i < nQ; i++ {
			qOff := (i*nHead + h) * headDim
			maxJ := nKV
			if causal && nQ == nKV {
				maxJ = i + 1
			}

			// Recompute the attention probabilities.
			best := math.Inf(-1)
			for j := 0; j < maxJ; j++ {
				kOff := (j*nHeadKV + kv) * headDim
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += float64(q.F32[qOff+d]) * float64(k.F32[kOff+d])
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

			// dP_j = dOut_i . v_j, and its probability-weighted mean.
			var pd float64
			for j := 0; j < maxJ; j++ {
				off := (j*nHeadKV + kv) * headDim
				var acc float64
				for d := 0; d < headDim; d++ {
					acc += float64(dOut.F32[qOff+d]) * float64(v.F32[off+d])
				}
				dp[j] = acc
				pd += logits[j] * acc
			}

			for j := 0; j < maxJ; j++ {
				off := (j*nHeadKV + kv) * headDim
				dDot := logits[j] * (dp[j] - pd) * float64(scale)
				for d := 0; d < headDim; d++ {
					dQ.F32[qOff+d] += float32(dDot * float64(k.F32[off+d]))
					dK.F32[off+d] += float32(dDot * float64(q.F32[qOff+d]))
					dV.F32[off+d] += float32(logits[j] * float64(dOut.F32[qOff+d]))
				}
			}
		}
	}
	return dQ, dK, dV
}
