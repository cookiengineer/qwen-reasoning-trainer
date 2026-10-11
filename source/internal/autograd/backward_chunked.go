package autograd

import (
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// GatedDeltaNetBackwardChunked is the bounded-memory form of
// GatedDeltaNetBackward. Instead of keeping a full state snapshot per token
// (O(nTok*S_v^2) per head), it keeps snapshots only at chunk boundaries
// (O((nTok/chunk)*S_v^2)) and recomputes the within-chunk states during the
// reverse pass. The state-gradient recurrence is row-independent, so this is
// exact: it produces the same gradients as GatedDeltaNetBackward.
func GatedDeltaNetBackwardChunked(q, k, v, g, beta, state, dOut, dNewState *compute.Tensor, chunk int) (dQ, dK, dV, dG, dBeta, dState *compute.Tensor, err error) {
	sv := v.Ne(0)
	h := v.Ne(1)
	nTok := v.Ne(2)
	if q.Ne(0) != sv || k.Ne(0) != sv || q.Ne(1) != h || k.Ne(1) != h {
		return nil, nil, nil, nil, nil, nil, compute.ErrShape
	}
	if state.Ne(0) != sv || state.Ne(1) != sv || state.Ne(2) != h {
		return nil, nil, nil, nil, nil, nil, compute.ErrShape
	}
	kda := g.Ne(0) == sv
	if g.Ne(0) != 1 && !kda {
		return nil, nil, nil, nil, nil, nil, compute.ErrShape
	}
	if chunk < 1 {
		chunk = 1
	}
	gstride := g.Ne(0)
	scale := float64(1 / math.Sqrt(float64(sv)))

	dQ = compute.NewF32(q.Dims...)
	dK = compute.NewF32(k.Dims...)
	dV = compute.NewF32(v.Dims...)
	dG = compute.NewF32(g.Dims...)
	dBeta = compute.NewF32(beta.Dims...)
	dState = compute.NewF32(state.Dims...)

	work := make([]float64, sv*sv)
	decay := make([]float64, sv)
	delta := make([]float64, sv)
	dS := make([]float64, sv*sv)
	dA := make([]float64, sv*sv)
	ddelta := make([]float64, sv)

	// Number of chunks and the within-chunk state ring (chunk+1 entries).
	numChunks := (nTok + chunk - 1) / chunk
	within := make([][]float64, chunk+1)
	for i := range within {
		within[i] = make([]float64, sv*sv)
	}

	for hh := 0; hh < h; hh++ {
		// Forward: keep only the state at the start of each chunk.
		boundaries := make([][]float64, numChunks+1)
		for c := range boundaries {
			boundaries[c] = make([]float64, sv*sv)
		}
		for i := 0; i < sv*sv; i++ {
			work[i] = float64(state.F32[hh*sv*sv+i])
		}
		copy(boundaries[0], work)
		for t := 0; t < nTok; t++ {
			gdnForwardStep(work, q, k, v, g, beta, gstride, sv, h, hh, t, kda, decay, delta)
			if (t+1)%chunk == 0 {
				copy(boundaries[(t+1)/chunk], work)
			}
		}

		// Reverse over chunks.
		for i := 0; i < sv*sv; i++ {
			dS[i] = float64(dNewState.F32[hh*sv*sv+i])
		}
		for c := numChunks - 1; c >= 0; c-- {
			t0 := c * chunk
			t1 := t0 + chunk
			if t1 > nTok {
				t1 = nTok
			}
			// Recompute within-chunk states from the boundary.
			copy(within[0], boundaries[c])
			for t := t0; t < t1; t++ {
				copy(within[t-t0+1], within[t-t0])
				row := within[t-t0+1]
				gdnForwardStep(row, q, k, v, g, beta, gstride, sv, h, hh, t, kda, decay, delta)
			}
			// Reverse within the chunk.
			for t := t1 - 1; t >= t0; t-- {
				gdnBackwardStep(dS, within[t-t0], within[t-t0+1], q, k, v, g, beta, dOut, gstride, sv, h, hh, t, kda, scale,
					dQ, dK, dV, dG, dBeta, dA, ddelta, decay, delta)
			}
		}
		for i := 0; i < sv*sv; i++ {
			dState.F32[hh*sv*sv+i] = float32(dS[i])
		}
	}
	return dQ, dK, dV, dG, dBeta, dState, nil
}

// gdnForwardStep applies one token of the gated delta rule to `work` (row-major
// S_v x S_v state). When record is non-nil, the state is copied into it first.
func gdnForwardStep(work []float64, q, k, v, g, beta *compute.Tensor, gstride, sv, h, hh, t int, kda bool, decay, delta []float64) {
	off := (t*h + hh) * sv
	gOff := (t*h + hh) * gstride
	bOff := t*h + hh
	if kda {
		for i := 0; i < sv; i++ {
			decay[i] = math.Exp(float64(g.F32[gOff+i]))
		}
		for j := 0; j < sv; j++ {
			row := work[j*sv : j*sv+sv]
			for i := 0; i < sv; i++ {
				row[i] *= decay[i]
			}
		}
	} else {
		dec := math.Exp(float64(g.F32[gOff]))
		for i := range work {
			work[i] *= dec
		}
	}
	b := float64(beta.F32[bOff])
	for j := 0; j < sv; j++ {
		var sum float64
		row := work[j*sv : j*sv+sv]
		for i := 0; i < sv; i++ {
			sum += row[i] * float64(k.F32[off+i])
		}
		delta[j] = (float64(v.F32[off+j]) - sum) * b
	}
	for j := 0; j < sv; j++ {
		dj := delta[j]
		row := work[j*sv : j*sv+sv]
		for i := 0; i < sv; i++ {
			row[i] += float64(k.F32[off+i]) * dj
		}
	}
}

// gdnBackwardStep applies one token of the reverse recurrence, consuming St
// (state before token t) and St1 (state after), and accumulating into dS and
// the output gradients.
func gdnBackwardStep(dS, St, St1 []float64, q, k, v, g, beta, dOut *compute.Tensor, gstride, sv, h, hh, t int, kda bool, scale float64,
	dQ, dK, dV, dG, dBeta *compute.Tensor, dA, ddelta, decay, delta []float64) {
	off := (t*h + hh) * sv
	gOff := (t*h + hh) * gstride
	bOff := t*h + hh
	if kda {
		for i := 0; i < sv; i++ {
			decay[i] = math.Exp(float64(g.F32[gOff+i]))
		}
	} else {
		dec := math.Exp(float64(g.F32[gOff]))
		for i := 0; i < sv; i++ {
			decay[i] = dec
		}
	}
	b := float64(beta.F32[bOff])

	for j := 0; j < sv; j++ {
		do := float64(dOut.F32[off+j])
		if do == 0 {
			continue
		}
		for i := 0; i < sv; i++ {
			dS[j*sv+i] += scale * do * float64(q.F32[off+i])
		}
	}
	for i := 0; i < sv; i++ {
		var sum float64
		for j := 0; j < sv; j++ {
			sum += float64(dOut.F32[off+j]) * St1[j*sv+i]
		}
		dQ.F32[off+i] += float32(scale * sum)
	}
	for j := 0; j < sv; j++ {
		var sum float64
		for i := 0; i < sv; i++ {
			sum += St[j*sv+i] * decay[i] * float64(k.F32[off+i])
		}
		delta[j] = (float64(v.F32[off+j]) - sum) * b
	}
	for j := 0; j < sv; j++ {
		for i := 0; i < sv; i++ {
			dA[j*sv+i] = dS[j*sv+i]
			dK.F32[off+i] += float32(dS[j*sv+i] * delta[j])
		}
	}
	for j := 0; j < sv; j++ {
		var sum float64
		for i := 0; i < sv; i++ {
			sum += dS[j*sv+i] * float64(k.F32[off+i])
		}
		ddelta[j] = sum
	}
	var dbetaAcc float64
	for j := 0; j < sv; j++ {
		var r float64
		for i := 0; i < sv; i++ {
			r += St[j*sv+i] * decay[i] * float64(k.F32[off+i])
		}
		r = float64(v.F32[off+j]) - r
		dV.F32[off+j] += float32(b * ddelta[j])
		dbetaAcc += ddelta[j] * r
		for i := 0; i < sv; i++ {
			dA[j*sv+i] -= b * ddelta[j] * float64(k.F32[off+i])
			dK.F32[off+i] -= float32(b * ddelta[j] * St[j*sv+i] * decay[i])
		}
	}
	dBeta.F32[bOff] += float32(dbetaAcc)
	for j := 0; j < sv; j++ {
		for i := 0; i < sv; i++ {
			dS[j*sv+i] = dA[j*sv+i] * decay[i]
		}
	}
	if kda {
		for i := 0; i < sv; i++ {
			var sum float64
			for j := 0; j < sv; j++ {
				sum += dA[j*sv+i] * St[j*sv+i]
			}
			dG.F32[gOff+i] += float32(sum * decay[i])
		}
	} else {
		var sum float64
		for c := 0; c < sv*sv; c++ {
			sum += dA[c] * St[c]
		}
		dG.F32[gOff] += float32(sum * decay[0])
	}
}
