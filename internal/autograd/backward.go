package autograd

import (
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// L2NormBackward computes the gradient of compute.L2Norm with respect to x.
// x and dOut have the same shape; every contiguous row of length Dims[0] is
// normalized to unit L2 norm.
func L2NormBackward(x, dOut *compute.Tensor, eps float32) *compute.Tensor {
	row := x.Ne(0)
	rows := len(x.F32) / row
	dx := compute.NewF32(x.Dims...)
	for r := 0; r < rows; r++ {
		base := r * row
		var ss float64
		for i := 0; i < row; i++ {
			v := float64(x.F32[base+i])
			ss += v * v
		}
		inv := 1 / math.Sqrt(ss+float64(eps))
		// out_i = x_i * inv; c = dOut . out; dx_i = inv * (dOut_i - c * out_i).
		var c float64
		for i := 0; i < row; i++ {
			c += float64(dOut.F32[base+i]) * float64(x.F32[base+i]) * inv
		}
		for i := 0; i < row; i++ {
			out := float64(x.F32[base+i]) * inv
			dx.F32[base+i] = float32(inv * (float64(dOut.F32[base+i]) - c*out))
		}
	}
	return dx
}

// SSMConvBackward computes gradients of compute.SSMConv with respect to sx and
// c. It mirrors the forward's indexing exactly, including its sequence handling
// (the model only ever calls it with nSeq == 1).
func SSMConvBackward(sx, c, dOut *compute.Tensor) (dSx, dC *compute.Tensor, err error) {
	dConv := c.Ne(0)
	dInner := c.Ne(1)
	ncs := sx.Ne(0)
	nT := ncs - dConv + 1
	nSeq := sx.Ne(2)
	if sx.Ne(1) != dInner || nT < 0 {
		return nil, nil, compute.ErrShape
	}
	dSx = compute.NewF32(sx.Dims...)
	dC = compute.NewF32(c.Dims...)
	for s := 0; s < nSeq; s++ {
		for t := 0; t < nT; t++ {
			for ch := 0; ch < dInner; ch++ {
				g := float64(dOut.F32[(s*nT+t)*dInner+ch])
				if g == 0 {
					continue
				}
				sOff := ch*ncs + t
				cOff := ch * dConv
				for i := 0; i < dConv; i++ {
					dSx.F32[sOff+i] += float32(g * float64(c.F32[cOff+i]))
					dC.F32[cOff+i] += float32(g * float64(sx.F32[sOff+i]))
				}
			}
		}
	}
	return dSx, dC, nil
}

// GatedDeltaNetBackward computes gradients of compute.GatedDeltaNet with respect
// to q, k, v, g, beta and the incoming state.
//
// Inputs and their layouts match the forward (see compute.GatedDeltaNet):
// q,k,v [S_v,nHead,nTok], g [1 or S_v,nHead,nTok], beta [1,nHead,nTok],
// state [S_v,S_v,nHead]. dOut and dNewState are the upstream gradients of the
// returned output and new state.
//
// This is the naive recurrent backward: it stores one full state snapshot per
// token (O(nTok*S_v^2) per head) and scans in reverse. A chunked formulation
// with bounded memory will replace it for the device path; this version is the
// correctness oracle.
func GatedDeltaNetBackward(q, k, v, g, beta, state, dOut, dNewState *compute.Tensor) (dQ, dK, dV, dG, dBeta, dState *compute.Tensor, err error) {
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
	if beta.Ne(0) != 1 {
		return nil, nil, nil, nil, nil, nil, compute.ErrShape
	}
	gstride := g.Ne(0)
	scale := float64(1 / math.Sqrt(float64(sv)))

	dQ = compute.NewF32(q.Dims...)
	dK = compute.NewF32(k.Dims...)
	dV = compute.NewF32(v.Dims...)
	dG = compute.NewF32(g.Dims...)
	dBeta = compute.NewF32(beta.Dims...)
	dState = compute.NewF32(state.Dims...)

	snap := make([][]float64, nTok+1)
	for t := range snap {
		snap[t] = make([]float64, sv*sv)
	}
	work := make([]float64, sv*sv)
	decay := make([]float64, sv)
	delta := make([]float64, sv)
	dS := make([]float64, sv*sv)
	dA := make([]float64, sv*sv)
	ddelta := make([]float64, sv)

	for hh := 0; hh < h; hh++ {
		// Forward pass: snapshot the state at the start of every token.
		for i := 0; i < sv*sv; i++ {
			work[i] = float64(state.F32[hh*sv*sv+i])
		}
		copy(snap[0], work)
		for t := 0; t < nTok; t++ {
			off := (t*h + hh) * sv
			gOff := (t*h + hh) * gstride
			bOff := (t*h + hh)
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
			copy(snap[t+1], work)
		}

		// Reverse pass.
		for i := 0; i < sv*sv; i++ {
			dS[i] = float64(dNewState.F32[hh*sv*sv+i])
		}
		for t := nTok - 1; t >= 0; t-- {
			off := (t*h + hh) * sv
			gOff := (t*h + hh) * gstride
			bOff := (t*h + hh)
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

			// out_j = scale * sum_i S_{t+1}[j][i] q_i.
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
					sum += float64(dOut.F32[off+j]) * snap[t+1][j*sv+i]
				}
				dQ.F32[off+i] += float32(scale * sum)
			}

			// delta_j = beta * (v_j - sum_i A[j][i] k_i), A = S_t * decay.
			for j := 0; j < sv; j++ {
				var sum float64
				for i := 0; i < sv; i++ {
					sum += snap[t][j*sv+i] * decay[i] * float64(k.F32[off+i])
				}
				delta[j] = (float64(v.F32[off+j]) - sum) * b
			}

			// S_{t+1} = A + k x delta.
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
					r += snap[t][j*sv+i] * decay[i] * float64(k.F32[off+i])
				}
				r = float64(v.F32[off+j]) - r
				dV.F32[off+j] += float32(b * ddelta[j])
				dbetaAcc += ddelta[j] * r
				for i := 0; i < sv; i++ {
					dA[j*sv+i] -= b * ddelta[j] * float64(k.F32[off+i])
					dK.F32[off+i] -= float32(b * ddelta[j] * snap[t][j*sv+i] * decay[i])
				}
			}
			dBeta.F32[bOff] += float32(dbetaAcc)

			// A = S_t * decay.
			for j := 0; j < sv; j++ {
				for i := 0; i < sv; i++ {
					dS[j*sv+i] = dA[j*sv+i] * decay[i]
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
			dState.F32[hh*sv*sv+i] = float32(dS[i])
		}
	}
	return dQ, dK, dV, dG, dBeta, dState, nil
}
