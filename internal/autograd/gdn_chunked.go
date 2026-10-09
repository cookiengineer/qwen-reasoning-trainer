package autograd

import (
	"math"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

// This file implements the chunked (FLA-style) gated delta rule forward and
// backward as a self-contained float64 reference. It matches
// compute.GatedDeltaNet exactly and exists so the device kernels can be ported
// against a structurally identical oracle instead of the naive per-row
// recurrence (which is sequential over the whole sequence and uses atomics for
// the cross-row reductions).
//
// Layout (identical to compute.GatedDeltaNet):
//   q,k,v   [S_v, H, T]
//   g       [gstride, H, T]  (gstride 1 = scalar gate, S_v = per-dim / KDA)
//   beta    [1, H, T]
//   state   [S_v, S_v, H]
//   out     [S_v, H, T]
//   newState[S_v, S_v, H]
//
// Per head and per state row j the recurrence is
//   a_t      = D_t (*) S_{t-1}[j,:]        D_t = exp(g_t)
//   kv_t     = a_t . k_t
//   delta_t  = beta_t (v_t[j] - kv_t)
//   S_t[j,:] = a_t + k_t * delta_t
//   out_t[j] = scale * S_t[j,:] . q_t      scale = 1/sqrt(S_v)
//
// Chunked form (chunk of C tokens, local index r, cum G_r = sum_{s<=r} g_s):
//   M delta = rhs,  M[r,r]=1, M[r,p]=beta_r A_{p,r} (p<r),
//   A_{p,r} = sum_i exp(G_r[i]-G_p[i]) k_p[i] k_r[i],
//   rhs_r[j]= beta_r ( v_r[j] - (S_c . (E_r (*) k_r))[j] ),  E_r=exp(G_r),
//   out_r[j]= scale ( (S_c . q~_r)[j] + sum_{p<=r} B_{r,p} delta_p[j] ),
//   B_{r,p} = sum_i exp(G_r[i]-G_p[i]) k_p[i] q_r[i],
//   S_{c+1} = e (*) S_c + sum_p (w_p (x) delta_p),
//   e[i] = exp(G_{C-1}[i]), w_p[i] = exp(G_{C-1}[i]-G_p[i]) k_p[i].

type gdnChunkCache struct {
	sv, h, nTok, c, nChunks int
	kda                     bool
	gstride                 int
	scale                   float64
	boundaries              []float64 // [h*(nChunks+1)*sv*sv]
	deltas                  []float64 // [h*nChunks*C*sv]
}

// GatedDeltaNetChunkedForward is the chunked forward; it matches
// compute.GatedDeltaNet.
func GatedDeltaNetChunkedForward(q, k, v, g, beta, state *compute.Tensor, chunk int) (*compute.Tensor, *compute.Tensor, error) {
	out, ns, _, err := gdnChunkedForwardCached(q, k, v, g, beta, state, chunk)
	return out, ns, err
}

// GatedDeltaNetChunked computes the chunked forward and backward. It returns the
// same gradients as GatedDeltaNetBackward.
func GatedDeltaNetChunked(q, k, v, g, beta, state, dOut, dNewState *compute.Tensor, chunk int) (dQ, dK, dV, dG, dBeta, dState *compute.Tensor, err error) {
	_, _, ca, err := gdnChunkedForwardCached(q, k, v, g, beta, state, chunk)
	if err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	return gdnChunkedBackward(q, k, v, g, beta, dOut, dNewState, ca)
}

func gdnChunkedForwardCached(q, k, v, g, beta, state *compute.Tensor, chunk int) (*compute.Tensor, *compute.Tensor, *gdnChunkCache, error) {
	sv := v.Ne(0)
	h := v.Ne(1)
	nTok := v.Ne(2)
	if q.Ne(0) != sv || k.Ne(0) != sv || q.Ne(1) != h || k.Ne(1) != h {
		return nil, nil, nil, compute.ErrShape
	}
	if state.Ne(0) != sv || state.Ne(1) != sv || state.Ne(2) != h {
		return nil, nil, nil, compute.ErrShape
	}
	kda := g.Ne(0) == sv
	if g.Ne(0) != 1 && !kda {
		return nil, nil, nil, compute.ErrShape
	}
	if chunk < 1 {
		chunk = 1
	}
	gstride := g.Ne(0)
	scale := 1 / math.Sqrt(float64(sv))
	C := chunk
	nChunks := (nTok + C - 1) / C

	out := compute.NewF32(sv, h, nTok)
	newState := compute.NewF32(sv, sv, h)
	ca := &gdnChunkCache{
		sv: sv, h: h, nTok: nTok, c: C, nChunks: nChunks,
		kda: kda, gstride: gstride, scale: scale,
		boundaries: make([]float64, h*(nChunks+1)*sv*sv),
		deltas:     make([]float64, h*nChunks*C*sv),
	}

	gAt := func(t, hh, i int) float64 {
		if kda {
			return float64(g.F32[(t*h+hh)*gstride+i])
		}
		return float64(g.F32[(t*h+hh)*gstride])
	}
	idx := func(t, hh, i int) int { return (t*h+hh)*sv + i }

	Gcum := make([]float64, C*sv)
	E := make([]float64, C*sv)
	A := make([]float64, C*C)  // A[p*C+r] = A_{p,r}
	Bt := make([]float64, C*C) // Bt[r*C+p] = B_{r,p}
	delta := make([]float64, C*sv)
	rhs := make([]float64, sv)
	kr := make([]float64, sv)
	qr := make([]float64, sv)

	for hh := 0; hh < h; hh++ {
		sbase := hh * sv * sv
		f64cpy(ca.boundaries[(hh*(nChunks+1))*sv*sv:], state.F32[sbase:sbase+sv*sv])
		for c := 0; c < nChunks; c++ {
			t0 := c * C
			clen := C
			if t0+clen > nTok {
				clen = nTok - t0
			}
			Sc := ca.boundaries[((hh*(nChunks+1))+c)*sv*sv:]
			for i := 0; i < sv; i++ {
				Gcum[i] = 0
			}
			for r := 0; r < clen; r++ {
				for i := 0; i < sv; i++ {
					prev := 0.0
					if r > 0 {
						prev = Gcum[(r-1)*sv+i]
					}
					Gcum[r*sv+i] = prev + gAt(t0+r, hh, i)
					E[r*sv+i] = math.Exp(Gcum[r*sv+i])
				}
			}
			for r := 0; r < clen; r++ {
				for i := 0; i < sv; i++ {
					kr[i] = float64(k.F32[idx(t0+r, hh, i)])
					qr[i] = float64(q.F32[idx(t0+r, hh, i)])
				}
				for p := 0; p < r; p++ {
					var a float64
					for i := 0; i < sv; i++ {
						cexp := math.Exp(Gcum[r*sv+i] - Gcum[p*sv+i])
						a += cexp * float64(k.F32[idx(t0+p, hh, i)]) * kr[i]
					}
					A[p*C+r] = a
				}
				for p := 0; p <= r; p++ {
					var bd float64
					for i := 0; i < sv; i++ {
						cexp := math.Exp(Gcum[r*sv+i] - Gcum[p*sv+i])
						bd += cexp * float64(k.F32[idx(t0+p, hh, i)]) * qr[i]
					}
					Bt[r*C+p] = bd
				}
			}
			for r := 0; r < clen; r++ {
				b := float64(beta.F32[(t0+r)*h+hh])
				for i := 0; i < sv; i++ {
					kr[i] = float64(k.F32[idx(t0+r, hh, i)])
				}
				for j := 0; j < sv; j++ {
					var s float64
					for i := 0; i < sv; i++ {
						s += Sc[j*sv+i] * E[r*sv+i] * kr[i]
					}
					rhs[j] = b * (float64(v.F32[idx(t0+r, hh, j)]) - s)
				}
				for p := 0; p < r; p++ {
					m := b * A[p*C+r]
					for j := 0; j < sv; j++ {
						rhs[j] -= m * delta[p*sv+j]
					}
				}
				copy(delta[r*sv:(r+1)*sv], rhs)
			}
			copy(ca.deltas[((hh*nChunks)+c)*C*sv:], delta[:clen*sv])
			for r := 0; r < clen; r++ {
				for i := 0; i < sv; i++ {
					qr[i] = float64(q.F32[idx(t0+r, hh, i)])
				}
				for j := 0; j < sv; j++ {
					var acc float64
					for i := 0; i < sv; i++ {
						acc += Sc[j*sv+i] * E[r*sv+i] * qr[i]
					}
					for p := 0; p <= r; p++ {
						acc += Bt[r*C+p] * delta[p*sv+j]
					}
					out.F32[idx(t0+r, hh, j)] = float32(scale * acc)
				}
			}
			last := clen - 1
			nb := ca.boundaries[((hh*(nChunks+1))+c+1)*sv*sv:]
			for j := 0; j < sv; j++ {
				for i := 0; i < sv; i++ {
					acc := E[last*sv+i] * Sc[j*sv+i]
					for p := 0; p < clen; p++ {
						kpi := float64(k.F32[idx(t0+p, hh, i)])
						w := math.Exp(Gcum[last*sv+i]-Gcum[p*sv+i]) * kpi
						acc += w * delta[p*sv+j]
					}
					nb[j*sv+i] = acc
				}
			}
		}
		lb := ((hh * (nChunks + 1)) + nChunks) * sv * sv
		f32cpy(newState.F32[sbase:sbase+sv*sv], ca.boundaries[lb:lb+sv*sv])
	}
	return out, newState, ca, nil
}

func gdnChunkedBackward(q, k, v, g, beta, dOut, dNewState *compute.Tensor, ca *gdnChunkCache) (dQ, dK, dV, dG, dBeta, dState *compute.Tensor, err error) {
	sv := ca.sv
	h := ca.h
	nTok := ca.nTok
	C := ca.c
	nChunks := ca.nChunks
	kda := ca.kda
	gstride := ca.gstride
	scale := ca.scale

	dQ = compute.NewF32(q.Dims...)
	dK = compute.NewF32(k.Dims...)
	dV = compute.NewF32(v.Dims...)
	dG = compute.NewF32(g.Dims...)
	dBeta = compute.NewF32(beta.Dims...)
	dState = compute.NewF32(sv, sv, h)

	gAt := func(t, hh, i int) float64 {
		if kda {
			return float64(g.F32[(t*h+hh)*gstride+i])
		}
		return float64(g.F32[(t*h+hh)*gstride])
	}
	idx := func(t, hh, i int) int { return (t*h+hh)*sv + i }

	Gcum := make([]float64, C*sv)
	E := make([]float64, C*sv)
	A := make([]float64, C*C)
	Bt := make([]float64, C*C)
	delta := make([]float64, C*sv)
	dA := make([]float64, C*C)
	dBt := make([]float64, C*C)
	dw := make([]float64, C*sv)
	de := make([]float64, sv)
	dqtil := make([]float64, C*sv)
	dktil := make([]float64, C*sv)
	dGc := make([]float64, C*sv)
	ddelta := make([]float64, C*sv)
	dkr := make([]float64, C*sv)
	dqr := make([]float64, C*sv)
	dS := make([]float64, sv*sv)
	dSc := make([]float64, sv*sv)
	zsol := make([]float64, C*sv)
	do := make([]float64, sv)

	for hh := 0; hh < h; hh++ {
		sbase := hh * sv * sv
		f64cpy(dS, dNewState.F32[sbase:sbase+sv*sv])
		for c := nChunks - 1; c >= 0; c-- {
			t0 := c * C
			clen := C
			if t0+clen > nTok {
				clen = nTok - t0
			}
			Sc := ca.boundaries[((hh*(nChunks+1))+c)*sv*sv:]
			copy(delta, ca.deltas[((hh*nChunks)+c)*C*sv:((hh*nChunks)+c)*C*sv+clen*sv])
			zeroF(dA)
			zeroF(dBt)
			zeroF(dw)
			zeroF(de)
			zeroF(dqtil)
			zeroF(dktil)
			zeroF(dGc)
			zeroF(ddelta)
			zeroF(dkr)
			zeroF(dqr)
			zeroF(dSc)
			// Recompute Gcum/E/A/B for the chunk.
			for r := 0; r < clen; r++ {
				for i := 0; i < sv; i++ {
					prev := 0.0
					if r > 0 {
						prev = Gcum[(r-1)*sv+i]
					}
					Gcum[r*sv+i] = prev + gAt(t0+r, hh, i)
					E[r*sv+i] = math.Exp(Gcum[r*sv+i])
				}
			}
			for r := 0; r < clen; r++ {
				for p := 0; p < r; p++ {
					var a float64
					for i := 0; i < sv; i++ {
						cexp := math.Exp(Gcum[r*sv+i] - Gcum[p*sv+i])
						a += cexp * float64(k.F32[idx(t0+p, hh, i)]) * float64(k.F32[idx(t0+r, hh, i)])
					}
					A[p*C+r] = a
				}
				for p := 0; p <= r; p++ {
					var bd float64
					for i := 0; i < sv; i++ {
						cexp := math.Exp(Gcum[r*sv+i] - Gcum[p*sv+i])
						bd += cexp * float64(k.F32[idx(t0+p, hh, i)]) * float64(q.F32[idx(t0+r, hh, i)])
					}
					Bt[r*C+p] = bd
				}
			}
			last := clen - 1
			// (a) Chunk-end state.
			for i := 0; i < sv; i++ {
				e := E[last*sv+i]
				for j := 0; j < sv; j++ {
					de[i] += dS[j*sv+i] * Sc[j*sv+i]
					dSc[j*sv+i] += e * dS[j*sv+i]
				}
			}
			for p := 0; p < clen; p++ {
				for j := 0; j < sv; j++ {
					var acc float64
					for i := 0; i < sv; i++ {
						w := math.Exp(Gcum[last*sv+i]-Gcum[p*sv+i]) * float64(k.F32[idx(t0+p, hh, i)])
						acc += dS[j*sv+i] * w
					}
					ddelta[p*sv+j] += acc
				}
				for i := 0; i < sv; i++ {
					var acc float64
					for j := 0; j < sv; j++ {
						acc += dS[j*sv+i] * delta[p*sv+j]
					}
					dw[p*sv+i] += acc
				}
			}
			// (b) Outputs.
			for r := 0; r < clen; r++ {
				for j := 0; j < sv; j++ {
					do[j] = float64(dOut.F32[idx(t0+r, hh, j)])
				}
				for i := 0; i < sv; i++ {
					var acc float64
					for j := 0; j < sv; j++ {
						acc += do[j] * Sc[j*sv+i]
					}
					dqtil[r*sv+i] += scale * acc
				}
				for j := 0; j < sv; j++ {
					for i := 0; i < sv; i++ {
						dSc[j*sv+i] += scale * do[j] * E[r*sv+i] * float64(q.F32[idx(t0+r, hh, i)])
					}
				}
				for p := 0; p <= r; p++ {
					var s float64
					for j := 0; j < sv; j++ {
						s += do[j] * delta[p*sv+j]
					}
					dBt[r*C+p] += scale * s
					for j := 0; j < sv; j++ {
						ddelta[p*sv+j] += scale * Bt[r*C+p] * do[j]
					}
				}
			}
			// (c) Solve M^T z = ddelta and accumulate dM. zsol holds the solved
			// adjoint vector per local index (the r index is coupled by M).
			for r := clen - 1; r >= 0; r-- {
				copy(zsol[r*sv:r*sv+sv], ddelta[r*sv:r*sv+sv])
				for p := r + 1; p < clen; p++ {
					m := float64(beta.F32[(t0+p)*h+hh]) * A[r*C+p]
					for j := 0; j < sv; j++ {
						zsol[r*sv+j] -= m * zsol[p*sv+j]
					}
				}
				br := float64(beta.F32[(t0+r)*h+hh])
				for p := 0; p < r; p++ {
					var dm float64
					for j := 0; j < sv; j++ {
						dm -= zsol[r*sv+j] * delta[p*sv+j]
					}
					dA[p*C+r] += br * dm
					dBeta.F32[(t0+r)*h+hh] += float32(dm * A[p*C+r])
				}
				// (d) rhs from step 6.
				for j := 0; j < sv; j++ {
					dV.F32[idx(t0+r, hh, j)] += float32(br * zsol[r*sv+j])
					var s float64
					for i := 0; i < sv; i++ {
						s += Sc[j*sv+i] * E[r*sv+i] * float64(k.F32[idx(t0+r, hh, i)])
					}
					dBeta.F32[(t0+r)*h+hh] += float32(zsol[r*sv+j] * (float64(v.F32[idx(t0+r, hh, j)]) - s))
				}
				for i := 0; i < sv; i++ {
					var acc float64
					for j := 0; j < sv; j++ {
						acc += zsol[r*sv+j] * Sc[j*sv+i]
					}
					dktil[r*sv+i] += -br * acc
				}
				for j := 0; j < sv; j++ {
					for i := 0; i < sv; i++ {
						dSc[j*sv+i] -= br * zsol[r*sv+j] * E[r*sv+i] * float64(k.F32[idx(t0+r, hh, i)])
					}
				}
			}
			// (e) A matrix -> k, G.
			for r := 0; r < clen; r++ {
				for p := 0; p < r; p++ {
					dap := dA[p*C+r]
					if dap == 0 {
						continue
					}
					for i := 0; i < sv; i++ {
						cexp := math.Exp(Gcum[r*sv+i] - Gcum[p*sv+i])
						kpi := float64(k.F32[idx(t0+p, hh, i)])
						kri := float64(k.F32[idx(t0+r, hh, i)])
						dkr[p*sv+i] += dap * cexp * kri
						dkr[r*sv+i] += dap * cexp * kpi
						hval := dap * cexp * kpi * kri
						dGc[r*sv+i] += hval
						dGc[p*sv+i] -= hval
					}
				}
			}
			// (f) B matrix -> k, q, G.
			for r := 0; r < clen; r++ {
				for p := 0; p <= r; p++ {
					dbp := dBt[r*C+p]
					if dbp == 0 {
						continue
					}
					for i := 0; i < sv; i++ {
						cexp := math.Exp(Gcum[r*sv+i] - Gcum[p*sv+i])
						kpi := float64(k.F32[idx(t0+p, hh, i)])
						qri := float64(q.F32[idx(t0+r, hh, i)])
						dkr[p*sv+i] += dbp * cexp * qri
						dqr[r*sv+i] += dbp * cexp * kpi
						hval := dbp * cexp * kpi * qri
						dGc[r*sv+i] += hval
						dGc[p*sv+i] -= hval
					}
				}
			}
			// (g) w -> k, G.
			for p := 0; p < clen; p++ {
				for i := 0; i < sv; i++ {
					if dw[p*sv+i] == 0 {
						continue
					}
					cexp := math.Exp(Gcum[last*sv+i] - Gcum[p*sv+i])
					kpi := float64(k.F32[idx(t0+p, hh, i)])
					dkr[p*sv+i] += dw[p*sv+i] * cexp
					hval := dw[p*sv+i] * cexp * kpi
					dGc[last*sv+i] += hval
					dGc[p*sv+i] -= hval
				}
			}
			// (h) E -> k, q, G.
			for r := 0; r < clen; r++ {
				for i := 0; i < sv; i++ {
					er := E[r*sv+i]
					dkr[r*sv+i] += dktil[r*sv+i] * er
					dqr[r*sv+i] += dqtil[r*sv+i] * er
					dGc[r*sv+i] += (dktil[r*sv+i]*float64(k.F32[idx(t0+r, hh, i)]) +
						dqtil[r*sv+i]*float64(q.F32[idx(t0+r, hh, i)])) * er
				}
			}
			// (i) e -> G (last local position).
			for i := 0; i < sv; i++ {
				dGc[last*sv+i] += de[i] * E[last*sv+i]
			}
			// Write local k/q grads.
			for r := 0; r < clen; r++ {
				for i := 0; i < sv; i++ {
					dK.F32[idx(t0+r, hh, i)] += float32(dkr[r*sv+i])
					dQ.F32[idx(t0+r, hh, i)] += float32(dqr[r*sv+i])
				}
			}
			// (j) cumsum -> g.
			for s := 0; s < clen; s++ {
				if kda {
					for i := 0; i < sv; i++ {
						var acc float64
						for r := s; r < clen; r++ {
							acc += dGc[r*sv+i]
						}
						dG.F32[(t0+s)*h*gstride+hh*gstride+i] += float32(acc)
					}
				} else {
					var acc float64
					for i := 0; i < sv; i++ {
						for r := s; r < clen; r++ {
							acc += dGc[r*sv+i]
						}
					}
					dG.F32[(t0+s)*h+hh] += float32(acc)
				}
			}
			// Propagate start-state gradient to the previous chunk.
			copy(dS, dSc)
		}
		f32cpy(dState.F32[sbase:sbase+sv*sv], dS)
	}
	return dQ, dK, dV, dG, dBeta, dState, nil
}

func zeroF(x []float64) {
	for i := range x {
		x[i] = 0
	}
}

func f64cpy(dst []float64, src []float32) {
	for i := range src {
		dst[i] = float64(src[i])
	}
}

func f32cpy(dst []float32, src []float64) {
	for i := range src {
		dst[i] = float32(src[i])
	}
}
