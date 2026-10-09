package autograd

import (
	"math"
	"testing"

	"github.com/cookiengineer/qwen-reasoning-trainer/internal/compute"
)

func testGDNTensors(seed uint64, sv, h, nTok int, kda bool) (q, k, v, g, beta, state *compute.Tensor) {
	q = Random(seed, 1.0, sv, h, nTok)
	k = Random(seed+1, 1.0, sv, h, nTok)
	v = Random(seed+2, 1.0, sv, h, nTok)
	if kda {
		g = Random(seed+3, 0.3, sv, h, nTok)
	} else {
		g = Random(seed+3, 0.3, 1, h, nTok)
	}
	beta = Random(seed+4, 0.5, 1, h, nTok)
	state = Random(seed+5, 0.5, sv, sv, h)
	return
}

func maxAbs(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > m {
			m = d
		}
	}
	return m
}

func TestChunkedForwardMatches(t *testing.T) {
	for _, kda := range []bool{false, true} {
		sv, h, nTok := 4, 3, 19
		q, k, v, g, beta, state := testGDNTensors(10, sv, h, nTok, kda)
		wantOut, wantState, err := compute.GatedDeltaNet(q, k, v, g, beta, state)
		if err != nil {
			t.Fatal(err)
		}
		for _, chunk := range []int{1, 2, 3, 4, 7, 19, 32} {
			out, ns, err := GatedDeltaNetChunkedForward(q, k, v, g, beta, state, chunk)
			if err != nil {
				t.Fatalf("chunk=%d: %v", chunk, err)
			}
			if d := maxAbs(out.F32, wantOut.F32); d > 1e-5 {
				t.Fatalf("kda=%v chunk=%d out max diff %g", kda, chunk, d)
			}
			if d := maxAbs(ns.F32, wantState.F32); d > 1e-5 {
				t.Fatalf("kda=%v chunk=%d state max diff %g", kda, chunk, d)
			}
		}
	}
}

func TestChunkedBackwardMatches(t *testing.T) {
	for _, kda := range []bool{false, true} {
		sv, h, nTok := 4, 3, 19
		q, k, v, g, beta, state := testGDNTensors(20, sv, h, nTok, kda)
		dOut := Random(30, 1.0, sv, h, nTok)
		dNew := Random(31, 1.0, sv, sv, h)

		rq, rk, rv, rg, rb, rs, err := GatedDeltaNetBackward(q, k, v, g, beta, state, dOut, dNew)
		if err != nil {
			t.Fatal(err)
		}
		for _, chunk := range []int{1, 2, 3, 5, 19} {
			cq, ck, cv, cg, cb, cs, err := GatedDeltaNetChunked(q, k, v, g, beta, state, dOut, dNew, chunk)
			if err != nil {
				t.Fatalf("chunk=%d: %v", chunk, err)
			}
			for _, tc := range []struct {
				name string
				got  []float32
				want []float32
			}{
				{"dQ", cq.F32, rq.F32}, {"dK", ck.F32, rk.F32}, {"dV", cv.F32, rv.F32},
				{"dG", cg.F32, rg.F32}, {"dBeta", cb.F32, rb.F32}, {"dState", cs.F32, rs.F32},
			} {
				if d := maxAbs(tc.got, tc.want); d > 1e-4 {
					t.Fatalf("kda=%v chunk=%d %s max diff %g", kda, chunk, tc.name, d)
				}
			}
		}
	}
}

func TestChunkedBackwardGradCheck(t *testing.T) {
	for _, kda := range []bool{false, true} {
		sv, h, nTok, chunk := 3, 2, 6, 3
		q, k, v, g, beta, state := testGDNTensors(40, sv, h, nTok, kda)
		inputs := []*compute.Tensor{q, k, v, g, beta, state}
		objective := func(xs []*compute.Tensor) float64 {
			out, ns, err := GatedDeltaNetChunkedForward(xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], chunk)
			if err != nil {
				t.Fatal(err)
			}
			var s float64
			for _, f := range out.F32 {
				s += float64(f)
			}
			for _, f := range ns.F32 {
				s += 0.5 * float64(f)
			}
			return s
		}
		// dOut/dNewState for d(sum(out)+0.5*sum(state)) are all 1.0/0.5.
		dOut := Ones(sv, h, nTok)
		dNew := Ones(sv, sv, h)
		for i := range dNew.F32 {
			dNew.F32[i] = 0.5
		}
		dq, dk, dv, dg, db, ds, err := GatedDeltaNetChunked(q, k, v, g, beta, state, dOut, dNew, chunk)
		if err != nil {
			t.Fatal(err)
		}
		analytic := []*compute.Tensor{dq, dk, dv, dg, db, ds}
		if err := GradCheckTol(inputs, analytic, objective, 1e-3, 3e-2); err != nil {
			t.Fatalf("kda=%v: %v", kda, err)
		}
	}
}
